from __future__ import annotations

import csv
import json
import re
import signal
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from app.config import get_config
from app.db import utc_now_iso
from app.market.price_provider import write_prices_for_run
from app.valuation.evidence_sufficiency import write_evidence_sufficiency_for_run
from app.valuation.facts import resolve_financial_facts_asof
from app.valuation.graham_dodd import (
    REASON_OK as GD_REASON_OK,
    STATUS_OK as GD_STATUS_OK,
    compute_graham_dodd_overlay,
)
from app.valuation.net_debt import resolve_net_debt_proxy, write_net_debt_coverage_for_run
from app.valuation.owner_earnings import (
    DEFAULT_MAINT_CAPEX_RATIO,
    REASON_INSUFFICIENT_HISTORY,
    REASON_MISSING_CAPEX,
    REASON_MISSING_CFO,
    REASON_NEGATIVE_CFO,
    REASON_NEGATIVE_OWNER_EARNINGS,
    compute_owner_earnings_series,
)
from app.valuation.owner_earnings import (
    _annual_series_for_priority as _annual_only_series_for_priority,
)
from app.valuation.owner_earnings_quality import write_owner_earnings_quality_for_run, compute_owner_earnings_quality
from app.valuation.maintenance_capex_discipline import (
    compute_maintenance_capex_discipline,
    write_maintenance_capex_discipline_for_run,
)
from app.valuation.accounting_quality import (
    compute_accounting_quality,
    write_accounting_quality_for_run,
)
from app.valuation.balance_sheet_stress import (
    CASHFLOW_UNITS_USD,
    compute_balance_sheet_stress,
    write_balance_sheet_stress_for_run,
)
from app.valuation.returns_persistence import (
    compute_returns_persistence,
    write_returns_persistence_for_run,
)
from app.valuation.revenue_dependence import (
    compute_revenue_dependence,
    write_revenue_dependence_for_run,
)
from app.valuation.intangible_economics import (
    compute_intangible_economics,
    write_intangible_economics_for_run,
)
from app.valuation.investment_readiness import write_investment_readiness_for_run
from app.valuation.intrinsic_discipline import (
    compute_intrinsic_discipline,
    write_intrinsic_discipline_for_run,
)
from app.valuation.valuation_confidence import (
    apply_valuation_integrity_headwind,
    compute_valuation_confidence,
    write_valuation_confidence_for_run,
)
from app.valuation.valuation_integrity import write_valuation_integrity_for_run
from app.valuation.fundamental_regression_analytics import (
    write_fundamental_regression_analytics_for_run,
)
from app.valuation.value_type import compute_value_type, write_value_type_for_run
from app.valuation.cyclical_normalization import (
    annual_revenue_series,
    compute_cyclical_normalization,
    write_cyclical_normalization_for_run,
)
from app.valuation.impairment_classification import write_impairment_classification_for_run
from app.valuation.normalization_credibility import (
    write_normalization_credibility_for_run,
)
from app.valuation.capital_allocation_discipline import (
    write_capital_allocation_discipline_for_run,
)
from app.valuation.reinvestment_efficiency import (
    compute_reinvestment_efficiency,
    write_reinvestment_efficiency_for_run,
)
from app.universe.facts_blockers import enrich_facts_blocker_fields, summarize_facts_blockers
from app.universe.ranking import (
    composite_thresholds_from_config,
    compute_composite_score,
    ranking_sort_key,
)
from app.util.json_io import JsonCorruptError, atomic_write_json, read_json_safe

import logging

logger = logging.getLogger(__name__)


UNKNOWN = "UNKNOWN"
PASS = "PASS"
WATCH = "WATCH"
FAIL = "FAIL"
_VALID_SCOUT_STATUSES = {PASS, WATCH, FAIL}
_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")

SCOUT_STATE_RUNNING = "RUNNING"
SCOUT_STATE_DONE = "DONE"
SCOUT_STATE_PARTIAL = "PARTIAL"
SCOUT_STATE_CANCELLED = "CANCELLED"

STOP_NONE = "NONE"
STOP_BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
STOP_MAX_BATCHES_REACHED = "MAX_BATCHES_REACHED"
STOP_CANCELLED = "CANCELLED"
STOP_STALE_SCOUT = "STALE_SCOUT"

HYDRATION_STATUS_IDLE = "IDLE"
HYDRATION_STATUS_RUNNING = "RUNNING"
HYDRATION_STATUS_OK = "OK"
HYDRATION_STATUS_DEGRADED = "DEGRADED"
HYDRATION_STATUS_STALE = "STALE"

PHASE_NOT_STARTED = "NOT_STARTED"
PHASE_BATCH_PREP = "BATCH_PREP"
PHASE_PRICE_COLLECTION = "PRICE_COLLECTION"
PHASE_COMPANYFACTS_ACQUISITION = "COMPANYFACTS_ACQUISITION"
PHASE_FACTS_NORMALIZATION = "FACTS_NORMALIZATION"
PHASE_NET_DEBT_RESOLUTION = "NET_DEBT_RESOLUTION"
PHASE_SCOUT_SCORING = "SCOUT_SCORING"
PHASE_BATCH_FINALIZATION = "BATCH_FINALIZATION"
PHASE_OUTPUT_FINALIZATION = "OUTPUT_FINALIZATION"
PHASE_DONE = "DONE"

REASON_SCOUT_FACTS_TIMEOUT = "SCOUT_FACTS_TIMEOUT"
REASON_SCOUT_NET_DEBT_TIMEOUT = "SCOUT_NET_DEBT_TIMEOUT"
REASON_SCOUT_SCORING_TIMEOUT = "SCOUT_SCORING_TIMEOUT"

BLOCKER_MISSING_PRICE = "MISSING_PRICE"
BLOCKER_MISSING_FACTS = "MISSING_FACTS"
BLOCKER_MISSING_FCF = "MISSING_FCF"
BLOCKER_MISSING_CFO = "MISSING_CFO"
BLOCKER_MISSING_CAPEX = "MISSING_CAPEX"
BLOCKER_MISSING_SHARES = "MISSING_SHARES"
BLOCKER_MISSING_EV = "MISSING_EV"
BLOCKER_NEGATIVE_CFO = "NEGATIVE_CFO"
BLOCKER_NEGATIVE_FCF = "NEGATIVE_FCF"
BLOCKER_LOW_YIELD_OWNER_EARNINGS = "LOW_YIELD_OWNER_EARNINGS"
BLOCKER_LOW_YIELD_FCF = "LOW_YIELD_FCF"
BLOCKER_LOW_YIELD_OWNER_EARNINGS_EV = "LOW_YIELD_OWNER_EARNINGS_EV"
BLOCKER_LOW_YIELD_FCF_EV = "LOW_YIELD_FCF_EV"
BLOCKER_INSUFFICIENT_MOS = "INSUFFICIENT_MOS"
BLOCKER_MISSING_GD_INPUTS = "MISSING_GD_INPUTS"
BLOCKER_INSUFFICIENT_MOS_EPV = "INSUFFICIENT_MOS_EPV"
BLOCKER_INSUFFICIENT_MOS_NETNET = "INSUFFICIENT_MOS_NETNET"
BLOCKER_EXCESS_NET_DEBT = "EXCESS_NET_DEBT"
BLOCKER_EXCESS_DILUTION = "EXCESS_DILUTION"
BLOCKER_OTHER_UNKNOWN = "OTHER_UNKNOWN"

_PRIMARY_BLOCKER_CATEGORY_ORDER = [
    BLOCKER_MISSING_PRICE,
    BLOCKER_MISSING_FACTS,
    BLOCKER_MISSING_FCF,
    BLOCKER_MISSING_CFO,
    BLOCKER_MISSING_CAPEX,
    BLOCKER_MISSING_SHARES,
    BLOCKER_MISSING_EV,
    BLOCKER_MISSING_GD_INPUTS,
    BLOCKER_NEGATIVE_CFO,
    BLOCKER_NEGATIVE_FCF,
    BLOCKER_INSUFFICIENT_MOS_EPV,
    BLOCKER_INSUFFICIENT_MOS_NETNET,
    BLOCKER_LOW_YIELD_OWNER_EARNINGS_EV,
    BLOCKER_LOW_YIELD_FCF_EV,
    BLOCKER_LOW_YIELD_OWNER_EARNINGS,
    BLOCKER_LOW_YIELD_FCF,
    BLOCKER_INSUFFICIENT_MOS,
    BLOCKER_EXCESS_NET_DEBT,
    BLOCKER_EXCESS_DILUTION,
    BLOCKER_OTHER_UNKNOWN,
]

_THRESHOLD_KEYS = [
    "scout_mos_min",
    "scout_valuation_gap_min",
    "scout_fcf_yield_min",
    "scout_net_debt_to_cfo_max",
    "scout_dilution_max",
    "gd_discount_rate",
]
_SCOUT_POLICY_KEYS = [
    *_THRESHOLD_KEYS,
    "scout_require_ev_yield",
    "scout_use_graham_dodd",
]

_SHARES_TAG_PRIORITY: list[tuple[str, str]] = [
    ("dei", "EntityCommonStockSharesOutstanding"),
    ("us-gaap", "CommonStockSharesOutstanding"),
]
_FCF_TAG_PRIORITY: list[tuple[str, str]] = [
    ("us-gaap", "FreeCashFlow"),
]
_CFO_TAG_PRIORITY: list[tuple[str, str]] = [
    ("us-gaap", "NetCashProvidedByUsedInOperatingActivities"),
    ("us-gaap", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"),
]
_CAPEX_TAG_PRIORITY: list[tuple[str, str]] = [
    ("us-gaap", "PaymentsToAcquirePropertyPlantAndEquipment"),
    ("us-gaap", "PaymentsToAcquireProductiveAssets"),
]
_NET_INCOME_TAG_PRIORITY: list[tuple[str, str]] = [
    ("us-gaap", "NetIncomeLoss"),
    ("us-gaap", "ProfitLoss"),
]


class ScoutPhaseTimeoutError(TimeoutError):
    def __init__(self, *, phase: str, ticker: str, timeout_seconds: float):
        self.phase = str(phase)
        self.ticker = str(ticker)
        self.timeout_seconds = float(timeout_seconds)
        super().__init__(f"{self.phase} timed out for {self.ticker} after {self.timeout_seconds:.1f}s")


def _scout_phase_timeout_seconds(*, cfg, max_seconds_remaining: int | None) -> float | None:
    configured = float(getattr(cfg, "scout_phase_timeout_seconds", 8.0))
    if configured <= 0:
        return None
    if max_seconds_remaining is not None:
        configured = min(configured, float(max(1, int(max_seconds_remaining))))
    return max(1.0, float(configured))


def _call_with_timeout(
    *,
    phase: str,
    ticker_label: str,
    timeout_seconds: float | None,
    fn,
    **kwargs,
):
    if timeout_seconds is None or float(timeout_seconds) <= 0:
        return fn(**kwargs)
    if not hasattr(signal, "SIGALRM") or threading.current_thread() is not threading.main_thread():
        return fn(**kwargs)

    previous_handler = signal.getsignal(signal.SIGALRM)

    def _raise_timeout(_signum, _frame) -> None:
        raise ScoutPhaseTimeoutError(phase=phase, ticker=ticker_label, timeout_seconds=float(timeout_seconds))

    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(timeout_seconds))
    try:
        return fn(**kwargs)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def _facts_timeout_row(
    *,
    ticker: str,
    as_of_date: str,
    run_id: str,
    reason_code: str,
    phase: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    ticker_norm = str(ticker).strip().upper()
    detail = f"{phase} exceeded scout timeout after {float(timeout_seconds):.1f}s."
    return {
        "ticker": ticker_norm,
        "requested_as_of": as_of_date,
        "run_id": run_id,
        "cik": None,
        "status": "UNKNOWN",
        "shares_status": "UNKNOWN",
        "shares_reason": reason_code,
        "shares_value": UNKNOWN,
        "shares_asof_used": None,
        "cfo_status": "UNKNOWN",
        "cfo_reason": reason_code,
        "cfo_value": UNKNOWN,
        "cfo_asof_used": None,
        "capex_status": "UNKNOWN",
        "capex_reason": reason_code,
        "capex_value": UNKNOWN,
        "capex_asof_used": None,
        "fcf_status": "UNKNOWN",
        "fcf_reason": reason_code,
        "fcf_value": UNKNOWN,
        "fcf_asof_used": None,
        "source_resolution": "timeout",
        "fetch_reason_code": reason_code,
        "fetch_reason_detail": detail,
        "cache_path": None,
        "source_url": None,
        "http_status": None,
        "network_attempted": True,
        "fetch_attempts": 0,
        "fetch_retries_configured": 0,
        "fetch_backoff_seconds": None,
        "fetch_timeout_seconds": float(timeout_seconds),
        "fetch_last_exception_type": "ScoutPhaseTimeoutError",
        "fetch_last_exception_message": detail,
        "derived_from": [
            f"scout_timeout.phase={phase}",
            f"scout_timeout.reason_code={reason_code}",
        ],
        "generated_at": utc_now_iso(),
    }


def _net_debt_timeout_row(
    *,
    ticker: str,
    as_of_date: str,
    reason_code: str,
    phase: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    return {
        "ticker": str(ticker).strip().upper(),
        "as_of_date": as_of_date,
        "status": "UNKNOWN",
        "reason_code": reason_code,
        "total_debt": {"value": UNKNOWN, "tag": None, "date": None, "derived_from": []},
        "cash_equivalents": {"value": UNKNOWN, "tag": None, "date": None, "derived_from": []},
        "net_debt_proxy": UNKNOWN,
        "derived_from": [
            f"scout_timeout.phase={phase}",
            f"scout_timeout.reason_code={reason_code}",
            f"scout_timeout.seconds={float(timeout_seconds):.1f}",
        ],
    }


def _safe_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float))


def _to_num(value: Any) -> float | str:
    return float(value) if _is_num(value) else UNKNOWN


def _sorted_unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        token = str(item).strip()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return sorted(out)


def normalize_scout_thresholds(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = get_config()
    thresholds: dict[str, Any] = {
        "scout_mos_min": float(getattr(cfg, "scout_mos_min", 0.30)),
        "scout_valuation_gap_min": float(getattr(cfg, "scout_valuation_gap_min", 0.10)),
        "scout_fcf_yield_min": float(getattr(cfg, "scout_fcf_yield_min", 0.03)),
        "scout_net_debt_to_cfo_max": float(getattr(cfg, "scout_net_debt_to_cfo_max", 2.5)),
        "scout_dilution_max": float(getattr(cfg, "scout_dilution_max", 0.06)),
        "scout_require_ev_yield": bool(getattr(cfg, "scout_require_ev_yield", False)),
        "gd_discount_rate": float(getattr(cfg, "graham_discount_rate_default", 0.10)),
        "scout_use_graham_dodd": bool(getattr(cfg, "scout_use_graham_dodd", True)),
    }
    for key, value in (overrides or {}).items():
        if key in {"scout_require_ev_yield", "scout_use_graham_dodd"}:
            if value is None:
                continue
            thresholds[key] = bool(value)
            continue
        if key not in thresholds:
            continue
        if not _is_num(value):
            continue
        thresholds[key] = float(value)
    thresholds["scout_mos_min"] = max(-5.0, min(5.0, float(thresholds["scout_mos_min"])))
    thresholds["scout_valuation_gap_min"] = max(-5.0, min(5.0, float(thresholds["scout_valuation_gap_min"])))
    thresholds["scout_fcf_yield_min"] = max(-1.0, min(1.0, float(thresholds["scout_fcf_yield_min"])))
    thresholds["scout_net_debt_to_cfo_max"] = max(0.0, min(100.0, float(thresholds["scout_net_debt_to_cfo_max"])))
    thresholds["scout_dilution_max"] = max(-1.0, min(1.0, float(thresholds["scout_dilution_max"])))
    thresholds["gd_discount_rate"] = max(0.0001, min(1.0, float(thresholds["gd_discount_rate"])))
    thresholds["scout_require_ev_yield"] = bool(thresholds.get("scout_require_ev_yield", False))
    thresholds["scout_use_graham_dodd"] = bool(thresholds.get("scout_use_graham_dodd", True))
    return thresholds


def _parse_yyyy_mm_dd(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(str(value), "%Y-%m-%d")
    except Exception:
        return None


def _ticker_token(value: Any) -> str:
    token = str(value or "").strip().upper()
    if not token:
        return ""
    return token if _TICKER_RE.match(token) else ""


def _canonicalize_tickers(raw: list[str]) -> list[str]:
    deduped = {_ticker_token(ticker) for ticker in raw}
    return sorted([ticker for ticker in deduped if ticker])


def _dedupe_tickers_keep_order(raw: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in raw:
        token = _ticker_token(item)
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _tickers_from_csv(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "ticker" not in (reader.fieldnames or []):
            raise ValueError(f"Universe CSV missing required 'ticker' column: {path}")
        return _canonicalize_tickers([str(row.get("ticker") or "") for row in reader if isinstance(row, dict)])


def _tickers_from_sector_run(run_id: str) -> list[str]:
    cfg = get_config()
    run_dir = cfg.sectors_dir / run_id
    payload = _safe_json(run_dir / "sector_peers.json")
    selected = [str(t) for t in (payload.get("selected_tickers") or [])]
    if selected:
        return _canonicalize_tickers(selected)
    scoreboard = _safe_json(run_dir / "peer_scoreboard.json")
    rows = [row for row in (scoreboard.get("rows") or []) if isinstance(row, dict)]
    from_rows = [str(row.get("ticker") or "") for row in rows]
    return _canonicalize_tickers(from_rows)


def _resolve_universe_tickers(
    *,
    tickers: list[str] | None = None,
    sector_run_id: str | None = None,
    universe_csv: Path | None = None,
) -> tuple[list[str], dict[str, Any]]:
    all_tokens: list[str] = []
    sources: list[dict[str, Any]] = []
    if tickers:
        all_tokens.extend([str(t) for t in tickers])
        sources.append({"source_type": "tickers", "count": len(_canonicalize_tickers([str(t) for t in tickers]))})
    if sector_run_id:
        sector_tickers = _tickers_from_sector_run(str(sector_run_id))
        all_tokens.extend(sector_tickers)
        sources.append(
            {
                "source_type": "sector_run",
                "source_run_id": str(sector_run_id),
                "count": len(sector_tickers),
            }
        )
    if universe_csv is not None:
        csv_tickers = _tickers_from_csv(universe_csv)
        all_tokens.extend(csv_tickers)
        sources.append(
            {
                "source_type": "universe_csv",
                "source_path": str(universe_csv),
                "count": len(csv_tickers),
            }
        )
    normalized = [_ticker_token(token) for token in all_tokens]
    valid_tokens = [token for token in normalized if token]
    resolved = sorted(set(valid_tokens))
    invalid_count = len([token for token in normalized if not token])
    duplicate_count = max(0, len(valid_tokens) - len(resolved))
    meta = {
        "sources": sources,
        "count_requested": len(all_tokens),
        "count_valid": len(valid_tokens),
        "count_invalid": int(invalid_count),
        "count_duplicates_removed": int(duplicate_count),
        "count_used": len(resolved),
    }
    return resolved, meta


def _load_companyfacts_payload_from_facts_row(facts_row: dict[str, Any]) -> dict[str, Any]:
    cache_path = str(facts_row.get("cache_path") or "").strip()
    if not cache_path:
        return {}
    payload = _safe_json(Path(cache_path))
    if isinstance(payload.get("companyfacts"), dict):
        return payload.get("companyfacts")  # type: ignore[return-value]
    if isinstance(payload.get("facts"), dict):
        return payload  # type: ignore[return-value]
    return {}


def _facts_node(companyfacts: dict[str, Any], taxonomy: str, tag: str) -> dict[str, Any]:
    facts = companyfacts.get("facts") if isinstance(companyfacts.get("facts"), dict) else {}
    tax = facts.get(taxonomy) if isinstance(facts, dict) else {}
    if not isinstance(tax, dict):
        return {}
    tag_node = tax.get(tag)
    return tag_node if isinstance(tag_node, dict) else {}


def _fact_rows(
    *,
    companyfacts: dict[str, Any],
    taxonomy: str,
    tag: str,
    as_of_date: str,
    expected_unit_exact: tuple[str, ...] = ("usd",),
) -> list[dict[str, Any]]:
    tag_node = _facts_node(companyfacts, taxonomy, tag)
    units = tag_node.get("units") if isinstance(tag_node.get("units"), dict) else {}
    if not isinstance(units, dict):
        return []
    asof_dt = _parse_yyyy_mm_dd(as_of_date)
    if asof_dt is None:
        return []
    rows_out: list[dict[str, Any]] = []
    expected = {token.lower() for token in expected_unit_exact}
    for unit in sorted(units.keys()):
        unit_norm = str(unit).strip().lower()
        if expected and unit_norm not in expected:
            continue
        rows = units.get(unit)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            value = row.get("val")
            if not _is_num(value):
                continue
            end_date = str(row.get("end") or "")
            end_dt = _parse_yyyy_mm_dd(end_date)
            if end_dt is None or end_dt > asof_dt:
                continue
            filed = str(row.get("filed") or "")
            rows_out.append(
                {
                    "value": float(value),
                    "end_date": end_date,
                    "filed": filed,
                    "unit": str(unit),
                    "ref": f"companyfacts.{taxonomy}.{tag}[end_date={end_date},unit={unit}]",
                }
            )
    rows_out.sort(key=lambda item: (str(item.get("end_date") or ""), str(item.get("filed") or "")))
    return rows_out


def _latest_fact(
    *,
    companyfacts: dict[str, Any],
    as_of_date: str,
    priority: list[tuple[str, str]],
    expected_unit_exact: tuple[str, ...] = ("usd",),
) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for taxonomy, tag in priority:
        rows = _fact_rows(
            companyfacts=companyfacts,
            taxonomy=taxonomy,
            tag=tag,
            as_of_date=as_of_date,
            expected_unit_exact=expected_unit_exact,
        )
        if not rows:
            continue
        chosen = rows[-1]
        chosen = {**chosen, "taxonomy": taxonomy, "tag": tag}
        candidates.append(chosen)
    if not candidates:
        return None
    candidates.sort(
        key=lambda item: (
            str(item.get("end_date") or ""),
            str(item.get("filed") or ""),
            str(item.get("taxonomy") or ""),
            str(item.get("tag") or ""),
        ),
        reverse=True,
    )
    return candidates[0]


def _annual_series_for_priority(
    *,
    companyfacts: dict[str, Any],
    as_of_date: str,
    priority: list[tuple[str, str]],
    expected_unit_exact: tuple[str, ...] = ("usd",),
) -> list[dict[str, Any]]:
    by_year: dict[int, dict[str, Any]] = {}
    for taxonomy, tag in priority:
        rows = _fact_rows(
            companyfacts=companyfacts,
            taxonomy=taxonomy,
            tag=tag,
            as_of_date=as_of_date,
            expected_unit_exact=expected_unit_exact,
        )
        for row in rows:
            end_date = str(row.get("end_date") or "")
            year = _parse_yyyy_mm_dd(end_date).year if _parse_yyyy_mm_dd(end_date) else 0
            if year <= 0:
                continue
            candidate = {**row, "year": year, "taxonomy": taxonomy, "tag": tag}
            existing = by_year.get(year)
            if existing is None:
                by_year[year] = candidate
                continue
            key_candidate = (
                str(candidate.get("end_date") or ""),
                str(candidate.get("filed") or ""),
                str(candidate.get("taxonomy") or ""),
                str(candidate.get("tag") or ""),
            )
            key_existing = (
                str(existing.get("end_date") or ""),
                str(existing.get("filed") or ""),
                str(existing.get("taxonomy") or ""),
                str(existing.get("tag") or ""),
            )
            if key_candidate > key_existing:
                by_year[year] = candidate
    return [by_year[year] for year in sorted(by_year.keys())]


def _derive_dilution_rate(*, companyfacts: dict[str, Any], as_of_date: str) -> tuple[float | str, str, list[str]]:
    if not companyfacts:
        return UNKNOWN, "FACTS_PAYLOAD_MISSING", []
    shares_series = _annual_series_for_priority(
        companyfacts=companyfacts,
        as_of_date=as_of_date,
        priority=_SHARES_TAG_PRIORITY,
        expected_unit_exact=("shares",),
    )
    if len(shares_series) < 2:
        return UNKNOWN, "INSUFFICIENT_HISTORY", [str(row.get("ref") or "") for row in shares_series]
    first = shares_series[0]
    last = shares_series[-1]
    first_value = first.get("value")
    last_value = last.get("value")
    if not (_is_num(first_value) and _is_num(last_value)):
        return UNKNOWN, "INSUFFICIENT_HISTORY", [str(row.get("ref") or "") for row in shares_series]
    if float(first_value) <= 0:
        return UNKNOWN, "NONPOSITIVE_BASE", [str(row.get("ref") or "") for row in shares_series]
    span_years = max(1, int(last.get("year", 0)) - int(first.get("year", 0)))
    cagr = (float(last_value) / float(first_value)) ** (1.0 / float(span_years)) - 1.0
    refs = [str(first.get("ref") or ""), str(last.get("ref") or "")]
    return float(cagr), "OK", [ref for ref in refs if ref]


def _normalize_recent_numeric(values: list[float]) -> tuple[float | str, str, int]:
    if not values:
        return UNKNOWN, REASON_INSUFFICIENT_HISTORY, 0
    recent = [float(v) for v in values[-3:]]
    if len(recent) >= 3:
        ordered = sorted(recent)
        return float(ordered[len(ordered) // 2]), "MEDIAN_3Y", len(recent)
    if len(recent) >= 2:
        return float(sum(recent) / float(len(recent))), f"AVG_{len(recent)}Y", len(recent)
    return float(recent[-1]), "LATEST", 1


def _derive_fcf_series(*, companyfacts: dict[str, Any], as_of_date: str) -> tuple[list[dict[str, Any]], str, list[str]]:
    if not companyfacts:
        return [], "FACTS_PAYLOAD_MISSING", []
    direct = _annual_series_for_priority(
        companyfacts=companyfacts,
        as_of_date=as_of_date,
        priority=_FCF_TAG_PRIORITY,
        expected_unit_exact=("usd",),
    )
    cfo = _annual_series_for_priority(
        companyfacts=companyfacts,
        as_of_date=as_of_date,
        priority=_CFO_TAG_PRIORITY,
        expected_unit_exact=("usd",),
    )
    capex = _annual_series_for_priority(
        companyfacts=companyfacts,
        as_of_date=as_of_date,
        priority=_CAPEX_TAG_PRIORITY,
        expected_unit_exact=("usd",),
    )
    cfo_by_year = {int(row.get("year", 0)): row for row in cfo}
    capex_by_year = {int(row.get("year", 0)): row for row in capex}
    direct_by_year = {int(row.get("year", 0)): row for row in direct}
    years = sorted(set(direct_by_year.keys()).union(set(cfo_by_year.keys()).intersection(set(capex_by_year.keys()))))
    if not years:
        reason = "INSUFFICIENT_HISTORY"
        if not cfo_by_year:
            reason = REASON_MISSING_CFO
        elif not capex_by_year:
            reason = REASON_MISSING_CAPEX
        return [], reason, []
    series: list[dict[str, Any]] = []
    refs: list[str] = []
    for year in years:
        row_direct = direct_by_year.get(year)
        if row_direct is not None and _is_num(row_direct.get("value")):
            value = float(row_direct["value"])
            ref = str(row_direct.get("ref") or "")
            series.append(
                {
                    "year": int(year),
                    "value": value,
                    "source": "DIRECT_FCF",
                    "derived_from": [ref] if ref else [],
                }
            )
            if ref:
                refs.append(ref)
            continue
        row_cfo = cfo_by_year.get(year)
        row_capex = capex_by_year.get(year)
        if row_cfo is None or row_capex is None:
            continue
        if not (_is_num(row_cfo.get("value")) and _is_num(row_capex.get("value"))):
            continue
        cfo_ref = str(row_cfo.get("ref") or "")
        capex_ref = str(row_capex.get("ref") or "")
        refs.extend([cfo_ref, capex_ref])
        series.append(
            {
                "year": int(year),
                "value": float(row_cfo["value"]) - float(row_capex["value"]),
                "source": "DERIVED_CFO_MINUS_CAPEX",
                "derived_from": [
                    token
                    for token in [cfo_ref, capex_ref, "derived:fcf=cfo-capex"]
                    if token
                ],
            }
        )
    series = sorted(series, key=lambda row: int(row.get("year", 0)))
    refs = [str(ref) for ref in refs if str(ref).strip()]
    if not series:
        return [], "INSUFFICIENT_HISTORY", sorted(set(refs))
    return series, "OK", sorted(set(refs))


def _derive_fcf_stability(*, companyfacts: dict[str, Any], as_of_date: str) -> tuple[float | str, str, list[str], list[float]]:
    fcf_series, fcf_series_reason, fcf_series_refs = _derive_fcf_series(
        companyfacts=companyfacts,
        as_of_date=as_of_date,
    )
    values = [float(row.get("value")) for row in fcf_series if _is_num(row.get("value"))][-5:]
    refs = [str(ref) for ref in fcf_series_refs if str(ref).strip()]
    values = values[-5:]
    if len(values) < 3:
        return UNKNOWN, fcf_series_reason if fcf_series_reason != "OK" else "INSUFFICIENT_HISTORY", refs, values
    positive = len([value for value in values if value > 0])
    ratio = float(positive) / float(len(values))
    if ratio >= 1.0:
        score = 1.0
    elif ratio >= 0.8:
        score = 0.7
    elif ratio >= 0.5:
        score = 0.4
    else:
        score = 0.1
    return float(score), "OK", refs, values


def _score_components(
    *,
    valuation_gap: float | str,
    fcf_yield: float | str,
    net_debt_to_cfo: float | str,
    net_debt_to_fcf: float | str,
    dilution_rate: float | str,
    fcf_stability_score: float | str,
    price_known: bool,
    shares_known: bool,
    fcf_known: bool,
) -> tuple[dict[str, float], float, list[str]]:
    components = {
        "valuation_mos": 0.0,
        "fcf_yield": 0.0,
        "balance_sheet": 0.0,
        "dilution_discipline": 0.0,
        "quality_stability": 0.0,
        "unknown_penalty": 0.0,
    }
    reasons: list[str] = []
    if _is_num(valuation_gap):
        value = float(valuation_gap)
        if value >= 0.50:
            components["valuation_mos"] = 30.0
        elif value >= 0.30:
            components["valuation_mos"] = 24.0
        elif value >= 0.15:
            components["valuation_mos"] = 16.0
        elif value >= 0.00:
            components["valuation_mos"] = 8.0
        reasons.append("MOS_PROXY_AVAILABLE")
    else:
        components["unknown_penalty"] -= 8.0
        reasons.append("MOS_PROXY_UNKNOWN")

    if _is_num(fcf_yield):
        value = float(fcf_yield)
        if value >= 0.08:
            components["fcf_yield"] = 25.0
        elif value >= 0.05:
            components["fcf_yield"] = 20.0
        elif value >= 0.03:
            components["fcf_yield"] = 14.0
        elif value >= 0.01:
            components["fcf_yield"] = 8.0
        elif value > 0:
            components["fcf_yield"] = 4.0
        reasons.append("FCF_YIELD_AVAILABLE")
    else:
        components["unknown_penalty"] -= 6.0
        reasons.append("FCF_YIELD_UNKNOWN")

    if _is_num(net_debt_to_cfo):
        value = float(net_debt_to_cfo)
        if value <= 1.5:
            components["balance_sheet"] = 20.0
        elif value <= 2.5:
            components["balance_sheet"] = 15.0
        elif value <= 4.0:
            components["balance_sheet"] = 8.0
        reasons.append("NET_DEBT_TO_CFO_AVAILABLE")
    elif _is_num(net_debt_to_fcf):
        value = float(net_debt_to_fcf)
        if value <= 3.0:
            components["balance_sheet"] = 16.0
        elif value <= 6.0:
            components["balance_sheet"] = 10.0
        elif value <= 10.0:
            components["balance_sheet"] = 5.0
        reasons.append("NET_DEBT_TO_FCF_AVAILABLE")
    else:
        components["unknown_penalty"] -= 4.0
        reasons.append("BALANCE_PROXY_UNKNOWN")

    if _is_num(dilution_rate):
        value = float(dilution_rate)
        if value <= 0.00:
            components["dilution_discipline"] = 15.0
        elif value <= 0.02:
            components["dilution_discipline"] = 12.0
        elif value <= 0.04:
            components["dilution_discipline"] = 8.0
        elif value <= 0.06:
            components["dilution_discipline"] = 4.0
        reasons.append("DILUTION_AVAILABLE")
    else:
        components["unknown_penalty"] -= 3.0
        reasons.append("DILUTION_UNKNOWN")

    if _is_num(fcf_stability_score):
        components["quality_stability"] = max(0.0, min(10.0, float(fcf_stability_score) * 10.0))
        reasons.append("FCF_STABILITY_AVAILABLE")
    else:
        components["quality_stability"] = 3.0 if fcf_known else 0.0
        reasons.append("FCF_STABILITY_UNKNOWN")

    if not price_known:
        components["unknown_penalty"] -= 4.0
    if not shares_known:
        components["unknown_penalty"] -= 4.0
    if not fcf_known:
        components["unknown_penalty"] -= 6.0
    total = round(sum(components.values()), 6)
    total = max(0.0, min(100.0, total))
    return components, total, reasons


def _primary_blocker_category(*, status: str, blocker_categories: list[str]) -> str:
    if str(status).upper() == PASS:
        return "NONE"
    categories = {str(code).strip().upper() for code in blocker_categories if str(code).strip()}
    if not categories:
        return BLOCKER_OTHER_UNKNOWN
    for code in _PRIMARY_BLOCKER_CATEGORY_ORDER:
        if code in categories:
            return code
    return BLOCKER_OTHER_UNKNOWN


def _near_miss_fields(
    *,
    valuation_gap: float | str,
    mos_epv: float | str,
    mos_netnet: float | str,
    fcf_yield: float | str,
    net_debt_to_cfo: float | str,
    dilution_rate: float | str,
    thresholds: dict[str, float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def _append(field: str, value: Any, threshold: Any, delta: Any) -> None:
        if not (_is_num(value) and _is_num(threshold) and _is_num(delta)):
            return
        if float(delta) <= 0:
            return
        rows.append(
            {
                "field": field,
                "value": round(float(value), 6),
                "threshold": round(float(threshold), 6),
                "delta": round(float(delta), 6),
            }
        )

    _append(
        "scout_valuation_gap_min",
        valuation_gap,
        thresholds["scout_valuation_gap_min"],
        float(thresholds["scout_valuation_gap_min"]) - float(valuation_gap) if _is_num(valuation_gap) else UNKNOWN,
    )
    _append(
        "scout_mos_min",
        valuation_gap,
        thresholds["scout_mos_min"],
        float(thresholds["scout_mos_min"]) - float(valuation_gap) if _is_num(valuation_gap) else UNKNOWN,
    )
    _append(
        "mos_epv",
        mos_epv,
        thresholds["scout_mos_min"],
        float(thresholds["scout_mos_min"]) - float(mos_epv) if _is_num(mos_epv) else UNKNOWN,
    )
    _append(
        "mos_netnet",
        mos_netnet,
        thresholds["scout_mos_min"],
        float(thresholds["scout_mos_min"]) - float(mos_netnet) if _is_num(mos_netnet) else UNKNOWN,
    )
    _append(
        "scout_fcf_yield_min",
        fcf_yield,
        thresholds["scout_fcf_yield_min"],
        float(thresholds["scout_fcf_yield_min"]) - float(fcf_yield) if _is_num(fcf_yield) else UNKNOWN,
    )
    _append(
        "scout_net_debt_to_cfo_max",
        net_debt_to_cfo,
        thresholds["scout_net_debt_to_cfo_max"],
        float(net_debt_to_cfo) - float(thresholds["scout_net_debt_to_cfo_max"]) if _is_num(net_debt_to_cfo) else UNKNOWN,
    )
    _append(
        "scout_dilution_max",
        dilution_rate,
        thresholds["scout_dilution_max"],
        float(dilution_rate) - float(thresholds["scout_dilution_max"]) if _is_num(dilution_rate) else UNKNOWN,
    )
    rows.sort(key=lambda row: (float(row.get("delta") or 0.0), str(row.get("field") or "")))
    return rows


def _recommendation_for_blocker(category: str) -> str:
    token = str(category or "").upper()
    if token == BLOCKER_MISSING_PRICE:
        return "hydrate price"
    if token in {
        BLOCKER_MISSING_FACTS,
        BLOCKER_MISSING_FCF,
        BLOCKER_MISSING_CFO,
        BLOCKER_MISSING_CAPEX,
        BLOCKER_MISSING_SHARES,
    }:
        return "hydrate facts"
    if token == BLOCKER_MISSING_GD_INPUTS:
        return "hydrate facts for EPV/Net-Net inputs (shares, current assets, liabilities)"
    if token == BLOCKER_MISSING_EV:
        return "hydrate facts (net debt proxy) to enable EV-based yield"
    if token == BLOCKER_NEGATIVE_CFO:
        return "negative CFO confirmed; keep FAIL unless operating cash flow turns positive"
    if token == BLOCKER_NEGATIVE_FCF:
        return "negative FCF confirmed; keep FAIL unless free cash flow turns positive"
    if token in {
        BLOCKER_LOW_YIELD_OWNER_EARNINGS,
        BLOCKER_LOW_YIELD_FCF,
        BLOCKER_LOW_YIELD_OWNER_EARNINGS_EV,
        BLOCKER_LOW_YIELD_FCF_EV,
    }:
        return "consider lowering scout_fcf_yield_min only after validating FCF quality"
    if token == BLOCKER_INSUFFICIENT_MOS:
        return "consider lowering scout_mos_min or scout_valuation_gap_min after reviewing valuation assumptions"
    if token in {BLOCKER_INSUFFICIENT_MOS_EPV, BLOCKER_INSUFFICIENT_MOS_NETNET}:
        return "review Graham/Dodd MOS assumptions (discount rate and asset/liability inputs)"
    if token == BLOCKER_EXCESS_NET_DEBT:
        return "consider raising scout_net_debt_to_cfo_max only with debt/CFO evidence"
    if token == BLOCKER_EXCESS_DILUTION:
        return "consider raising scout_dilution_max only with durable share-policy evidence"
    return "inspect unknown coverage reasons and hydrate missing inputs"


def _whole_to_millions(value: Any) -> float | str:
    return float(value) / 1_000_000.0 if _is_num(value) else value


def _yield_value(value: Any, denominator: Any) -> float | str:
    if not (_is_num(value) and _is_num(denominator)):
        return UNKNOWN
    if float(denominator) <= 0:
        return UNKNOWN
    return float(value) / float(denominator)


def _select_yield_metric(
    *,
    price_status: str,
    shares_status: str,
    market_cap: float | str,
    ev: float | str,
    ev_status: str,
    ev_reason_code: str,
    require_ev_yield: bool,
    owner_payload: dict[str, Any],
    fcf_series: list[dict[str, Any]],
    fcf_latest: float | str,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    owner_summary = owner_payload.get("summary") if isinstance(owner_payload.get("summary"), dict) else {}
    owner_reasons = [str(code) for code in (owner_payload.get("reason_codes") or []) if str(code).strip()]
    # The owner-earnings series and the companyfacts FCF series are whole dollars;
    # market cap and EV here are $millions (facts-row shares in millions, net debt
    # in $millions). Put the numerators in $millions before dividing, or every
    # yield reads a million times too high. ``fcf_latest`` is the facts row's
    # $millions figure already.
    owner_norm = _whole_to_millions(owner_summary.get("owner_earnings_normalized_3y", UNKNOWN))
    owner_latest = _whole_to_millions(owner_summary.get("owner_earnings_latest", UNKNOWN))
    owner_method = str(owner_summary.get("owner_earnings_normalized_method") or REASON_INSUFFICIENT_HISTORY)

    fcf_values = [
        float(row.get("value")) / 1_000_000.0 for row in fcf_series if _is_num(row.get("value"))
    ]
    fcf_norm, fcf_norm_method, fcf_norm_points = _normalize_recent_numeric(fcf_values)
    fcf_latest_from_series = fcf_values[-1] if fcf_values else UNKNOWN
    fcf_latest_value = fcf_latest_from_series if _is_num(fcf_latest_from_series) else fcf_latest

    owner_yield_ev_3y = _yield_value(owner_norm, ev)
    owner_yield_ev_latest = _yield_value(owner_latest, ev)
    fcf_yield_ev_3y = _yield_value(fcf_norm, ev)
    fcf_yield_ev_latest = _yield_value(fcf_latest_value, ev)

    owner_yield_mc_3y = _yield_value(owner_norm, market_cap)
    owner_yield_mc_latest = _yield_value(owner_latest, market_cap)
    fcf_yield_mc_3y = _yield_value(fcf_norm, market_cap)
    fcf_yield_mc_latest = _yield_value(fcf_latest_value, market_cap)

    metric_name = UNKNOWN
    metric_type = UNKNOWN
    yield_value_used: float | str = UNKNOWN
    yield_reason_code = "UNKNOWN"
    yield_status = "UNKNOWN"
    yield_subreason = "UNKNOWN"
    denominator_used = UNKNOWN
    calibration_note = ""

    if bool(require_ev_yield) and str(ev_status).upper() != "OK":
        yield_reason_code = BLOCKER_MISSING_EV
        yield_subreason = str(ev_reason_code or "MISSING_EV").upper()
        return {
            "owner_earnings_yield_3y": _to_num(owner_yield_mc_3y),
            "owner_earnings_yield_latest": _to_num(owner_yield_mc_latest),
            "fcf_yield_3y": _to_num(fcf_yield_mc_3y),
            "fcf_yield_latest": _to_num(fcf_yield_mc_latest),
            "owner_earnings_yield_ev_3y": _to_num(owner_yield_ev_3y),
            "owner_earnings_yield_ev_latest": _to_num(owner_yield_ev_latest),
            "fcf_yield_ev_3y": _to_num(fcf_yield_ev_3y),
            "fcf_yield_ev_latest": _to_num(fcf_yield_ev_latest),
            "yield_gate_value_used": UNKNOWN,
            "yield_metric_used": UNKNOWN,
            "yield_metric_type": UNKNOWN,
            "yield_denominator_used": UNKNOWN,
            "yield_status": "UNKNOWN",
            "yield_reason_code": yield_reason_code,
            "yield_blocker_subreason": yield_subreason,
            "yield_delta_to_pass": UNKNOWN,
            "yield_calibration_note": "REQUIRE_EV_YIELD_MISSING_EV",
        }

    candidates: list[tuple[str, str, str, float | str, str]] = []
    if str(ev_status).upper() == "OK":
        candidates.extend(
            [
                (f"OWNER_EARNINGS_EV_{owner_method}", "OWNER_EARNINGS", "EV", owner_yield_ev_3y, "OWNER_EARNINGS_YIELD_EV_3Y"),
                (f"FCF_EV_{fcf_norm_method}", "FCF", "EV", fcf_yield_ev_3y, "FCF_YIELD_EV_3Y"),
            ]
        )
        if not bool(require_ev_yield):
            candidates.extend(
                [
                    (f"OWNER_EARNINGS_{owner_method}", "OWNER_EARNINGS", "MARKET_CAP", owner_yield_mc_3y, "OWNER_EARNINGS_YIELD_3Y"),
                    (f"FCF_{fcf_norm_method}", "FCF", "MARKET_CAP", fcf_yield_mc_3y, "FCF_YIELD_3Y"),
                ]
            )
        candidates.extend(
            [
                ("OWNER_EARNINGS_EV_LATEST", "OWNER_EARNINGS", "EV", owner_yield_ev_latest, "OWNER_EARNINGS_YIELD_EV_LATEST"),
                ("FCF_EV_LATEST", "FCF", "EV", fcf_yield_ev_latest, "FCF_YIELD_EV_LATEST"),
            ]
        )
        if not bool(require_ev_yield):
            candidates.extend(
                [
                    ("OWNER_EARNINGS_LATEST", "OWNER_EARNINGS", "MARKET_CAP", owner_yield_mc_latest, "OWNER_EARNINGS_YIELD_LATEST"),
                    ("FCF_LATEST", "FCF", "MARKET_CAP", fcf_yield_mc_latest, "FCF_YIELD_LATEST"),
                ]
            )
    else:
        candidates.extend(
            [
                (f"OWNER_EARNINGS_{owner_method}", "OWNER_EARNINGS", "MARKET_CAP", owner_yield_mc_3y, "OWNER_EARNINGS_YIELD_3Y"),
                (f"FCF_{fcf_norm_method}", "FCF", "MARKET_CAP", fcf_yield_mc_3y, "FCF_YIELD_3Y"),
                ("OWNER_EARNINGS_LATEST", "OWNER_EARNINGS", "MARKET_CAP", owner_yield_mc_latest, "OWNER_EARNINGS_YIELD_LATEST"),
                ("FCF_LATEST", "FCF", "MARKET_CAP", fcf_yield_mc_latest, "FCF_YIELD_LATEST"),
            ]
        )

    selected: tuple[str, str, str, float, str] | None = None
    for candidate in candidates:
        if _is_num(candidate[3]):
            selected = (candidate[0], candidate[1], candidate[2], float(candidate[3]), candidate[4])
            break

    if selected is not None:
        metric_name, metric_type, denominator_used, yield_value_used, yield_reason_code = selected
        if denominator_used == "MARKET_CAP" and str(ev_status).upper() != "OK":
            calibration_note = "EV_UNKNOWN_MARKET_CAP_FALLBACK"
    else:
        if price_status != "OK":
            yield_reason_code = BLOCKER_MISSING_PRICE
            yield_subreason = BLOCKER_MISSING_PRICE
        elif shares_status != "OK":
            yield_reason_code = BLOCKER_MISSING_SHARES
            yield_subreason = BLOCKER_MISSING_SHARES
        elif str(ev_status).upper() != "OK" and bool(require_ev_yield):
            yield_reason_code = BLOCKER_MISSING_EV
            yield_subreason = str(ev_reason_code or "MISSING_EV").upper()
        elif REASON_MISSING_CFO in owner_reasons:
            yield_reason_code = REASON_MISSING_CFO
            yield_subreason = REASON_MISSING_CFO
        elif REASON_MISSING_CAPEX in owner_reasons:
            yield_reason_code = REASON_MISSING_CAPEX
            yield_subreason = REASON_MISSING_CAPEX
        elif REASON_NEGATIVE_CFO in owner_reasons:
            yield_reason_code = REASON_NEGATIVE_CFO
            yield_subreason = REASON_NEGATIVE_CFO
        elif REASON_NEGATIVE_OWNER_EARNINGS in owner_reasons:
            yield_reason_code = REASON_NEGATIVE_OWNER_EARNINGS
            yield_subreason = "HIGH_CAPEX_PROXY"
        elif fcf_norm_points == 0 and not _is_num(fcf_latest):
            yield_reason_code = BLOCKER_MISSING_FCF
            yield_subreason = BLOCKER_MISSING_FCF
        elif fcf_norm_points == 0:
            yield_reason_code = REASON_INSUFFICIENT_HISTORY
            yield_subreason = REASON_INSUFFICIENT_HISTORY
        else:
            yield_reason_code = BLOCKER_OTHER_UNKNOWN
            yield_subreason = BLOCKER_OTHER_UNKNOWN
        return {
            "owner_earnings_yield_3y": _to_num(owner_yield_mc_3y),
            "owner_earnings_yield_latest": _to_num(owner_yield_mc_latest),
            "fcf_yield_3y": _to_num(fcf_yield_mc_3y),
            "fcf_yield_latest": _to_num(fcf_yield_mc_latest),
            "owner_earnings_yield_ev_3y": _to_num(owner_yield_ev_3y),
            "owner_earnings_yield_ev_latest": _to_num(owner_yield_ev_latest),
            "fcf_yield_ev_3y": _to_num(fcf_yield_ev_3y),
            "fcf_yield_ev_latest": _to_num(fcf_yield_ev_latest),
            "yield_gate_value_used": UNKNOWN,
            "yield_metric_used": UNKNOWN,
            "yield_metric_type": UNKNOWN,
            "yield_denominator_used": UNKNOWN,
            "yield_status": "UNKNOWN",
            "yield_reason_code": yield_reason_code,
            "yield_blocker_subreason": yield_subreason,
            "yield_delta_to_pass": UNKNOWN,
            "yield_calibration_note": calibration_note,
        }

    if float(yield_value_used) < float(thresholds["scout_fcf_yield_min"]):
        yield_status = "LOW"
        if metric_type == "OWNER_EARNINGS":
            yield_reason_code = (
                BLOCKER_LOW_YIELD_OWNER_EARNINGS_EV if denominator_used == "EV" else BLOCKER_LOW_YIELD_OWNER_EARNINGS
            )
            if REASON_NEGATIVE_CFO in owner_reasons:
                yield_subreason = REASON_NEGATIVE_CFO
            elif REASON_NEGATIVE_OWNER_EARNINGS in owner_reasons:
                yield_subreason = "HIGH_CAPEX_PROXY"
            else:
                yield_subreason = "LOW_YIELD"
        else:
            yield_reason_code = BLOCKER_LOW_YIELD_FCF_EV if denominator_used == "EV" else BLOCKER_LOW_YIELD_FCF
            yield_subreason = "NEGATIVE_FCF" if float(yield_value_used) < 0 else "LOW_YIELD"
    else:
        yield_status = "OK"
        yield_subreason = "YIELD_ABOVE_THRESHOLD"

    delta = round(float(thresholds["scout_fcf_yield_min"]) - float(yield_value_used), 6) if _is_num(yield_value_used) else UNKNOWN
    return {
        "owner_earnings_yield_3y": _to_num(owner_yield_mc_3y),
        "owner_earnings_yield_latest": _to_num(owner_yield_mc_latest),
        "fcf_yield_3y": _to_num(fcf_yield_mc_3y),
        "fcf_yield_latest": _to_num(fcf_yield_mc_latest),
        "owner_earnings_yield_ev_3y": _to_num(owner_yield_ev_3y),
        "owner_earnings_yield_ev_latest": _to_num(owner_yield_ev_latest),
        "fcf_yield_ev_3y": _to_num(fcf_yield_ev_3y),
        "fcf_yield_ev_latest": _to_num(fcf_yield_ev_latest),
        "yield_gate_value_used": _to_num(yield_value_used),
        "yield_metric_used": metric_name,
        "yield_metric_type": metric_type,
        "yield_denominator_used": denominator_used,
        "yield_status": yield_status,
        "yield_reason_code": yield_reason_code,
        "yield_blocker_subreason": yield_subreason,
        "yield_delta_to_pass": delta if _is_num(delta) and float(delta) > 0 else 0.0,
        "yield_calibration_note": calibration_note,
    }


def _build_scout_record(
    *,
    ticker: str,
    as_of_date: str,
    price_row: dict[str, Any],
    facts_row: dict[str, Any],
    net_debt_resolved: dict[str, Any],
    thresholds: dict[str, float],
    require_ev_yield: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    ticker_norm = str(ticker).upper()
    raw_price_status = str(price_row.get("status") or "").upper()
    raw_price_reason = str(price_row.get("reason_code") or "").upper()
    raw_price_value = price_row.get("price")
    if raw_price_status == "OK":
        if _is_num(raw_price_value) and float(raw_price_value) > 0:
            price_status = "OK"
            price_reason = raw_price_reason or "PROVIDER_OK"
            price_reason_detail = ""
            price_value = float(raw_price_value)
        else:
            price_status = "PARTIAL"
            price_reason = "INVALID_PRICE_VALUE"
            price_reason_detail = "Provider returned OK but current_price was missing or non-positive."
            price_value = UNKNOWN
    else:
        price_status = "MISSING"
        price_reason = raw_price_reason or "PROVIDER_NO_DATA"
        price_reason_detail = str(price_row.get("reason_detail") or "No current price snapshot was resolved.")
        price_value = UNKNOWN
    price_refs = [f"prices_summary.rows[{ticker_norm}]"]
    facts_status = str(facts_row.get("status") or "UNKNOWN").upper()
    shares_status = str(facts_row.get("shares_status") or "UNKNOWN").upper()
    cfo_status = str(facts_row.get("cfo_status") or "UNKNOWN").upper()
    capex_status = str(facts_row.get("capex_status") or "UNKNOWN").upper()
    fcf_status = str(facts_row.get("fcf_status") or "UNKNOWN").upper()
    shares_value = facts_row.get("shares_value", UNKNOWN)
    cfo_value = facts_row.get("cfo_value", UNKNOWN)
    capex_value = facts_row.get("capex_value", UNKNOWN)
    fcf_value = facts_row.get("fcf_value", UNKNOWN)
    facts_refs = [str(ref) for ref in (facts_row.get("derived_from") or []) if str(ref).strip()]
    companyfacts = _load_companyfacts_payload_from_facts_row(facts_row)
    owner_payload = compute_owner_earnings_series(
        ticker=ticker_norm,
        as_of_date=as_of_date,
        years_back=5,
        run_id=str(facts_row.get("run_id") or "") or None,
        facts_row=facts_row,
        maintenance_capex_ratio=DEFAULT_MAINT_CAPEX_RATIO,
    )
    maintenance_capex_fundamentals_rows = [
        {
            "year": int(series_row.get("year") or 0),
            "revenue": series_row.get("revenue", UNKNOWN),
            "cfo": series_row.get("cfo", UNKNOWN),
            "capex": series_row.get("capex", UNKNOWN),
            "maintenance_capex_proxy": series_row.get("maintenance_capex_proxy", UNKNOWN),
            "owner_earnings": series_row.get("owner_earnings", UNKNOWN),
        }
        for series_row in (owner_payload.get("series") or [])
        if isinstance(series_row, dict) and int(series_row.get("year") or 0) > 0
    ]
    maintenance_capex_payload = compute_maintenance_capex_discipline(
        ticker=ticker_norm,
        as_of_date=as_of_date,
        fundamentals={
            "ticker": ticker_norm,
            "rows": maintenance_capex_fundamentals_rows,
            "derived_from": list(owner_payload.get("derived_from") or []),
            "maintenance_capex_ratio": owner_payload.get("maintenance_capex_ratio", UNKNOWN),
        },
        owner_payload=owner_payload,
        price_status=price_status,
        facts_status=facts_status,
        shares_status=shares_status,
        fcf_status=fcf_status,
        row_derived_from=facts_refs,
    )
    asset_intensity_class = str(
        maintenance_capex_payload.get("asset_intensity_class") or "ASSET_INTENSITY_UNKNOWN"
    )
    asset_intensity_reason_codes = [
        str(code)
        for code in (maintenance_capex_payload.get("asset_intensity_reason_codes") or [])
        if str(code).strip()
    ]
    maintenance_capex_credibility_class = str(
        maintenance_capex_payload.get("maintenance_capex_credibility_class")
        or "MAINTENANCE_CAPEX_CREDIBILITY_UNKNOWN"
    )
    maintenance_capex_credibility_reason_codes = [
        str(code)
        for code in (maintenance_capex_payload.get("maintenance_capex_credibility_reason_codes") or [])
        if str(code).strip()
    ]
    maintenance_capex_support_signals = [
        str(code)
        for code in (maintenance_capex_payload.get("maintenance_capex_support_signals") or [])
        if str(code).strip()
    ]
    maintenance_capex_headwind_signals = [
        str(code)
        for code in (maintenance_capex_payload.get("maintenance_capex_headwind_signals") or [])
        if str(code).strip()
    ]
    primary_maintenance_capex_caution = str(
        maintenance_capex_payload.get("primary_maintenance_capex_caution") or "OWNER_EARNINGS_UNCLEAR"
    )
    maintenance_capex_discipline_summary = str(
        maintenance_capex_payload.get("maintenance_capex_discipline_summary") or ""
    )
    maintenance_capex_refs = [
        str(ref) for ref in (maintenance_capex_payload.get("derived_from") or []) if str(ref).strip()
    ]
    owner_quality_payload = compute_owner_earnings_quality(
        ticker=ticker_norm,
        as_of_date=as_of_date,
        run_id=str(facts_row.get("run_id") or "") or None,
        facts_row=facts_row,
        owner_payload=owner_payload,
        maintenance_capex_payload=maintenance_capex_payload,
    )
    oe_quality_total = owner_quality_payload.get("oe_quality_total", UNKNOWN)
    owner_earnings_stability_score = owner_quality_payload.get("owner_earnings_stability_score", UNKNOWN)
    capital_allocation_score = owner_quality_payload.get("capital_allocation_score", UNKNOWN)
    cash_conversion_score = owner_quality_payload.get("cash_conversion_score", UNKNOWN)
    oe_quality_reason_codes = [
        str(code)
        for code in (owner_quality_payload.get("oe_quality_reason_codes") or [])
        if str(code).strip()
    ]
    owner_quality_refs = [str(ref) for ref in (owner_quality_payload.get("derived_from") or []) if str(ref).strip()]
    intangible_payload = compute_intangible_economics(
        ticker=ticker_norm,
        as_of_date=as_of_date,
        run_id=str(facts_row.get("run_id") or "") or None,
        facts_row=facts_row,
        owner_payload=owner_payload,
        owner_quality_payload=owner_quality_payload,
        net_debt_resolved=net_debt_resolved,
    )
    gross_margin_durability_score = intangible_payload.get("gross_margin_durability_score", UNKNOWN)
    balance_sheet_optionality_score = intangible_payload.get("balance_sheet_optionality_score", UNKNOWN)
    cycle_resilience_score = intangible_payload.get("cycle_resilience_score", UNKNOWN)
    rnd_productivity_score = intangible_payload.get("rnd_productivity_score", UNKNOWN)
    sga_leverage_score = intangible_payload.get("sga_leverage_score", UNKNOWN)
    owner_value_capture_score = intangible_payload.get("owner_value_capture_score", UNKNOWN)
    intangible_economics_total = intangible_payload.get("intangible_economics_total", UNKNOWN)
    intangible_economics_reason_codes = [
        str(code)
        for code in (intangible_payload.get("intangible_economics_reason_codes") or [])
        if str(code).strip()
    ]
    rnd_productivity_reason_codes = [
        str(code)
        for code in (intangible_payload.get("rnd_productivity_reason_codes") or [])
        if str(code).strip()
    ]
    sga_leverage_reason_codes = [
        str(code)
        for code in (intangible_payload.get("sga_leverage_reason_codes") or [])
        if str(code).strip()
    ]
    owner_value_capture_reason_codes = [
        str(code)
        for code in (intangible_payload.get("owner_value_capture_reason_codes") or [])
        if str(code).strip()
    ]
    intangible_refs = [str(ref) for ref in (intangible_payload.get("derived_from") or []) if str(ref).strip()]
    reinvestment_fundamentals_rows = [
        {
            "year": int(series_row.get("year") or 0),
            "cfo": series_row.get("cfo", UNKNOWN),
            "capex": series_row.get("capex", UNKNOWN),
        }
        for series_row in (owner_payload.get("series") or [])
        if isinstance(series_row, dict) and int(series_row.get("year") or 0) > 0
    ]
    balance_sheet_fundamentals_rows = [
        {
            "year": int(series_row.get("year") or 0),
            "revenue": series_row.get("revenue", UNKNOWN),
            "cfo": series_row.get("cfo", UNKNOWN),
            "fcf": series_row.get("fcf", UNKNOWN),
            "owner_earnings": series_row.get("owner_earnings", UNKNOWN),
        }
        for series_row in (owner_payload.get("series") or [])
        if isinstance(series_row, dict) and int(series_row.get("year") or 0) > 0
    ]
    reinvestment_payload = compute_reinvestment_efficiency(
        ticker=ticker_norm,
        as_of_date=as_of_date,
        fundamentals={
            "ticker": ticker_norm,
            "rows": reinvestment_fundamentals_rows,
            "derived_from": list(owner_payload.get("derived_from") or []),
        },
        owner_quality_payload=owner_quality_payload,
        intangible_payload=intangible_payload,
        maintenance_capex_payload=maintenance_capex_payload,
        evidence_sufficiency_payload={},
        price_status=price_status,
        facts_status=facts_status,
        shares_status=shares_status,
        fcf_status=fcf_status,
        row_derived_from=facts_refs + maintenance_capex_refs + owner_quality_refs + intangible_refs,
    )
    reinvestment_efficiency_class = str(
        reinvestment_payload.get("reinvestment_efficiency_class") or "REINVESTMENT_EFFICIENCY_UNKNOWN"
    )
    reinvestment_efficiency_reason_codes = [
        str(code)
        for code in (reinvestment_payload.get("reinvestment_efficiency_reason_codes") or [])
        if str(code).strip()
    ]
    primary_reinvestment_caution = str(
        reinvestment_payload.get("primary_reinvestment_caution") or "REINVESTMENT_UNCLEAR"
    )
    reinvestment_support_signals = [
        str(code)
        for code in (reinvestment_payload.get("reinvestment_support_signals") or [])
        if str(code).strip()
    ]
    reinvestment_headwind_signals = [
        str(code)
        for code in (reinvestment_payload.get("reinvestment_headwind_signals") or [])
        if str(code).strip()
    ]
    reinvestment_summary = str(reinvestment_payload.get("reinvestment_efficiency_summary") or "")
    reinvestment_refs = [
        str(ref) for ref in (reinvestment_payload.get("derived_from") or []) if str(ref).strip()
    ]
    revenue_dependence_payload = compute_revenue_dependence(
        ticker=ticker_norm,
        as_of_date=as_of_date,
        fundamentals={
            "ticker": ticker_norm,
            "rows": list(reinvestment_fundamentals_rows),
            "derived_from": list(owner_payload.get("derived_from") or []),
        },
        cyclical_normalization_payload={},
        customer_concentration_pct=facts_row.get("customer_concentration_pct", UNKNOWN),
        top_customer_pct=facts_row.get("top_customer_pct", UNKNOWN),
        top_channel_pct=facts_row.get("top_channel_pct", UNKNOWN),
        top_end_market_pct=facts_row.get("top_end_market_pct", UNKNOWN),
        segment_count=facts_row.get("segment_count", UNKNOWN),
        channel_count=facts_row.get("channel_count", UNKNOWN),
        end_market_count=facts_row.get("end_market_count", UNKNOWN),
        customer_concentration_flag=facts_row.get("customer_concentration_signal", UNKNOWN),
        price_status=price_status,
        facts_status=facts_status,
        shares_status=shares_status,
        row_derived_from=facts_refs + owner_quality_refs + intangible_refs + reinvestment_refs,
    )
    revenue_dependence_class = str(
        revenue_dependence_payload.get("revenue_dependence_risk_class") or "REVENUE_DEPENDENCE_UNKNOWN"
    )
    revenue_dependence_reason_codes = [
        str(code)
        for code in (revenue_dependence_payload.get("revenue_dependence_risk_reason_codes") or [])
        if str(code).strip()
    ]
    revenue_dependence_support_signals = [
        str(code)
        for code in (revenue_dependence_payload.get("revenue_dependence_support_signals") or [])
        if str(code).strip()
    ]
    revenue_dependence_headwind_signals = [
        str(code)
        for code in (revenue_dependence_payload.get("revenue_dependence_headwind_signals") or [])
        if str(code).strip()
    ]
    primary_revenue_dependence_caution = str(
        revenue_dependence_payload.get("primary_revenue_dependence_caution") or "REVENUE_BASE_UNCLEAR"
    )
    revenue_fragility_summary = str(revenue_dependence_payload.get("revenue_fragility_summary") or "")
    revenue_dependence_refs = [
        str(ref) for ref in (revenue_dependence_payload.get("derived_from") or []) if str(ref).strip()
    ]
    returns_persistence_payload = compute_returns_persistence(
        ticker=ticker_norm,
        as_of_date=as_of_date,
        fundamentals={
            "ticker": ticker_norm,
            "rows": list(reinvestment_fundamentals_rows),
            "derived_from": list(owner_payload.get("derived_from") or []),
        },
        owner_quality_payload=owner_quality_payload,
        intangible_payload=intangible_payload,
        reinvestment_efficiency_payload=reinvestment_payload,
        revenue_dependence_payload=revenue_dependence_payload,
        capital_allocation_discipline_payload={},
        roic_proxy=facts_row.get("roic_proxy", UNKNOWN),
        price_status=price_status,
        shares_status=shares_status,
        fcf_status=fcf_status,
        facts_status=facts_status,
        row_derived_from=(
            facts_refs
            + owner_quality_refs
            + intangible_refs
            + reinvestment_refs
            + revenue_dependence_refs
        ),
    )
    returns_persistence_class = str(
        returns_persistence_payload.get("returns_persistence_class") or "RETURNS_PERSISTENCE_UNKNOWN"
    )
    returns_persistence_reason_codes = [
        str(code)
        for code in (returns_persistence_payload.get("returns_persistence_reason_codes") or [])
        if str(code).strip()
    ]
    returns_support_signals = [
        str(code)
        for code in (returns_persistence_payload.get("returns_support_signals") or [])
        if str(code).strip()
    ]
    returns_headwind_signals = [
        str(code)
        for code in (returns_persistence_payload.get("returns_headwind_signals") or [])
        if str(code).strip()
    ]
    primary_returns_caution = str(
        returns_persistence_payload.get("primary_returns_caution") or "RETURNS_DURABILITY_UNCLEAR"
    )
    economic_durability_summary = str(
        returns_persistence_payload.get("economic_durability_summary") or ""
    )
    returns_refs = [
        str(ref) for ref in (returns_persistence_payload.get("derived_from") or []) if str(ref).strip()
    ]
    # The accounting-quality module compares cash flow with reported earnings, so its
    # rows need net income. The owner-earnings series carries only cash-flow items;
    # handing the module {year, cfo, capex} left every scouted issuer at
    # ACCOUNTING_QUALITY_UNKNOWN whatever its filings said. Net income comes from the
    # same cached companyfacts, in the same whole-dollar units and annual-period
    # discipline as the series; a year with no filed net income stays UNKNOWN.
    net_income_by_year = {
        int(row.get("year") or 0): row.get("value")
        for row in _annual_only_series_for_priority(
            companyfacts=companyfacts,
            as_of_date=as_of_date,
            priority=_NET_INCOME_TAG_PRIORITY,
            expected_unit_exact=("usd",),
        )
    }
    accounting_fundamentals_rows = []
    for _series_row in owner_payload.get("series") or []:
        if not isinstance(_series_row, dict) or int(_series_row.get("year") or 0) <= 0:
            continue
        _year = int(_series_row["year"])
        _cfo = _series_row.get("cfo", UNKNOWN)
        _capex = _series_row.get("capex", UNKNOWN)
        accounting_fundamentals_rows.append(
            {
                "year": _year,
                "cfo": _cfo,
                "capex": _capex,
                "fcf": (
                    float(_cfo) - abs(float(_capex)) if _is_num(_cfo) and _is_num(_capex) else UNKNOWN
                ),
                "owner_earnings": _series_row.get("owner_earnings", UNKNOWN),
                "net_income": (
                    float(net_income_by_year[_year])
                    if _is_num(net_income_by_year.get(_year))
                    else UNKNOWN
                ),
            }
        )
    accounting_payload = compute_accounting_quality(
        ticker=ticker_norm,
        as_of_date=as_of_date,
        fundamentals={
            "ticker": ticker_norm,
            "rows": accounting_fundamentals_rows,
            "derived_from": list(owner_payload.get("derived_from") or []),
        },
        owner_quality_payload=owner_quality_payload,
        capital_allocation_discipline_payload={},
        reinvestment_efficiency_payload=reinvestment_payload,
        price_status=price_status,
        shares_status=shares_status,
        fcf_status=fcf_status,
        facts_status=facts_status,
        row_derived_from=facts_refs + owner_quality_refs + reinvestment_refs,
    )
    accounting_quality_class = str(
        accounting_payload.get("accounting_quality_class") or "ACCOUNTING_QUALITY_UNKNOWN"
    )
    accounting_quality_reason_codes = [
        str(code)
        for code in (accounting_payload.get("accounting_quality_reason_codes") or [])
        if str(code).strip()
    ]
    cash_earnings_support_signals = [
        str(code)
        for code in (accounting_payload.get("cash_earnings_support_signals") or [])
        if str(code).strip()
    ]
    cash_earnings_headwind_signals = [
        str(code)
        for code in (accounting_payload.get("cash_earnings_headwind_signals") or [])
        if str(code).strip()
    ]
    primary_accounting_caution = str(
        accounting_payload.get("primary_accounting_caution") or "ACCOUNTING_QUALITY_UNCLEAR"
    )
    cash_earnings_discipline_summary = str(
        accounting_payload.get("cash_earnings_discipline_summary") or ""
    )
    accounting_refs = [
        str(ref) for ref in (accounting_payload.get("derived_from") or []) if str(ref).strip()
    ]

    net_debt_payload = net_debt_resolved if isinstance(net_debt_resolved, dict) else {}
    net_debt_proxy = net_debt_payload.get("net_debt_proxy", UNKNOWN)
    net_debt_reason = str(net_debt_payload.get("reason_code") or "EXTRACT_EXCEPTION").upper()
    net_debt_refs = [str(ref) for ref in (net_debt_payload.get("derived_from") or []) if str(ref).strip()]
    net_debt_total_debt = (
        net_debt_payload.get("total_debt", {}).get("value", UNKNOWN)
        if isinstance(net_debt_payload.get("total_debt"), dict)
        else UNKNOWN
    )
    net_debt_cash = (
        net_debt_payload.get("cash_equivalents", {}).get("value", UNKNOWN)
        if isinstance(net_debt_payload.get("cash_equivalents"), dict)
        else UNKNOWN
    )
    balance_sheet_stress_payload = compute_balance_sheet_stress(
        ticker=ticker_norm,
        as_of_date=as_of_date,
        fundamentals={
            "ticker": ticker_norm,
            "rows": list(balance_sheet_fundamentals_rows),
            "derived_from": list(owner_payload.get("derived_from") or []),
        },
        # These rows come from compute_owner_earnings_series, which reports
        # companyfacts values unscaled — whole dollars, not the $millions the
        # net-debt proxy below is in.
        fundamentals_cashflow_units=CASHFLOW_UNITS_USD,
        owner_quality_payload=owner_quality_payload,
        intangible_payload=intangible_payload,
        net_debt_payload=net_debt_payload,
        net_debt_proxy=net_debt_proxy,
        total_debt=net_debt_total_debt,
        cash_equivalents=net_debt_cash,
        price_status=price_status,
        facts_status=facts_status,
        shares_status=shares_status,
        fcf_status=fcf_status,
        row_derived_from=facts_refs + owner_quality_refs + intangible_refs + net_debt_refs,
    )
    balance_sheet_stress_class = str(
        balance_sheet_stress_payload.get("balance_sheet_stress_class") or "BALANCE_SHEET_STRESS_UNKNOWN"
    )
    balance_sheet_stress_reason_codes = [
        str(code)
        for code in (balance_sheet_stress_payload.get("balance_sheet_stress_reason_codes") or [])
        if str(code).strip()
    ]
    refinancing_risk_class = str(
        balance_sheet_stress_payload.get("refinancing_risk_class") or "REFINANCING_RISK_UNKNOWN"
    )
    refinancing_risk_reason_codes = [
        str(code)
        for code in (balance_sheet_stress_payload.get("refinancing_risk_reason_codes") or [])
        if str(code).strip()
    ]
    balance_sheet_support_signals = [
        str(code)
        for code in (balance_sheet_stress_payload.get("balance_sheet_support_signals") or [])
        if str(code).strip()
    ]
    balance_sheet_headwind_signals = [
        str(code)
        for code in (balance_sheet_stress_payload.get("balance_sheet_headwind_signals") or [])
        if str(code).strip()
    ]
    primary_balance_sheet_caution = str(
        balance_sheet_stress_payload.get("primary_balance_sheet_caution") or "BALANCE_SHEET_UNCLEAR"
    )
    balance_sheet_discipline_summary = str(
        balance_sheet_stress_payload.get("balance_sheet_discipline_summary") or ""
    )
    balance_sheet_refs = [
        str(ref) for ref in (balance_sheet_stress_payload.get("derived_from") or []) if str(ref).strip()
    ]
    dilution_rate, dilution_reason, dilution_refs = _derive_dilution_rate(
        companyfacts=companyfacts,
        as_of_date=as_of_date,
    )
    fcf_stability_score, fcf_stability_reason, fcf_stability_refs, fcf_history_values = _derive_fcf_stability(
        companyfacts=companyfacts,
        as_of_date=as_of_date,
    )
    fcf_series, fcf_series_reason, fcf_series_refs = _derive_fcf_series(
        companyfacts=companyfacts,
        as_of_date=as_of_date,
    )

    # Cyclical normalization: extract owner earnings series for variability analysis
    _owner_earnings_for_cyclical = [
        {"year": int(row.get("year") or 0), "value": float(row.get("owner_earnings")), "derived_from": []}
        for row in (owner_payload.get("series") or [])
        if isinstance(row, dict) and _is_num(row.get("owner_earnings")) and int(row.get("year") or 0) > 0
    ]
    cyclical_payload = compute_cyclical_normalization(
        ticker=ticker_norm,
        as_of_date=as_of_date,
        owner_earnings_series=_owner_earnings_for_cyclical,
        fcf_series=fcf_series,
        cfo_series=[],
        # The conservative denominator normalizes margins against revenue.
        revenue_series=annual_revenue_series(companyfacts=companyfacts, as_of_date=as_of_date),
    )

    market_cap = UNKNOWN
    market_cap_reason = "MISSING_MARKET_CAP"
    market_cap_refs: list[str] = []
    if _is_num(price_value) and _is_num(shares_value) and float(shares_value) > 0:
        market_cap = float(price_value) * float(shares_value)
        market_cap_reason = "PRICE_X_SHARES"
        market_cap_refs = [*price_refs, *facts_refs, "derived:market_cap=price*shares"]
    elif _is_num(facts_row.get("market_cap_value")) and float(facts_row.get("market_cap_value")) > 0:
        market_cap = float(facts_row.get("market_cap_value"))
        market_cap_reason = str(facts_row.get("market_cap_reason") or "FACTS_MARKET_CAP")
        market_cap_refs = [*facts_refs]

    ev = UNKNOWN
    ev_status = "UNKNOWN"
    ev_reason_code = "MISSING_MARKET_CAP"
    ev_refs: list[str] = []
    if _is_num(market_cap) and _is_num(net_debt_proxy):
        ev = float(market_cap) + float(net_debt_proxy)
        ev_status = "OK"
        ev_reason_code = "OK"
        ev_refs = sorted(set(market_cap_refs + net_debt_refs + ["derived:ev=market_cap+net_debt_proxy"]))
    elif not _is_num(market_cap):
        ev_status = "UNKNOWN"
        ev_reason_code = "MISSING_MARKET_CAP"
    elif not _is_num(net_debt_proxy):
        ev_status = "UNKNOWN"
        ev_reason_code = "MISSING_NET_DEBT"

    fcf_yield_latest = _yield_value(fcf_value, market_cap)
    fcf_yield_ev_latest = _yield_value(fcf_value, ev)
    yield_profile = _select_yield_metric(
        price_status=price_status,
        shares_status=shares_status,
        market_cap=market_cap,
        ev=ev,
        ev_status=ev_status,
        ev_reason_code=ev_reason_code,
        require_ev_yield=bool(require_ev_yield),
        owner_payload=owner_payload,
        fcf_series=fcf_series,
        fcf_latest=fcf_value,
        thresholds=thresholds,
    )
    yield_gate_value_used = yield_profile.get("yield_gate_value_used", UNKNOWN)
    owner_earnings_yield_3y = yield_profile.get("owner_earnings_yield_3y", UNKNOWN)
    fcf_yield_3y = yield_profile.get("fcf_yield_3y", UNKNOWN)
    yield_status = str(yield_profile.get("yield_status") or "UNKNOWN").upper()
    yield_reason_code = str(yield_profile.get("yield_reason_code") or "UNKNOWN").upper()
    yield_metric_used = str(yield_profile.get("yield_metric_used") or UNKNOWN)
    yield_metric_type = str(yield_profile.get("yield_metric_type") or UNKNOWN)
    yield_denominator_used = str(yield_profile.get("yield_denominator_used") or UNKNOWN).upper()
    yield_delta_to_pass = yield_profile.get("yield_delta_to_pass", UNKNOWN)
    yield_blocker_subreason = str(yield_profile.get("yield_blocker_subreason") or "UNKNOWN")
    yield_calibration_note = str(yield_profile.get("yield_calibration_note") or "")
    owner_earnings_yield_ev_3y = yield_profile.get("owner_earnings_yield_ev_3y", UNKNOWN)
    fcf_yield_ev_3y = yield_profile.get("fcf_yield_ev_3y", UNKNOWN)

    intrinsic_per_share = UNKNOWN
    # The FCF here is cash from operations less capex: CFO is AFTER interest, so
    # 12x it is already a value to EQUITY. Net debt must not be subtracted again
    # (that charged the debt twice: once as interest in the cash flow, once here).
    if _is_num(fcf_value) and float(fcf_value) > 0 and _is_num(shares_value) and float(shares_value) > 0:
        equity_value = float(fcf_value) * 12.0
        intrinsic_per_share = float(equity_value) / float(shares_value)
    valuation_gap = (
        (float(intrinsic_per_share) / float(price_value)) - 1.0
        if _is_num(intrinsic_per_share) and _is_num(price_value) and float(price_value) > 0
        else UNKNOWN
    )
    net_debt_to_cfo = float(net_debt_proxy) / float(cfo_value) if _is_num(net_debt_proxy) and _is_num(cfo_value) and float(cfo_value) > 0 else UNKNOWN
    net_debt_to_fcf = float(net_debt_proxy) / float(fcf_value) if _is_num(net_debt_proxy) and _is_num(fcf_value) and float(fcf_value) > 0 else UNKNOWN
    use_graham_dodd = bool(thresholds.get("scout_use_graham_dodd", True))
    gd_discount_rate = float(thresholds.get("gd_discount_rate", 0.10))
    gd_payload = compute_graham_dodd_overlay(
        ticker=ticker_norm,
        as_of_date=as_of_date,
        price_value=price_value,
        price_status=price_status,
        discount_rate=gd_discount_rate,
        run_id=str(facts_row.get("run_id") or "") or None,
        facts_row=facts_row,
        owner_payload=owner_payload,
    )
    mos_epv = gd_payload.get("mos_epv", UNKNOWN)
    mos_netnet = gd_payload.get("mos_netnet", UNKNOWN)
    epv_per_share = gd_payload.get("epv_per_share", UNKNOWN)
    netnet_per_share = gd_payload.get("netnet_per_share", UNKNOWN)
    gd_value_status = str(gd_payload.get("gd_value_status") or "UNKNOWN").upper()
    gd_primary_reason_code = str(gd_payload.get("gd_primary_reason_code") or BLOCKER_OTHER_UNKNOWN).upper()
    intrinsic_proxy_refs = sorted(
        set(
            facts_refs
            + net_debt_refs
            + [
                "derived:intrinsic_per_share_proxy=fcf_value*12",
                "derived:intrinsic_per_share_proxy/share_count",
            ]
        )
    )
    intrinsic_payload = compute_intrinsic_discipline(
        ticker=ticker_norm,
        as_of_date=as_of_date,
        run_id=str(facts_row.get("run_id") or "") or None,
        facts_row=facts_row,
        owner_payload=owner_payload,
        owner_quality_payload=owner_quality_payload,
        intangible_payload=intangible_payload,
        maintenance_capex_payload=maintenance_capex_payload,
        cyclical_normalization_payload=cyclical_payload,
        price_value=price_value,
        price_refs=price_refs,
        shares_value=shares_value,
        shares_refs=facts_refs,
        net_debt_value=net_debt_proxy,
        net_debt_refs=net_debt_refs,
        epv_per_share=epv_per_share,
        epv_refs=[str(ref) for ref in (gd_payload.get("derived_from") or []) if str(ref).strip()],
        netnet_per_share=netnet_per_share,
        netnet_refs=[str(ref) for ref in (gd_payload.get("derived_from") or []) if str(ref).strip()],
        existing_intrinsic_base=intrinsic_per_share,
        existing_intrinsic_base_refs=intrinsic_proxy_refs,
        existing_intrinsic_conservative=UNKNOWN,
        existing_intrinsic_conservative_refs=[],
        existing_intrinsic_ceiling=intrinsic_per_share,
        existing_intrinsic_ceiling_refs=intrinsic_proxy_refs,
    )
    intrinsic_floor = intrinsic_payload.get("intrinsic_floor", UNKNOWN)
    intrinsic_base = intrinsic_payload.get("intrinsic_base", UNKNOWN)
    intrinsic_ceiling = intrinsic_payload.get("intrinsic_ceiling", UNKNOWN)
    mos_to_floor = intrinsic_payload.get("mos_to_floor", UNKNOWN)
    mos_to_base = intrinsic_payload.get("mos_to_base", UNKNOWN)
    mos_classification = str(intrinsic_payload.get("mos_classification") or UNKNOWN)
    downside_support_type = str(intrinsic_payload.get("downside_support_type") or UNKNOWN)
    downside_support_status = str(intrinsic_payload.get("downside_support_status") or UNKNOWN)
    normalized_earnings_power_value = intrinsic_payload.get("normalized_earnings_power_value", UNKNOWN)
    normalized_earnings_power_method_used = str(
        intrinsic_payload.get("normalized_earnings_power_method_used") or UNKNOWN
    )
    normalized_earnings_power_status = str(
        intrinsic_payload.get("normalized_earnings_power_status") or UNKNOWN
    )
    normalized_earnings_power_reason_codes = [
        str(code)
        for code in (intrinsic_payload.get("normalized_earnings_power_reason_codes") or [])
        if str(code).strip()
    ]
    valuation_range_reason_codes = [
        str(code)
        for code in (intrinsic_payload.get("valuation_range_reason_codes") or [])
        if str(code).strip()
    ]
    downside_support_reason_codes = [
        str(code)
        for code in (intrinsic_payload.get("downside_support_reason_codes") or [])
        if str(code).strip()
    ]
    intrinsic_refs = [str(ref) for ref in (intrinsic_payload.get("derived_from") or []) if str(ref).strip()]
    valuation_confidence_payload = compute_valuation_confidence(
        ticker=ticker_norm,
        as_of_date=as_of_date,
        intrinsic_payload=intrinsic_payload,
        accounting_quality_payload=accounting_payload,
        balance_sheet_stress_payload=balance_sheet_stress_payload,
        maintenance_capex_payload=maintenance_capex_payload,
        price_status=price_status,
        shares_status=shares_status,
        fcf_status=fcf_status,
        facts_status=facts_status,
        valuation_status="OK" if _is_num(valuation_gap) else "UNKNOWN",
        epv_per_share=epv_per_share,
        epv_refs=[str(ref) for ref in (gd_payload.get("derived_from") or []) if str(ref).strip()],
        netnet_per_share=netnet_per_share,
        netnet_refs=[str(ref) for ref in (gd_payload.get("derived_from") or []) if str(ref).strip()],
        existing_intrinsic_base=intrinsic_per_share,
        existing_intrinsic_base_refs=intrinsic_proxy_refs,
        existing_intrinsic_conservative=UNKNOWN,
        existing_intrinsic_conservative_refs=[],
    )
    valuation_support_count = int(valuation_confidence_payload.get("valuation_support_count") or 0)
    valuation_support_types_present = [
        str(value)
        for value in (valuation_confidence_payload.get("valuation_support_types_present") or [])
        if str(value).strip()
    ]
    valuation_support_count_reason_codes = [
        str(code)
        for code in (valuation_confidence_payload.get("valuation_support_count_reason_codes") or [])
        if str(code).strip()
    ]
    valuation_convergence_status = str(
        valuation_confidence_payload.get("valuation_convergence_status") or UNKNOWN
    )
    valuation_convergence_band_pct = valuation_confidence_payload.get(
        "valuation_convergence_band_pct",
        UNKNOWN,
    )
    valuation_convergence_reason_codes = [
        str(code)
        for code in (valuation_confidence_payload.get("valuation_convergence_reason_codes") or [])
        if str(code).strip()
    ]
    valuation_fragility_status = str(
        valuation_confidence_payload.get("valuation_fragility_status") or UNKNOWN
    )
    valuation_fragility_reason_codes = [
        str(code)
        for code in (valuation_confidence_payload.get("valuation_fragility_reason_codes") or [])
        if str(code).strip()
    ]
    valuation_confidence_class = str(
        valuation_confidence_payload.get("valuation_confidence_class") or UNKNOWN
    )
    valuation_confidence_reason_codes = [
        str(code)
        for code in (valuation_confidence_payload.get("valuation_confidence_reason_codes") or [])
        if str(code).strip()
    ]
    valuation_confidence_refs = [
        str(ref)
        for ref in (valuation_confidence_payload.get("derived_from") or [])
        if str(ref).strip()
    ]
    score_valuation_input = (
        max(float(mos_epv), float(mos_netnet))
        if _is_num(mos_epv) and _is_num(mos_netnet)
        else (mos_epv if _is_num(mos_epv) else (mos_netnet if _is_num(mos_netnet) else valuation_gap))
    )

    score_components, score_total, score_flags = _score_components(
        valuation_gap=score_valuation_input,
        fcf_yield=yield_gate_value_used,
        net_debt_to_cfo=net_debt_to_cfo,
        net_debt_to_fcf=net_debt_to_fcf,
        dilution_rate=dilution_rate,
        fcf_stability_score=fcf_stability_score,
        price_known=price_status == "OK",
        shares_known=shares_status == "OK" and _is_num(shares_value),
        fcf_known=fcf_status == "OK" and _is_num(fcf_value),
    )

    reasons: list[str] = []
    blocker_categories: list[str] = []
    negative_cfo = cfo_status == "OK" and _is_num(cfo_value) and float(cfo_value) <= 0
    negative_fcf = fcf_status == "OK" and _is_num(fcf_value) and float(fcf_value) < 0
    missing_all_fundamentals = shares_status != "OK" and cfo_status != "OK" and fcf_status != "OK"
    valuation_fail = _is_num(valuation_gap) and float(valuation_gap) < float(thresholds["scout_valuation_gap_min"])
    valuation_watch = (
        _is_num(valuation_gap)
        and not valuation_fail
        and float(valuation_gap) < float(thresholds["scout_mos_min"])
    )
    gd_epv_fail = _is_num(mos_epv) and float(mos_epv) < float(thresholds["scout_mos_min"])
    gd_netnet_fail = _is_num(mos_netnet) and float(mos_netnet) < float(thresholds["scout_mos_min"])
    gd_has_known = _is_num(mos_epv) or _is_num(mos_netnet)
    gd_pass = (
        (_is_num(mos_epv) and float(mos_epv) >= float(thresholds["scout_mos_min"]))
        or (_is_num(mos_netnet) and float(mos_netnet) >= float(thresholds["scout_mos_min"]))
    )
    gd_close = any(
        [
            _is_num(mos_epv)
            and 0.0 < (float(thresholds["scout_mos_min"]) - float(mos_epv)) <= 0.10,
            _is_num(mos_netnet)
            and 0.0 < (float(thresholds["scout_mos_min"]) - float(mos_netnet)) <= 0.10,
        ]
    )
    excess_net_debt = _is_num(net_debt_to_cfo) and float(net_debt_to_cfo) > float(thresholds["scout_net_debt_to_cfo_max"])
    excess_dilution = _is_num(dilution_rate) and float(dilution_rate) > float(thresholds["scout_dilution_max"])

    if price_status != "OK":
        reasons.append("PRICE_UNKNOWN")
        blocker_categories.append(BLOCKER_MISSING_PRICE)
    if facts_status != "OK":
        reasons.append("FACTS_UNKNOWN")
        blocker_categories.append(BLOCKER_MISSING_FACTS)
    if shares_status != "OK":
        reasons.append("MISSING_SHARES")
        blocker_categories.append(BLOCKER_MISSING_SHARES)
    if fcf_status != "OK":
        reasons.append("MISSING_FCF")
        blocker_categories.append(BLOCKER_MISSING_FCF)
    if cfo_status != "OK":
        reasons.append("MISSING_CFO")
        blocker_categories.append(BLOCKER_MISSING_CFO)
    if capex_status != "OK":
        reasons.append("MISSING_CAPEX")
        blocker_categories.append(BLOCKER_MISSING_CAPEX)
    if missing_all_fundamentals:
        reasons.append("FAIL_MISSING_ALL_FUNDAMENTALS")
        blocker_categories.append(BLOCKER_MISSING_FACTS)
    if negative_cfo:
        reasons.append("FAIL_NEGATIVE_CFO")
        blocker_categories.append(BLOCKER_NEGATIVE_CFO)
    if use_graham_dodd:
        if not gd_has_known:
            reasons.append("MISSING_GD_INPUTS")
            blocker_categories.append(BLOCKER_MISSING_GD_INPUTS)
        else:
            if gd_epv_fail:
                reasons.append("INSUFFICIENT_MOS_EPV")
                blocker_categories.append(BLOCKER_INSUFFICIENT_MOS_EPV)
            if gd_netnet_fail:
                reasons.append("INSUFFICIENT_MOS_NETNET")
                blocker_categories.append(BLOCKER_INSUFFICIENT_MOS_NETNET)
    else:
        if valuation_fail:
            reasons.append("VALUATION_GAP_FAIL")
            blocker_categories.append(BLOCKER_INSUFFICIENT_MOS)
        elif valuation_watch:
            reasons.append("VALUATION_GAP_WATCH")
            blocker_categories.append(BLOCKER_INSUFFICIENT_MOS)
        elif not _is_num(valuation_gap):
            reasons.append("VALUATION_GAP_UNKNOWN")
    if yield_status == "LOW":
        reasons.append(yield_reason_code)
        if yield_reason_code == BLOCKER_LOW_YIELD_OWNER_EARNINGS_EV:
            blocker_categories.append(BLOCKER_LOW_YIELD_OWNER_EARNINGS_EV)
        elif yield_reason_code == BLOCKER_LOW_YIELD_FCF_EV:
            blocker_categories.append(BLOCKER_LOW_YIELD_FCF_EV)
        elif yield_reason_code == BLOCKER_LOW_YIELD_OWNER_EARNINGS:
            blocker_categories.append(BLOCKER_LOW_YIELD_OWNER_EARNINGS)
        elif yield_reason_code == BLOCKER_LOW_YIELD_FCF:
            blocker_categories.append(BLOCKER_LOW_YIELD_FCF)
    elif yield_status != "OK":
        reasons.append(yield_reason_code)
        if yield_reason_code == BLOCKER_MISSING_PRICE:
            blocker_categories.append(BLOCKER_MISSING_PRICE)
        elif yield_reason_code == BLOCKER_MISSING_SHARES:
            blocker_categories.append(BLOCKER_MISSING_SHARES)
        elif yield_reason_code == REASON_MISSING_CFO:
            blocker_categories.append(BLOCKER_MISSING_CFO)
        elif yield_reason_code == REASON_MISSING_CAPEX:
            blocker_categories.append(BLOCKER_MISSING_CAPEX)
        elif yield_reason_code == BLOCKER_MISSING_FCF:
            blocker_categories.append(BLOCKER_MISSING_FCF)
        elif yield_reason_code == REASON_NEGATIVE_CFO:
            blocker_categories.append(BLOCKER_NEGATIVE_CFO)
        elif yield_reason_code == REASON_NEGATIVE_OWNER_EARNINGS:
            blocker_categories.append(BLOCKER_NEGATIVE_FCF)
        elif yield_reason_code == BLOCKER_MISSING_EV:
            blocker_categories.append(BLOCKER_MISSING_EV)
    if yield_calibration_note:
        reasons.append(yield_calibration_note)
    if excess_net_debt:
        reasons.append("NET_DEBT_TO_CFO_EXCESS")
        blocker_categories.append(BLOCKER_EXCESS_NET_DEBT)
    elif not _is_num(net_debt_to_cfo):
        reasons.append("NET_DEBT_TO_CFO_UNKNOWN")
    if excess_dilution:
        reasons.append("DILUTION_EXCESS")
        blocker_categories.append(BLOCKER_EXCESS_DILUTION)
    elif not _is_num(dilution_rate):
        reasons.append("DILUTION_UNKNOWN")
    if not use_graham_dodd and not blocker_categories and not _is_num(valuation_gap):
        blocker_categories.append(BLOCKER_OTHER_UNKNOWN)
    if use_graham_dodd and not blocker_categories and not gd_has_known:
        blocker_categories.append(BLOCKER_MISSING_GD_INPUTS)
    if not blocker_categories and not _is_num(valuation_gap):
        blocker_categories.append(BLOCKER_OTHER_UNKNOWN)
    reasons.extend(score_flags)
    reasons = sorted(set(reasons))
    blocker_categories = _sorted_unique(blocker_categories)

    require_ev_missing = bool(require_ev_yield) and yield_reason_code == BLOCKER_MISSING_EV
    yield_pass = bool(
        yield_status == "OK"
        and _is_num(yield_gate_value_used)
        and float(yield_gate_value_used) >= float(thresholds["scout_fcf_yield_min"])
        and (not bool(require_ev_yield) or yield_denominator_used == "EV")
    )
    hard_fail = any([missing_all_fundamentals, negative_cfo, excess_net_debt, excess_dilution, require_ev_missing])
    if use_graham_dodd:
        if hard_fail:
            scout_status = FAIL
        elif gd_pass and yield_pass:
            scout_status = PASS
        elif not gd_has_known:
            scout_status = WATCH
        elif gd_close or yield_status != "OK":
            scout_status = WATCH
        else:
            scout_status = FAIL
    else:
        has_fail = any([hard_fail, valuation_fail])
        if has_fail:
            scout_status = FAIL
        else:
            pass_floor = bool(
                price_status == "OK"
                and shares_status == "OK"
                and cfo_status == "OK"
                and fcf_status == "OK"
                and _is_num(cfo_value)
                and float(cfo_value) > 0
                and _is_num(fcf_value)
                and float(fcf_value) >= 0
                and _is_num(valuation_gap)
                and float(valuation_gap) >= float(thresholds["scout_mos_min"])
                and yield_pass
                and _is_num(net_debt_to_cfo)
                and float(net_debt_to_cfo) <= float(thresholds["scout_net_debt_to_cfo_max"])
                and _is_num(dilution_rate)
                and float(dilution_rate) <= float(thresholds["scout_dilution_max"])
            )
            scout_status = PASS if pass_floor else WATCH

    if use_graham_dodd and not gd_pass and gd_has_known and not gd_close and scout_status != FAIL and not hard_fail:
        scout_status = FAIL

    primary_blocker_category = _primary_blocker_category(status=scout_status, blocker_categories=blocker_categories)
    near_miss_fields = _near_miss_fields(
        valuation_gap=valuation_gap,
        mos_epv=mos_epv,
        mos_netnet=mos_netnet,
        fcf_yield=yield_gate_value_used,
        net_debt_to_cfo=net_debt_to_cfo,
        dilution_rate=dilution_rate,
        thresholds=thresholds,
    )
    delta_to_pass: float | str
    if str(scout_status).upper() == PASS:
        delta_to_pass = 0.0
    elif {
        BLOCKER_MISSING_PRICE,
        BLOCKER_MISSING_FACTS,
        BLOCKER_MISSING_FCF,
        BLOCKER_MISSING_CFO,
        BLOCKER_MISSING_CAPEX,
        BLOCKER_MISSING_SHARES,
        BLOCKER_MISSING_EV,
        BLOCKER_MISSING_GD_INPUTS,
        BLOCKER_NEGATIVE_CFO,
        BLOCKER_NEGATIVE_FCF,
    } & set(blocker_categories):
        delta_to_pass = UNKNOWN
    elif near_miss_fields:
        delta_to_pass = round(sum(float(row.get("delta") or 0.0) for row in near_miss_fields), 6)
    else:
        delta_to_pass = UNKNOWN

    if not use_graham_dodd and valuation_fail and BLOCKER_INSUFFICIENT_MOS not in blocker_categories:
        blocker_categories.append(BLOCKER_INSUFFICIENT_MOS)
    if use_graham_dodd and scout_status == FAIL and not blocker_categories:
        blocker_categories.append(BLOCKER_OTHER_UNKNOWN)

    if use_graham_dodd and gd_value_status != GD_STATUS_OK and gd_primary_reason_code != GD_REASON_OK:
        reasons.append(gd_primary_reason_code)
    reasons = sorted(set(reasons))
    blocker_categories = _sorted_unique(blocker_categories)

    if use_graham_dodd and gd_pass:
        reasons.append("GRAHAM_DODD_MOS_PASS")
    elif use_graham_dodd and gd_close:
        reasons.append("GRAHAM_DODD_MOS_NEAR_MISS")
    reasons = sorted(set(reasons))

    primary_blocker_category = _primary_blocker_category(status=scout_status, blocker_categories=blocker_categories)

    inputs_used = {
        "current_price": {"value": _to_num(price_value), "derived_from": price_refs},
        "price_reason_code": {"value": price_reason, "derived_from": price_refs},
        "shares_outstanding": {"value": _to_num(shares_value), "derived_from": facts_refs},
        "market_cap": {"value": _to_num(market_cap), "derived_from": market_cap_refs, "reason_code": market_cap_reason},
        "cfo_value": {"value": _to_num(cfo_value), "derived_from": facts_refs},
        "capex_value": {"value": _to_num(capex_value), "derived_from": facts_refs},
        "fcf_value": {"value": _to_num(fcf_value), "derived_from": facts_refs},
        "owner_earnings_series": {
            "value": owner_payload.get("summary", {}).get("owner_earnings_normalized_3y", UNKNOWN)
            if isinstance(owner_payload.get("summary"), dict)
            else UNKNOWN,
            "derived_from": list(owner_payload.get("derived_from") or []),
        },
        "net_debt_proxy": {"value": _to_num(net_debt_proxy), "derived_from": net_debt_refs},
        "total_debt": {"value": _to_num(net_debt_total_debt), "derived_from": net_debt_refs},
        "cash_equivalents": {"value": _to_num(net_debt_cash), "derived_from": net_debt_refs},
        "enterprise_value": {"value": _to_num(ev), "derived_from": ev_refs, "status": ev_status, "reason_code": ev_reason_code},
        "dilution_rate_shares_cagr": {"value": _to_num(dilution_rate), "derived_from": dilution_refs},
        "graham_dodd": {
            "value": {
                "epv_per_share": _to_num(epv_per_share),
                "netnet_per_share": _to_num(netnet_per_share),
                "mos_epv": _to_num(mos_epv),
                "mos_netnet": _to_num(mos_netnet),
            },
            "derived_from": list(gd_payload.get("derived_from") or []),
            "status": gd_value_status,
            "reason_code": gd_primary_reason_code,
        },
        "owner_earnings_quality": {
            "value": {
                "owner_earnings_stability_score": _to_num(owner_earnings_stability_score),
                "capital_allocation_score": _to_num(capital_allocation_score),
                "cash_conversion_score": _to_num(cash_conversion_score),
                "oe_quality_total": _to_num(oe_quality_total),
            },
            "derived_from": owner_quality_refs,
            "reason_codes": oe_quality_reason_codes,
        },
        "modern_intangible_economics": {
            "value": {
                "gross_margin_durability_score": _to_num(gross_margin_durability_score),
                "balance_sheet_optionality_score": _to_num(balance_sheet_optionality_score),
                "cycle_resilience_score": _to_num(cycle_resilience_score),
                "rnd_productivity_score": _to_num(rnd_productivity_score),
                "sga_leverage_score": _to_num(sga_leverage_score),
                "owner_value_capture_score": _to_num(owner_value_capture_score),
                "intangible_economics_total": _to_num(intangible_economics_total),
            },
            "derived_from": intangible_refs,
            "reason_codes": intangible_economics_reason_codes,
        },
            "incremental_reinvestment_efficiency": {
                "value": {
                    "reinvestment_efficiency_class": reinvestment_efficiency_class,
                    "primary_reinvestment_caution": primary_reinvestment_caution,
                },
                "derived_from": reinvestment_refs,
                "reason_codes": reinvestment_efficiency_reason_codes,
                "support_signals": reinvestment_support_signals,
                "headwind_signals": reinvestment_headwind_signals,
            },
            "returns_persistence_economic_durability": {
                "value": {
                    "returns_persistence_class": returns_persistence_class,
                    "primary_returns_caution": primary_returns_caution,
                },
                "derived_from": returns_refs,
                "reason_codes": returns_persistence_reason_codes,
                "support_signals": returns_support_signals,
                "headwind_signals": returns_headwind_signals,
            },
            "intrinsic_value_discipline": {
            "value": {
                "normalized_earnings_power_value": _to_num(normalized_earnings_power_value),
                "intrinsic_floor": _to_num(intrinsic_floor),
                "intrinsic_base": _to_num(intrinsic_base),
                "intrinsic_ceiling": _to_num(intrinsic_ceiling),
                "mos_to_floor": _to_num(mos_to_floor),
                "mos_to_base": _to_num(mos_to_base),
            },
            "derived_from": intrinsic_refs,
            "status": str(intrinsic_payload.get("valuation_range_status") or UNKNOWN),
            "reason_codes": _sorted_unique(
                normalized_earnings_power_reason_codes
                + valuation_range_reason_codes
                + downside_support_reason_codes
            ),
        },
        "valuation_confidence_fragility": {
            "value": {
                "valuation_support_count": int(valuation_support_count),
                "valuation_convergence_status": valuation_convergence_status,
                "valuation_convergence_band_pct": _to_num(valuation_convergence_band_pct),
                "valuation_fragility_status": valuation_fragility_status,
                "valuation_confidence_class": valuation_confidence_class,
            },
            "derived_from": valuation_confidence_refs,
            "reason_codes": _sorted_unique(
                valuation_support_count_reason_codes
                + valuation_convergence_reason_codes
                + valuation_fragility_reason_codes
                + valuation_confidence_reason_codes
            ),
        },
    }

    scoreboard_row = {
        "ticker": ticker_norm,
        "scout_status": scout_status if scout_status in _VALID_SCOUT_STATUSES else WATCH,
        "score_total": float(score_total),
        "score_components": {key: round(float(value), 6) for key, value in sorted(score_components.items())},
        "primary_blocker": primary_blocker_category,
        "primary_blocker_category": primary_blocker_category,
        "blocker_categories": blocker_categories,
        "delta_to_pass": _to_num(delta_to_pass),
        "near_miss_fields": near_miss_fields,
        "recommendation": _recommendation_for_blocker(primary_blocker_category),
        "yield_metric_used": yield_metric_used,
        "yield_gate_value_used": _to_num(yield_gate_value_used),
        "yield_status": yield_status,
        "yield_reason_code": yield_reason_code,
        "yield_blocker_subreason": yield_blocker_subreason,
        "yield_denominator_used": yield_denominator_used,
        "yield_calibration_note": yield_calibration_note,
        "yield_delta_to_pass": yield_delta_to_pass if _is_num(yield_delta_to_pass) else UNKNOWN,
        "ev_status": ev_status,
        "ev_reason_code": ev_reason_code,
        "owner_earnings_stability_score": _to_num(owner_earnings_stability_score),
        "capital_allocation_score": _to_num(capital_allocation_score),
        "cash_conversion_score": _to_num(cash_conversion_score),
        "oe_quality_total": _to_num(oe_quality_total),
        "oe_quality_reason_codes": oe_quality_reason_codes,
        "gross_margin_durability_score": _to_num(gross_margin_durability_score),
        "balance_sheet_optionality_score": _to_num(balance_sheet_optionality_score),
        "cycle_resilience_score": _to_num(cycle_resilience_score),
        "rnd_productivity_score": _to_num(rnd_productivity_score),
        "sga_leverage_score": _to_num(sga_leverage_score),
        "owner_value_capture_score": _to_num(owner_value_capture_score),
        "intangible_economics_total": _to_num(intangible_economics_total),
        "rnd_productivity_reason_codes": rnd_productivity_reason_codes,
        "sga_leverage_reason_codes": sga_leverage_reason_codes,
        "owner_value_capture_reason_codes": owner_value_capture_reason_codes,
        "intangible_economics_reason_codes": intangible_economics_reason_codes,
        "reinvestment_efficiency_class": reinvestment_efficiency_class,
        "reinvestment_efficiency_reason_codes": reinvestment_efficiency_reason_codes,
        "reinvestment_support_signals": reinvestment_support_signals,
        "reinvestment_headwind_signals": reinvestment_headwind_signals,
        "primary_reinvestment_caution": primary_reinvestment_caution,
        "reinvestment_efficiency_summary": reinvestment_summary,
        "asset_intensity_class": asset_intensity_class,
        "asset_intensity_reason_codes": asset_intensity_reason_codes,
        "maintenance_capex_credibility_class": maintenance_capex_credibility_class,
        "maintenance_capex_credibility_reason_codes": maintenance_capex_credibility_reason_codes,
        "maintenance_capex_support_signals": maintenance_capex_support_signals,
        "maintenance_capex_headwind_signals": maintenance_capex_headwind_signals,
        "primary_maintenance_capex_caution": primary_maintenance_capex_caution,
        "maintenance_capex_discipline_summary": maintenance_capex_discipline_summary,
        "returns_persistence_class": returns_persistence_class,
        "returns_persistence_reason_codes": returns_persistence_reason_codes,
        "returns_support_signals": returns_support_signals,
        "returns_headwind_signals": returns_headwind_signals,
        "primary_returns_caution": primary_returns_caution,
        "economic_durability_summary": economic_durability_summary,
        "accounting_quality_class": accounting_quality_class,
        "accounting_quality_reason_codes": accounting_quality_reason_codes,
        "cash_earnings_support_signals": cash_earnings_support_signals,
        "cash_earnings_headwind_signals": cash_earnings_headwind_signals,
        "primary_accounting_caution": primary_accounting_caution,
        "cash_earnings_discipline_summary": cash_earnings_discipline_summary,
        "balance_sheet_stress_class": balance_sheet_stress_class,
        "balance_sheet_stress_reason_codes": balance_sheet_stress_reason_codes,
        "refinancing_risk_class": refinancing_risk_class,
        "refinancing_risk_reason_codes": refinancing_risk_reason_codes,
        "balance_sheet_support_signals": balance_sheet_support_signals,
        "balance_sheet_headwind_signals": balance_sheet_headwind_signals,
        "primary_balance_sheet_caution": primary_balance_sheet_caution,
        "balance_sheet_discipline_summary": balance_sheet_discipline_summary,
        "normalized_earnings_power_value": _to_num(normalized_earnings_power_value),
        "normalized_earnings_power_method_used": normalized_earnings_power_method_used,
        "normalized_earnings_power_status": normalized_earnings_power_status,
        "normalized_earnings_power_reason_codes": normalized_earnings_power_reason_codes,
        "intrinsic_floor": _to_num(intrinsic_floor),
        "intrinsic_base": _to_num(intrinsic_base),
        "intrinsic_ceiling": _to_num(intrinsic_ceiling),
        "mos_to_floor": _to_num(mos_to_floor),
        "mos_to_base": _to_num(mos_to_base),
        "mos_classification": mos_classification,
        "downside_support_type": downside_support_type,
        "downside_support_status": downside_support_status,
        "valuation_range_reason_codes": valuation_range_reason_codes,
        "downside_support_reason_codes": downside_support_reason_codes,
        "valuation_support_count": int(valuation_support_count),
        "valuation_support_types_present": valuation_support_types_present,
        "valuation_support_count_reason_codes": valuation_support_count_reason_codes,
        "valuation_convergence_status": valuation_convergence_status,
        "valuation_convergence_band_pct": _to_num(valuation_convergence_band_pct),
        "valuation_convergence_reason_codes": valuation_convergence_reason_codes,
        "valuation_fragility_status": valuation_fragility_status,
        "valuation_fragility_reason_codes": valuation_fragility_reason_codes,
        "valuation_confidence_class": valuation_confidence_class,
        "valuation_confidence_reason_codes": valuation_confidence_reason_codes,
        "epv_per_share": _to_num(epv_per_share),
        "mos_epv": _to_num(mos_epv),
        "netnet_per_share": _to_num(netnet_per_share),
        "mos_netnet": _to_num(mos_netnet),
        "gd_value_status": gd_value_status,
        "gd_primary_reason_code": gd_primary_reason_code,
        "reasons": reasons,
        "metric_values": {
            "current_price": _to_num(price_value),
            "market_cap": _to_num(market_cap),
            "ev": _to_num(ev),
            "cfo_value": _to_num(cfo_value),
            "capex_value": _to_num(capex_value),
            "fcf_value": _to_num(fcf_value),
            "fcf_yield": _to_num(fcf_yield_latest),
            "fcf_yield_3y": _to_num(fcf_yield_3y),
            "fcf_yield_ev": _to_num(fcf_yield_ev_latest),
            "fcf_yield_ev_3y": _to_num(fcf_yield_ev_3y),
            "owner_earnings_yield_3y": _to_num(owner_earnings_yield_3y),
            "owner_earnings_yield_latest": _to_num(yield_profile.get("owner_earnings_yield_latest", UNKNOWN)),
            "owner_earnings_yield_ev_3y": _to_num(owner_earnings_yield_ev_3y),
            "owner_earnings_yield_ev_latest": _to_num(yield_profile.get("owner_earnings_yield_ev_latest", UNKNOWN)),
            "net_debt_proxy": _to_num(net_debt_proxy),
            "total_debt": _to_num(net_debt_total_debt),
            "cash_equivalents": _to_num(net_debt_cash),
            "net_debt_to_cfo": _to_num(net_debt_to_cfo),
            "net_debt_to_fcf": _to_num(net_debt_to_fcf),
            "dilution_rate": _to_num(dilution_rate),
            "dilution_rate_shares_cagr": _to_num(owner_quality_payload.get("dilution_rate_shares_cagr", dilution_rate)),
            "fcf_stability_score": _to_num(fcf_stability_score),
            "intrinsic_per_share_proxy": _to_num(intrinsic_per_share),
            "valuation_gap": _to_num(valuation_gap),
            "epv_per_share": _to_num(epv_per_share),
            "netnet_per_share": _to_num(netnet_per_share),
            "mos_epv": _to_num(mos_epv),
            "mos_netnet": _to_num(mos_netnet),
            "owner_earnings_positive_years_5y": _to_num(owner_quality_payload.get("owner_earnings_positive_years_5y", UNKNOWN)),
            "owner_earnings_volatility_5y": _to_num(owner_quality_payload.get("owner_earnings_volatility_5y", UNKNOWN)),
            "owner_earnings_stability_score": _to_num(owner_earnings_stability_score),
            "capex_burden_vs_cfo": _to_num(owner_quality_payload.get("capex_burden_vs_cfo", UNKNOWN)),
            "capital_allocation_score": _to_num(capital_allocation_score),
            "cfo_margin_proxy": _to_num(owner_quality_payload.get("cfo_margin_proxy", UNKNOWN)),
            "fcf_conversion_proxy": _to_num(owner_quality_payload.get("fcf_conversion_proxy", UNKNOWN)),
            "cash_conversion_score": _to_num(cash_conversion_score),
            "oe_quality_total": _to_num(oe_quality_total),
            "gross_margin_avg_5y": _to_num(intangible_payload.get("gross_margin_avg_5y", UNKNOWN)),
            "gross_margin_volatility_5y": _to_num(intangible_payload.get("gross_margin_volatility_5y", UNKNOWN)),
            "gross_margin_floor_5y": _to_num(intangible_payload.get("gross_margin_floor_5y", UNKNOWN)),
            "gross_margin_durability_score": _to_num(gross_margin_durability_score),
            "rnd_to_revenue_avg_5y": _to_num(intangible_payload.get("rnd_to_revenue_avg_5y", UNKNOWN)),
            "revenue_per_rnd_proxy": _to_num(intangible_payload.get("revenue_per_rnd_proxy", UNKNOWN)),
            "gross_profit_per_rnd_proxy": _to_num(intangible_payload.get("gross_profit_per_rnd_proxy", UNKNOWN)),
            "owner_earnings_per_rnd_proxy": _to_num(intangible_payload.get("owner_earnings_per_rnd_proxy", UNKNOWN)),
            "rnd_productivity_score": _to_num(rnd_productivity_score),
            "net_debt_proxy": _to_num(intangible_payload.get("net_debt_proxy", net_debt_proxy)),
            "net_debt_to_cfo_proxy": _to_num(intangible_payload.get("net_debt_to_cfo_proxy", UNKNOWN)),
            "cash_pct_revenue": _to_num(intangible_payload.get("cash_pct_revenue", UNKNOWN)),
            "balance_sheet_optionality_score": _to_num(balance_sheet_optionality_score),
            "margin_volatility_5y": _to_num(intangible_payload.get("margin_volatility_5y", UNKNOWN)),
            "fcf_cfo_conversion_median_3y": _to_num(intangible_payload.get("fcf_cfo_conversion_median_3y", UNKNOWN)),
            "fcf_cfo_conversion_volatility_3y": _to_num(intangible_payload.get("fcf_cfo_conversion_volatility_3y", UNKNOWN)),
            "cycle_resilience_score": _to_num(cycle_resilience_score),
            "sga_to_revenue_avg_5y": _to_num(intangible_payload.get("sga_to_revenue_avg_5y", UNKNOWN)),
            "sga_growth_vs_revenue_growth_proxy": _to_num(intangible_payload.get("sga_growth_vs_revenue_growth_proxy", UNKNOWN)),
            "operating_leverage_proxy": _to_num(intangible_payload.get("operating_leverage_proxy", UNKNOWN)),
            "sga_leverage_score": _to_num(sga_leverage_score),
            "revenue_per_share_cagr_proxy": _to_num(intangible_payload.get("revenue_per_share_cagr_proxy", UNKNOWN)),
            "fcf_per_share_cagr_proxy": _to_num(intangible_payload.get("fcf_per_share_cagr_proxy", UNKNOWN)),
            "owner_earnings_per_share_cagr_proxy": _to_num(intangible_payload.get("owner_earnings_per_share_cagr_proxy", UNKNOWN)),
            "owner_value_capture_score": _to_num(owner_value_capture_score),
            "intangible_economics_total": _to_num(intangible_economics_total),
            "reinvestment_efficiency_class": reinvestment_efficiency_class,
            "asset_intensity_class": asset_intensity_class,
            "maintenance_capex_credibility_class": maintenance_capex_credibility_class,
            "revenue_dependence_risk_class": revenue_dependence_class,
            "returns_persistence_class": returns_persistence_class,
            "accounting_quality_class": accounting_quality_class,
            "balance_sheet_stress_class": balance_sheet_stress_class,
            "refinancing_risk_class": refinancing_risk_class,
            "normalized_earnings_power_value": _to_num(normalized_earnings_power_value),
            "intrinsic_floor": _to_num(intrinsic_floor),
            "intrinsic_base": _to_num(intrinsic_base),
            "intrinsic_ceiling": _to_num(intrinsic_ceiling),
            "mos_to_floor": _to_num(mos_to_floor),
            "mos_to_base": _to_num(mos_to_base),
            "valuation_support_count": int(valuation_support_count),
            "valuation_convergence_band_pct": _to_num(valuation_convergence_band_pct),
            "valuation_confidence_class": valuation_confidence_class,
        },
        "inputs_used": inputs_used,
        "graham_dodd_detail": gd_payload,
        "owner_earnings_detail": owner_payload,
        "owner_earnings_quality_detail": owner_quality_payload,
        "maintenance_capex_discipline_detail": maintenance_capex_payload,
        "accounting_quality_detail": accounting_payload,
        "balance_sheet_stress_detail": balance_sheet_stress_payload,
        "intangible_economics_detail": intangible_payload,
        "reinvestment_efficiency_detail": reinvestment_payload,
        "revenue_dependence_detail": revenue_dependence_payload,
        "returns_persistence_detail": returns_persistence_payload,
        "intrinsic_discipline_detail": intrinsic_payload,
        "valuation_confidence_detail": valuation_confidence_payload,
        "derived_from": sorted(
            set(
                price_refs
                + facts_refs
                + net_debt_refs
                + dilution_refs
                + fcf_stability_refs
                + fcf_series_refs
                + [str(ref) for ref in (owner_payload.get("derived_from") or []) if str(ref).strip()]
                + owner_quality_refs
                + maintenance_capex_refs
                + intangible_refs
                + reinvestment_refs
                + revenue_dependence_refs
                + returns_refs
                + accounting_refs
                + balance_sheet_refs
                + intrinsic_refs
                + valuation_confidence_refs
                + market_cap_refs
                + ev_refs
                + [str(ref) for ref in (gd_payload.get("derived_from") or []) if str(ref).strip()]
            )
        ),
    }
    composite_result = compute_composite_score(
        scoreboard_row,
        gd=gd_payload,
        yield_data={
            "yield_gate_value_used": yield_gate_value_used,
            "derived_from": scoreboard_row.get("derived_from") or [],
        },
        fundamentals={
            "roic_proxy": facts_row.get("roic_proxy", UNKNOWN),
            "cfo_value": cfo_value,
            "fcf_value": fcf_value,
            "fcf_margin_trend_slope": facts_row.get("fcf_margin_trend_slope", UNKNOWN),
        },
        valuation_cov={
            "fcf_margin_trend_slope": facts_row.get("fcf_margin_trend_slope", UNKNOWN),
        },
        risk={
            "dilution_rate": dilution_rate,
            "net_debt_to_cfo": net_debt_to_cfo,
            "risk_factor_keyword_delta": facts_row.get("risk_factor_keyword_delta", UNKNOWN),
        },
        thresholds=composite_thresholds_from_config(),
    )
    scoreboard_row["composite_score_total"] = composite_result.get("composite_score_total", 0.0)
    scoreboard_row["gd_score"] = (composite_result.get("components") or {}).get("gd_score", 0.0)
    scoreboard_row["yield_score"] = (composite_result.get("components") or {}).get("yield_score", 0.0)
    scoreboard_row["quality_score"] = (composite_result.get("components") or {}).get("quality_score", 0.0)
    scoreboard_row["risk_penalty"] = (composite_result.get("components") or {}).get("risk_penalty", 0.0)
    scoreboard_row["composite_status"] = str(composite_result.get("status") or "UNKNOWN")
    scoreboard_row["composite_reason_code"] = str(composite_result.get("primary_reason_code") or "UNKNOWN")
    scoreboard_row["composite_detail"] = composite_result

    coverage_row = {
        "ticker": ticker_norm,
        "scout_status": scoreboard_row["scout_status"],
        "primary_blocker": primary_blocker_category,
        "primary_blocker_category": primary_blocker_category,
        "blocker_categories": blocker_categories,
        "near_miss_fields": near_miss_fields,
        "recommendation": _recommendation_for_blocker(primary_blocker_category),
        "yield_metric_used": yield_metric_used,
        "yield_status": yield_status,
        "yield_reason_code": yield_reason_code,
        "yield_blocker_subreason": yield_blocker_subreason,
        "yield_denominator_used": yield_denominator_used,
        "yield_calibration_note": yield_calibration_note,
        "yield_delta_to_pass": yield_delta_to_pass if _is_num(yield_delta_to_pass) else UNKNOWN,
        "price_status": price_status,
        "price_reason_code": price_reason,
        "price_reason_detail": price_reason_detail,
        "current_price": _to_num(price_value),
        "price_asof_used": price_row.get("price_asof_used", UNKNOWN),
        "price_source": price_row.get("source", UNKNOWN),
        "shares_outstanding": _to_num(shares_value),
        "epv_per_share": _to_num(epv_per_share),
        "mos_epv": _to_num(mos_epv),
        "netnet_per_share": _to_num(netnet_per_share),
        "mos_netnet": _to_num(mos_netnet),
        "gd_value_status": gd_value_status,
        "gd_primary_reason_code": gd_primary_reason_code,
        "composite_score_total": scoreboard_row.get("composite_score_total", 0.0),
        "gd_score": scoreboard_row.get("gd_score", 0.0),
        "yield_score": scoreboard_row.get("yield_score", 0.0),
        "quality_score": scoreboard_row.get("quality_score", 0.0),
        "risk_penalty": scoreboard_row.get("risk_penalty", 0.0),
        "owner_earnings_stability_score": _to_num(owner_earnings_stability_score),
        "capital_allocation_score": _to_num(capital_allocation_score),
        "cash_conversion_score": _to_num(cash_conversion_score),
        "oe_quality_total": _to_num(oe_quality_total),
        "oe_quality_reason_codes": oe_quality_reason_codes,
        "gross_margin_durability_score": _to_num(gross_margin_durability_score),
        "balance_sheet_optionality_score": _to_num(balance_sheet_optionality_score),
        "cycle_resilience_score": _to_num(cycle_resilience_score),
        "rnd_productivity_score": _to_num(rnd_productivity_score),
        "sga_leverage_score": _to_num(sga_leverage_score),
        "owner_value_capture_score": _to_num(owner_value_capture_score),
        "intangible_economics_total": _to_num(intangible_economics_total),
        "rnd_productivity_reason_codes": rnd_productivity_reason_codes,
        "sga_leverage_reason_codes": sga_leverage_reason_codes,
        "owner_value_capture_reason_codes": owner_value_capture_reason_codes,
        "intangible_economics_reason_codes": intangible_economics_reason_codes,
        "reinvestment_efficiency_class": reinvestment_efficiency_class,
        "reinvestment_efficiency_reason_codes": reinvestment_efficiency_reason_codes,
        "reinvestment_support_signals": reinvestment_support_signals,
        "reinvestment_headwind_signals": reinvestment_headwind_signals,
        "primary_reinvestment_caution": primary_reinvestment_caution,
        "reinvestment_efficiency_summary": reinvestment_summary,
        "asset_intensity_class": asset_intensity_class,
        "asset_intensity_reason_codes": asset_intensity_reason_codes,
        "maintenance_capex_credibility_class": maintenance_capex_credibility_class,
        "maintenance_capex_credibility_reason_codes": maintenance_capex_credibility_reason_codes,
        "maintenance_capex_support_signals": maintenance_capex_support_signals,
        "maintenance_capex_headwind_signals": maintenance_capex_headwind_signals,
        "primary_maintenance_capex_caution": primary_maintenance_capex_caution,
        "maintenance_capex_discipline_summary": maintenance_capex_discipline_summary,
        "returns_persistence_class": returns_persistence_class,
        "returns_persistence_reason_codes": returns_persistence_reason_codes,
        "returns_support_signals": returns_support_signals,
        "returns_headwind_signals": returns_headwind_signals,
        "primary_returns_caution": primary_returns_caution,
        "economic_durability_summary": economic_durability_summary,
        "accounting_quality_class": accounting_quality_class,
        "accounting_quality_reason_codes": accounting_quality_reason_codes,
        "cash_earnings_support_signals": cash_earnings_support_signals,
        "cash_earnings_headwind_signals": cash_earnings_headwind_signals,
        "primary_accounting_caution": primary_accounting_caution,
        "cash_earnings_discipline_summary": cash_earnings_discipline_summary,
        "balance_sheet_stress_class": balance_sheet_stress_class,
        "balance_sheet_stress_reason_codes": balance_sheet_stress_reason_codes,
        "refinancing_risk_class": refinancing_risk_class,
        "refinancing_risk_reason_codes": refinancing_risk_reason_codes,
        "balance_sheet_support_signals": balance_sheet_support_signals,
        "balance_sheet_headwind_signals": balance_sheet_headwind_signals,
        "primary_balance_sheet_caution": primary_balance_sheet_caution,
        "balance_sheet_discipline_summary": balance_sheet_discipline_summary,
        "normalized_earnings_power_value": _to_num(normalized_earnings_power_value),
        "normalized_earnings_power_method_used": normalized_earnings_power_method_used,
        "normalized_earnings_power_status": normalized_earnings_power_status,
        "normalized_earnings_power_reason_codes": normalized_earnings_power_reason_codes,
        "intrinsic_floor": _to_num(intrinsic_floor),
        "intrinsic_base": _to_num(intrinsic_base),
        "intrinsic_ceiling": _to_num(intrinsic_ceiling),
        "mos_to_floor": _to_num(mos_to_floor),
        "mos_to_base": _to_num(mos_to_base),
        "mos_classification": mos_classification,
        "downside_support_type": downside_support_type,
        "downside_support_status": downside_support_status,
        "valuation_range_reason_codes": valuation_range_reason_codes,
        "downside_support_reason_codes": downside_support_reason_codes,
        "valuation_support_count": int(valuation_support_count),
        "valuation_support_types_present": valuation_support_types_present,
        "valuation_support_count_reason_codes": valuation_support_count_reason_codes,
        "valuation_convergence_status": valuation_convergence_status,
        "valuation_convergence_band_pct": _to_num(valuation_convergence_band_pct),
        "valuation_convergence_reason_codes": valuation_convergence_reason_codes,
        "valuation_fragility_status": valuation_fragility_status,
        "valuation_fragility_reason_codes": valuation_fragility_reason_codes,
        "valuation_confidence_class": valuation_confidence_class,
        "valuation_confidence_reason_codes": valuation_confidence_reason_codes,
        "composite_status": scoreboard_row.get("composite_status", "UNKNOWN"),
        "composite_reason_code": scoreboard_row.get("composite_reason_code", "UNKNOWN"),
        "facts_status": facts_status,
        "fetch_reason_code": str(facts_row.get("fetch_reason_code") or "UNKNOWN"),
        "shares_status": shares_status,
        "shares_reason_code": str(facts_row.get("shares_reason") or "UNKNOWN"),
        "cfo_status": cfo_status,
        "cfo_reason_code": str(facts_row.get("cfo_reason") or "UNKNOWN"),
        "capex_status": capex_status,
        "capex_reason_code": str(facts_row.get("capex_reason") or "UNKNOWN"),
        "fcf_status": fcf_status,
        "fcf_reason_code": str(facts_row.get("fcf_reason") or "UNKNOWN"),
        "net_debt_status": "OK" if _is_num(net_debt_proxy) else "UNKNOWN",
        "net_debt_reason_code": net_debt_reason,
        "market_cap_status": "OK" if _is_num(market_cap) else "UNKNOWN",
        "market_cap_reason_code": market_cap_reason,
        "market_cap": _to_num(market_cap),
        "ev_status": ev_status,
        "ev_reason_code": ev_reason_code,
        "enterprise_value": _to_num(ev),
        "require_ev_yield": bool(require_ev_yield),
        "dilution_status": "OK" if _is_num(dilution_rate) else "UNKNOWN",
        "dilution_reason_code": dilution_reason,
        "fcf_stability_status": "OK" if _is_num(fcf_stability_score) else "UNKNOWN",
        "fcf_stability_reason_code": fcf_stability_reason,
        "valuation_gap_status": "OK" if _is_num(valuation_gap) else "UNKNOWN",
        "valuation_gap_reason_code": "OK" if _is_num(valuation_gap) else "MISSING_INTRINSIC_OR_PRICE",
        "unknown_reasons": sorted(
            {
                code
                for code in [
                    price_reason if price_status != "OK" else "",
                    str(facts_row.get("fetch_reason_code") or "") if facts_status != "OK" else "",
                    str(facts_row.get("shares_reason") or "") if shares_status != "OK" else "",
                    str(facts_row.get("cfo_reason") or "") if cfo_status != "OK" else "",
                    str(facts_row.get("capex_reason") or "") if capex_status != "OK" else "",
                    str(facts_row.get("fcf_reason") or "") if fcf_status != "OK" else "",
                    yield_reason_code if yield_status != "OK" else "",
                    ev_reason_code if str(ev_status).upper() != "OK" else "",
                    net_debt_reason if not _is_num(net_debt_proxy) else "",
                    dilution_reason if not _is_num(dilution_rate) else "",
                    fcf_stability_reason if not _is_num(fcf_stability_score) else "",
                    *([str(code) for code in oe_quality_reason_codes] if not _is_num(oe_quality_total) else []),
                    *(
                        [str(code) for code in intangible_economics_reason_codes]
                        if not _is_num(intangible_economics_total)
                        else []
                    ),
                    *(
                        [str(code) for code in valuation_range_reason_codes]
                        if not _is_num(intrinsic_base)
                        else []
                    ),
                    *(
                        [str(code) for code in valuation_fragility_reason_codes]
                        if valuation_confidence_class == UNKNOWN
                        else []
                    ),
                    "MISSING_INTRINSIC_OR_PRICE" if not _is_num(valuation_gap) else "",
                    gd_primary_reason_code if gd_value_status != GD_STATUS_OK else "",
                ]
                if str(code).strip()
            }
        ),
        "fcf_history_values": [float(value) for value in fcf_history_values],
        "graham_dodd_detail": gd_payload,
        "derived_from": scoreboard_row["derived_from"],
    }
    yield_row = {
        "ticker": ticker_norm,
        "scout_status": scoreboard_row["scout_status"],
        "yield_status": yield_status,
        "yield_reason_code": yield_reason_code,
        "yield_blocker_subreason": yield_blocker_subreason,
        "yield_metric_used": yield_metric_used,
        "yield_metric_type": yield_metric_type,
        "yield_denominator_used": yield_denominator_used,
        "yield_calibration_note": yield_calibration_note,
        "yield_gate_value_used": _to_num(yield_gate_value_used),
        "yield_delta_to_pass": yield_delta_to_pass if _is_num(yield_delta_to_pass) else UNKNOWN,
        "owner_earnings_yield_3y": _to_num(owner_earnings_yield_3y),
        "fcf_yield_3y": _to_num(fcf_yield_3y),
        "owner_earnings_yield_ev_3y": _to_num(owner_earnings_yield_ev_3y),
        "fcf_yield_ev_3y": _to_num(fcf_yield_ev_3y),
        "ev": _to_num(ev),
        "ev_value": _to_num(ev),
        "ev_status": ev_status,
        "ev_reason_code": ev_reason_code,
        "ev_used": yield_denominator_used == "EV",
        "net_debt_proxy_used": _to_num(net_debt_proxy),
        "net_debt_proxy_reason_code": net_debt_reason,
        "denominator_used": yield_denominator_used,
        "primary_blocker_category": primary_blocker_category,
        "epv_per_share": _to_num(epv_per_share),
        "mos_epv": _to_num(mos_epv),
        "netnet_per_share": _to_num(netnet_per_share),
        "mos_netnet": _to_num(mos_netnet),
        "gd_value_status": gd_value_status,
        "gd_primary_reason_code": gd_primary_reason_code,
        "composite_score_total": scoreboard_row.get("composite_score_total", 0.0),
        "gd_score": scoreboard_row.get("gd_score", 0.0),
        "yield_score": scoreboard_row.get("yield_score", 0.0),
        "quality_score": scoreboard_row.get("quality_score", 0.0),
        "risk_penalty": scoreboard_row.get("risk_penalty", 0.0),
        "owner_earnings_stability_score": _to_num(owner_earnings_stability_score),
        "capital_allocation_score": _to_num(capital_allocation_score),
        "cash_conversion_score": _to_num(cash_conversion_score),
        "oe_quality_total": _to_num(oe_quality_total),
        "oe_quality_reason_codes": oe_quality_reason_codes,
        "gross_margin_durability_score": _to_num(gross_margin_durability_score),
        "balance_sheet_optionality_score": _to_num(balance_sheet_optionality_score),
        "cycle_resilience_score": _to_num(cycle_resilience_score),
        "rnd_productivity_score": _to_num(rnd_productivity_score),
        "sga_leverage_score": _to_num(sga_leverage_score),
        "owner_value_capture_score": _to_num(owner_value_capture_score),
        "intangible_economics_total": _to_num(intangible_economics_total),
        "rnd_productivity_reason_codes": rnd_productivity_reason_codes,
        "sga_leverage_reason_codes": sga_leverage_reason_codes,
        "owner_value_capture_reason_codes": owner_value_capture_reason_codes,
        "intangible_economics_reason_codes": intangible_economics_reason_codes,
        "reinvestment_efficiency_class": reinvestment_efficiency_class,
        "reinvestment_efficiency_reason_codes": reinvestment_efficiency_reason_codes,
        "reinvestment_support_signals": reinvestment_support_signals,
        "reinvestment_headwind_signals": reinvestment_headwind_signals,
        "primary_reinvestment_caution": primary_reinvestment_caution,
        "reinvestment_efficiency_summary": reinvestment_summary,
        "asset_intensity_class": asset_intensity_class,
        "asset_intensity_reason_codes": asset_intensity_reason_codes,
        "maintenance_capex_credibility_class": maintenance_capex_credibility_class,
        "maintenance_capex_credibility_reason_codes": maintenance_capex_credibility_reason_codes,
        "maintenance_capex_support_signals": maintenance_capex_support_signals,
        "maintenance_capex_headwind_signals": maintenance_capex_headwind_signals,
        "primary_maintenance_capex_caution": primary_maintenance_capex_caution,
        "maintenance_capex_discipline_summary": maintenance_capex_discipline_summary,
        "returns_persistence_class": returns_persistence_class,
        "returns_persistence_reason_codes": returns_persistence_reason_codes,
        "returns_support_signals": returns_support_signals,
        "returns_headwind_signals": returns_headwind_signals,
        "primary_returns_caution": primary_returns_caution,
        "economic_durability_summary": economic_durability_summary,
        "accounting_quality_class": accounting_quality_class,
        "accounting_quality_reason_codes": accounting_quality_reason_codes,
        "cash_earnings_support_signals": cash_earnings_support_signals,
        "cash_earnings_headwind_signals": cash_earnings_headwind_signals,
        "primary_accounting_caution": primary_accounting_caution,
        "cash_earnings_discipline_summary": cash_earnings_discipline_summary,
        "balance_sheet_stress_class": balance_sheet_stress_class,
        "balance_sheet_stress_reason_codes": balance_sheet_stress_reason_codes,
        "refinancing_risk_class": refinancing_risk_class,
        "refinancing_risk_reason_codes": refinancing_risk_reason_codes,
        "balance_sheet_support_signals": balance_sheet_support_signals,
        "balance_sheet_headwind_signals": balance_sheet_headwind_signals,
        "primary_balance_sheet_caution": primary_balance_sheet_caution,
        "balance_sheet_discipline_summary": balance_sheet_discipline_summary,
        "normalized_earnings_power_value": _to_num(normalized_earnings_power_value),
        "normalized_earnings_power_method_used": normalized_earnings_power_method_used,
        "normalized_earnings_power_status": normalized_earnings_power_status,
        "normalized_earnings_power_reason_codes": normalized_earnings_power_reason_codes,
        "intrinsic_floor": _to_num(intrinsic_floor),
        "intrinsic_base": _to_num(intrinsic_base),
        "intrinsic_ceiling": _to_num(intrinsic_ceiling),
        "mos_to_floor": _to_num(mos_to_floor),
        "mos_to_base": _to_num(mos_to_base),
        "mos_classification": mos_classification,
        "downside_support_type": downside_support_type,
        "downside_support_status": downside_support_status,
        "valuation_range_reason_codes": valuation_range_reason_codes,
        "downside_support_reason_codes": downside_support_reason_codes,
        "valuation_support_count": int(valuation_support_count),
        "valuation_support_types_present": valuation_support_types_present,
        "valuation_support_count_reason_codes": valuation_support_count_reason_codes,
        "valuation_convergence_status": valuation_convergence_status,
        "valuation_convergence_band_pct": _to_num(valuation_convergence_band_pct),
        "valuation_convergence_reason_codes": valuation_convergence_reason_codes,
        "valuation_fragility_status": valuation_fragility_status,
        "valuation_fragility_reason_codes": valuation_fragility_reason_codes,
        "valuation_confidence_class": valuation_confidence_class,
        "valuation_confidence_reason_codes": valuation_confidence_reason_codes,
        "composite_status": scoreboard_row.get("composite_status", "UNKNOWN"),
        "composite_reason_code": scoreboard_row.get("composite_reason_code", "UNKNOWN"),
        "owner_earnings_summary": owner_payload.get("summary") if isinstance(owner_payload.get("summary"), dict) else {},
        "maintenance_capex_ratio": float(owner_payload.get("maintenance_capex_ratio", DEFAULT_MAINT_CAPEX_RATIO))
        if _is_num(owner_payload.get("maintenance_capex_ratio"))
        else float(DEFAULT_MAINT_CAPEX_RATIO),
        "proxy_flags": {
            "maintenance_capex_proxy": True,
            "owner_earnings_proxy": True,
        },
        "inputs_used": {
            "current_price": {"value": _to_num(price_value), "derived_from": price_refs},
            "shares_outstanding": {"value": _to_num(shares_value), "derived_from": facts_refs},
            "owner_earnings_series": {
                "value": owner_payload.get("summary", {}).get("owner_earnings_normalized_3y", UNKNOWN)
                if isinstance(owner_payload.get("summary"), dict)
                else UNKNOWN,
                "derived_from": list(owner_payload.get("derived_from") or []),
            },
            "fcf_series": {
                "value": _to_num([row.get("value") for row in fcf_series if _is_num(row.get("value"))][-1] if fcf_series else UNKNOWN),
                "derived_from": fcf_series_refs,
            },
            "enterprise_value": {"value": _to_num(ev), "derived_from": ev_refs},
            "net_debt_proxy": {"value": _to_num(net_debt_proxy), "derived_from": net_debt_refs},
            "owner_earnings_quality": {
                "value": _to_num(oe_quality_total),
                "derived_from": owner_quality_refs,
            },
            "modern_intangible_economics": {
                "value": _to_num(intangible_economics_total),
                "derived_from": intangible_refs,
            },
            "incremental_reinvestment_efficiency": {
                "value": reinvestment_efficiency_class,
                "derived_from": reinvestment_refs,
            },
            "maintenance_capex_asset_intensity_discipline": {
                "value": {
                    "asset_intensity_class": asset_intensity_class,
                    "maintenance_capex_credibility_class": maintenance_capex_credibility_class,
                    "primary_maintenance_capex_caution": primary_maintenance_capex_caution,
                },
                "derived_from": maintenance_capex_refs,
            },
            "accounting_quality_cash_earnings_discipline": {
                "value": accounting_quality_class,
                "derived_from": accounting_refs,
            },
            "balance_sheet_stress_refinancing_risk": {
                "value": {
                    "balance_sheet_stress_class": balance_sheet_stress_class,
                    "refinancing_risk_class": refinancing_risk_class,
                    "primary_balance_sheet_caution": primary_balance_sheet_caution,
                },
                "derived_from": balance_sheet_refs,
            },
            "intrinsic_value_discipline": {
                "value": _to_num(intrinsic_base),
                "derived_from": intrinsic_refs,
            },
            "valuation_confidence_fragility": {
                "value": valuation_confidence_class,
                "derived_from": valuation_confidence_refs,
            },
        },
        "derived_from": scoreboard_row["derived_from"],
    }
    facts_blocker_fields = enrich_facts_blocker_fields(
        {
            **coverage_row,
            "scout_status": scoreboard_row["scout_status"],
            "blocker_categories": list(scoreboard_row.get("blocker_categories") or []),
            "primary_blocker_category": scoreboard_row.get("primary_blocker_category"),
            "primary_blocker": scoreboard_row.get("primary_blocker"),
        }
    )
    scoreboard_row.update(
        {
            key: value
            for key, value in facts_blocker_fields.items()
            if key
            in {
                "facts_blocker_class",
                "facts_blocker_retryable",
                "facts_blocker_terminal",
                "facts_blocker_partial_usable",
                "facts_missing_key_inputs",
                "facts_retry_recommended",
                "facts_blocker_reason_codes",
                "facts_recommended_action",
                "fail_due_to_missing_evidence",
                "fail_due_to_economic_weakness",
                "primary_fail_domain",
            }
        }
    )
    coverage_row.update(facts_blocker_fields)
    yield_row.update(
        {
            "facts_blocker_class": facts_blocker_fields["facts_blocker_class"],
            "facts_blocker_retryable": facts_blocker_fields["facts_blocker_retryable"],
            "facts_blocker_terminal": facts_blocker_fields["facts_blocker_terminal"],
            "facts_blocker_partial_usable": facts_blocker_fields["facts_blocker_partial_usable"],
            "facts_missing_key_inputs": list(facts_blocker_fields["facts_missing_key_inputs"]),
            "facts_retry_recommended": facts_blocker_fields["facts_retry_recommended"],
            "facts_blocker_reason_codes": list(facts_blocker_fields["facts_blocker_reason_codes"]),
            "facts_recommended_action": facts_blocker_fields["facts_recommended_action"],
            "fail_due_to_missing_evidence": facts_blocker_fields["fail_due_to_missing_evidence"],
            "fail_due_to_economic_weakness": facts_blocker_fields["fail_due_to_economic_weakness"],
            "primary_fail_domain": facts_blocker_fields["primary_fail_domain"],
        }
    )
    value_type_payload = compute_value_type(
        ticker=ticker_norm,
        as_of_date=as_of_date,
        intrinsic_payload=intrinsic_payload,
        valuation_confidence_payload=valuation_confidence_payload,
        owner_quality_payload=owner_quality_payload,
        intangible_payload=intangible_payload,
        accounting_quality_payload=accounting_payload,
        balance_sheet_stress_payload=balance_sheet_stress_payload,
        returns_persistence_payload=returns_persistence_payload,
        revenue_dependence_payload=revenue_dependence_payload,
        reinvestment_efficiency_payload=reinvestment_payload,
        cyclical_normalization_payload=cyclical_payload,
        fail_due_to_missing_evidence=bool(facts_blocker_fields["fail_due_to_missing_evidence"]),
        fail_due_to_economic_weakness=bool(facts_blocker_fields["fail_due_to_economic_weakness"]),
        primary_fail_domain=str(facts_blocker_fields["primary_fail_domain"] or "NONE"),
    )
    value_type_primary = str(value_type_payload.get("value_type_primary") or UNKNOWN)
    value_type_secondary = (
        str(value_type_payload.get("value_type_secondary"))
        if str(value_type_payload.get("value_type_secondary") or "").strip()
        else None
    )
    value_type_reason_codes = [
        str(code)
        for code in (value_type_payload.get("value_type_reason_codes") or [])
        if str(code).strip()
    ]
    value_type_support_summary = str(value_type_payload.get("value_type_support_summary") or "")
    value_type_refs = [
        str(ref)
        for ref in (value_type_payload.get("value_type_derived_from") or value_type_payload.get("derived_from") or [])
        if str(ref).strip()
    ]
    value_type_detail = {
        "value": {
            "value_type_primary": value_type_primary,
            "value_type_secondary": value_type_secondary or UNKNOWN,
            "value_type_support_summary": value_type_support_summary or UNKNOWN,
        },
        "derived_from": value_type_refs,
        "reason_codes": value_type_reason_codes,
    }
    scoreboard_row["value_type_primary"] = value_type_primary
    scoreboard_row["value_type_secondary"] = value_type_secondary
    scoreboard_row["value_type_reason_codes"] = value_type_reason_codes
    scoreboard_row["value_type_support_summary"] = value_type_support_summary
    scoreboard_row["value_type_detail"] = value_type_payload
    scoreboard_row["inputs_used"]["value_type_classification"] = value_type_detail
    scoreboard_row["metric_values"]["value_type_primary"] = value_type_primary
    # Cyclical normalization fields
    scoreboard_row["cyclical_profile_class"] = str(cyclical_payload.get("cyclical_profile_class") or "CYCLICALITY_UNKNOWN")
    scoreboard_row["cycle_position_class"] = str(cyclical_payload.get("cycle_position_class") or "CYCLE_POSITION_UNKNOWN")
    scoreboard_row["cyclical_valuation_risk_class"] = str(cyclical_payload.get("cyclical_valuation_risk_class") or "CYCLE_RISK_UNKNOWN")
    scoreboard_row["conservative_cyclical_denominator"] = cyclical_payload.get("conservative_cyclical_denominator", UNKNOWN)
    scoreboard_row["cycle_aware_value_support_summary"] = str(cyclical_payload.get("cycle_aware_value_support_summary") or "")
    scoreboard_row["cyclical_normalization_detail"] = cyclical_payload
    scoreboard_row["intrinsic_cycle_awareness_status"] = str(intrinsic_payload.get("intrinsic_cycle_awareness_status") or "CYCLICALITY_UNKNOWN/CYCLE_RISK_UNKNOWN")
    scoreboard_row["derived_from"] = sorted(
        set(
            [str(ref) for ref in (scoreboard_row.get("derived_from") or []) if str(ref).strip()]
            + value_type_refs
        )
    )
    coverage_row.update(
        {
            "value_type_primary": value_type_primary,
            "value_type_secondary": value_type_secondary,
            "value_type_reason_codes": value_type_reason_codes,
            "value_type_support_summary": value_type_support_summary,
            "derived_from": scoreboard_row["derived_from"],
        }
    )
    yield_row["inputs_used"]["value_type_classification"] = value_type_detail
    yield_row.update(
        {
            "value_type_primary": value_type_primary,
            "value_type_secondary": value_type_secondary,
            "value_type_reason_codes": value_type_reason_codes,
            "value_type_support_summary": value_type_support_summary,
            "derived_from": scoreboard_row["derived_from"],
        }
    )
    return scoreboard_row, coverage_row, yield_row


def _score_sort_key(row: dict[str, Any]) -> tuple[float, str]:
    score = row.get("score_total", 0.0)
    return (-float(score), str(row.get("ticker") or ""))


def _status_rank(status: str) -> int:
    code = str(status or "").upper()
    if code == PASS:
        return 0
    if code == WATCH:
        return 1
    return 2


def _sector_map_from_taxonomy() -> dict[str, str]:
    cfg = get_config()
    path = cfg.sector_taxonomy_path
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = {str(name).strip().lower() for name in (reader.fieldnames or [])}
            if "ticker" not in fieldnames:
                return {}
            sector_col = "sector"
            for candidate in ["sector", "category", "industry", "group"]:
                if candidate in fieldnames:
                    sector_col = candidate
                    break
            mapping: dict[str, str] = {}
            for row in reader:
                if not isinstance(row, dict):
                    continue
                ticker = _ticker_token(row.get("ticker"))
                if not ticker:
                    continue
                sector = str(row.get(sector_col) or "").strip() or "UNKNOWN_SECTOR"
                mapping[ticker] = sector
            return mapping
    except Exception:
        return {}


def _row_sector(row: dict[str, Any], sector_map: dict[str, str]) -> str:
    ticker = str(row.get("ticker") or "").strip().upper()
    return str(sector_map.get(ticker) or "UNKNOWN_SECTOR")


def _ranked_rows(scoreboard_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(scoreboard_rows, key=ranking_sort_key)


def _serialize_rank_row(row: dict[str, Any], sector_map: dict[str, str]) -> dict[str, Any]:
    metric_values = row.get("metric_values") if isinstance(row.get("metric_values"), dict) else {}
    return {
        "ticker": str(row.get("ticker") or ""),
        "sector": _row_sector(row, sector_map),
        "scout_status": str(row.get("scout_status") or WATCH),
        "composite_score_total": row.get("composite_score_total", UNKNOWN),
        "gd_score": row.get("gd_score", 0.0),
        "yield_score": row.get("yield_score", 0.0),
        "quality_score": row.get("quality_score", 0.0),
        "risk_penalty": row.get("risk_penalty", 0.0),
        "owner_earnings_stability_score": row.get("owner_earnings_stability_score", metric_values.get("owner_earnings_stability_score", UNKNOWN)),
        "capital_allocation_score": row.get("capital_allocation_score", metric_values.get("capital_allocation_score", UNKNOWN)),
        "cash_conversion_score": row.get("cash_conversion_score", metric_values.get("cash_conversion_score", UNKNOWN)),
        "oe_quality_total": row.get("oe_quality_total", metric_values.get("oe_quality_total", UNKNOWN)),
        "oe_quality_reason_codes": [
            str(code)
            for code in (row.get("oe_quality_reason_codes") or [])
            if str(code).strip()
        ],
        "gross_margin_durability_score": row.get(
            "gross_margin_durability_score",
            metric_values.get("gross_margin_durability_score", UNKNOWN),
        ),
        "balance_sheet_optionality_score": row.get(
            "balance_sheet_optionality_score",
            metric_values.get("balance_sheet_optionality_score", UNKNOWN),
        ),
        "cycle_resilience_score": row.get(
            "cycle_resilience_score",
            metric_values.get("cycle_resilience_score", UNKNOWN),
        ),
        "rnd_productivity_score": row.get(
            "rnd_productivity_score",
            metric_values.get("rnd_productivity_score", UNKNOWN),
        ),
        "sga_leverage_score": row.get(
            "sga_leverage_score",
            metric_values.get("sga_leverage_score", UNKNOWN),
        ),
        "owner_value_capture_score": row.get(
            "owner_value_capture_score",
            metric_values.get("owner_value_capture_score", UNKNOWN),
        ),
        "intangible_economics_total": row.get(
            "intangible_economics_total",
            metric_values.get("intangible_economics_total", UNKNOWN),
        ),
        "rnd_productivity_reason_codes": [
            str(code)
            for code in (row.get("rnd_productivity_reason_codes") or [])
            if str(code).strip()
        ],
        "sga_leverage_reason_codes": [
            str(code)
            for code in (row.get("sga_leverage_reason_codes") or [])
            if str(code).strip()
        ],
        "owner_value_capture_reason_codes": [
            str(code)
            for code in (row.get("owner_value_capture_reason_codes") or [])
            if str(code).strip()
        ],
        "reinvestment_efficiency_class": str(
            row.get("reinvestment_efficiency_class")
            or metric_values.get("reinvestment_efficiency_class")
            or "REINVESTMENT_EFFICIENCY_UNKNOWN"
        ),
        "reinvestment_efficiency_reason_codes": [
            str(code)
            for code in (row.get("reinvestment_efficiency_reason_codes") or [])
            if str(code).strip()
        ],
        "reinvestment_support_signals": [
            str(code)
            for code in (row.get("reinvestment_support_signals") or [])
            if str(code).strip()
        ],
        "reinvestment_headwind_signals": [
            str(code)
            for code in (row.get("reinvestment_headwind_signals") or [])
            if str(code).strip()
        ],
        "primary_reinvestment_caution": str(
            row.get("primary_reinvestment_caution") or "REINVESTMENT_UNCLEAR"
        ),
        "reinvestment_efficiency_summary": str(
            row.get("reinvestment_efficiency_summary") or ""
        ),
        "returns_persistence_class": str(
            row.get("returns_persistence_class")
            or metric_values.get("returns_persistence_class")
            or "RETURNS_PERSISTENCE_UNKNOWN"
        ),
        "returns_persistence_reason_codes": [
            str(code)
            for code in (row.get("returns_persistence_reason_codes") or [])
            if str(code).strip()
        ],
        "returns_support_signals": [
            str(code)
            for code in (row.get("returns_support_signals") or [])
            if str(code).strip()
        ],
        "returns_headwind_signals": [
            str(code)
            for code in (row.get("returns_headwind_signals") or [])
            if str(code).strip()
        ],
        "primary_returns_caution": str(
            row.get("primary_returns_caution") or "RETURNS_DURABILITY_UNCLEAR"
        ),
        "economic_durability_summary": str(
            row.get("economic_durability_summary") or ""
        ),
        "accounting_quality_class": str(
            row.get("accounting_quality_class")
            or metric_values.get("accounting_quality_class")
            or "ACCOUNTING_QUALITY_UNKNOWN"
        ),
        "accounting_quality_reason_codes": [
            str(code)
            for code in (row.get("accounting_quality_reason_codes") or [])
            if str(code).strip()
        ],
        "cash_earnings_support_signals": [
            str(code)
            for code in (row.get("cash_earnings_support_signals") or [])
            if str(code).strip()
        ],
        "cash_earnings_headwind_signals": [
            str(code)
            for code in (row.get("cash_earnings_headwind_signals") or [])
            if str(code).strip()
        ],
        "primary_accounting_caution": str(
            row.get("primary_accounting_caution") or "ACCOUNTING_QUALITY_UNCLEAR"
        ),
        "cash_earnings_discipline_summary": str(
            row.get("cash_earnings_discipline_summary") or ""
        ),
        "balance_sheet_stress_class": str(
            row.get("balance_sheet_stress_class")
            or metric_values.get("balance_sheet_stress_class")
            or "BALANCE_SHEET_STRESS_UNKNOWN"
        ),
        "balance_sheet_stress_reason_codes": [
            str(code)
            for code in (row.get("balance_sheet_stress_reason_codes") or [])
            if str(code).strip()
        ],
        "refinancing_risk_class": str(
            row.get("refinancing_risk_class")
            or metric_values.get("refinancing_risk_class")
            or "REFINANCING_RISK_UNKNOWN"
        ),
        "refinancing_risk_reason_codes": [
            str(code)
            for code in (row.get("refinancing_risk_reason_codes") or [])
            if str(code).strip()
        ],
        "balance_sheet_support_signals": [
            str(code)
            for code in (row.get("balance_sheet_support_signals") or [])
            if str(code).strip()
        ],
        "balance_sheet_headwind_signals": [
            str(code)
            for code in (row.get("balance_sheet_headwind_signals") or [])
            if str(code).strip()
        ],
        "primary_balance_sheet_caution": str(
            row.get("primary_balance_sheet_caution") or "BALANCE_SHEET_UNCLEAR"
        ),
        "balance_sheet_discipline_summary": str(
            row.get("balance_sheet_discipline_summary") or ""
        ),
        "normalized_earnings_power_value": row.get(
            "normalized_earnings_power_value",
            metric_values.get("normalized_earnings_power_value", UNKNOWN),
        ),
        "normalized_earnings_power_method_used": str(
            row.get("normalized_earnings_power_method_used")
            or metric_values.get("normalized_earnings_power_method_used")
            or UNKNOWN
        ),
        "normalized_earnings_power_status": str(
            row.get("normalized_earnings_power_status")
            or metric_values.get("normalized_earnings_power_status")
            or UNKNOWN
        ),
        "normalized_earnings_power_reason_codes": [
            str(code)
            for code in (row.get("normalized_earnings_power_reason_codes") or [])
            if str(code).strip()
        ],
        "intrinsic_floor": row.get("intrinsic_floor", metric_values.get("intrinsic_floor", UNKNOWN)),
        "intrinsic_base": row.get("intrinsic_base", metric_values.get("intrinsic_base", UNKNOWN)),
        "intrinsic_ceiling": row.get("intrinsic_ceiling", metric_values.get("intrinsic_ceiling", UNKNOWN)),
        "mos_to_floor": row.get("mos_to_floor", metric_values.get("mos_to_floor", UNKNOWN)),
        "mos_to_base": row.get("mos_to_base", metric_values.get("mos_to_base", UNKNOWN)),
        "mos_classification": str(row.get("mos_classification") or UNKNOWN),
        "downside_support_type": str(row.get("downside_support_type") or UNKNOWN),
        "downside_support_status": str(row.get("downside_support_status") or UNKNOWN),
        "valuation_range_reason_codes": [
            str(code)
            for code in (row.get("valuation_range_reason_codes") or [])
            if str(code).strip()
        ],
        "downside_support_reason_codes": [
            str(code)
            for code in (row.get("downside_support_reason_codes") or [])
            if str(code).strip()
        ],
        "valuation_support_count": int(row.get("valuation_support_count") or 0),
        "valuation_support_types_present": [
            str(value)
            for value in (row.get("valuation_support_types_present") or [])
            if str(value).strip()
        ],
        "valuation_support_count_reason_codes": [
            str(code)
            for code in (row.get("valuation_support_count_reason_codes") or [])
            if str(code).strip()
        ],
        "valuation_convergence_status": str(row.get("valuation_convergence_status") or UNKNOWN),
        "valuation_convergence_band_pct": row.get("valuation_convergence_band_pct", UNKNOWN),
        "valuation_convergence_reason_codes": [
            str(code)
            for code in (row.get("valuation_convergence_reason_codes") or [])
            if str(code).strip()
        ],
        "valuation_fragility_status": str(row.get("valuation_fragility_status") or UNKNOWN),
        "valuation_fragility_reason_codes": [
            str(code)
            for code in (row.get("valuation_fragility_reason_codes") or [])
            if str(code).strip()
        ],
        "valuation_confidence_class": str(row.get("valuation_confidence_class") or UNKNOWN),
        "valuation_confidence_reason_codes": [
            str(code)
            for code in (row.get("valuation_confidence_reason_codes") or [])
            if str(code).strip()
        ],
        "valuation_integrity_class": str(row.get("valuation_integrity_class") or UNKNOWN),
        "valuation_integrity_reason_codes": [
            str(code)
            for code in (row.get("valuation_integrity_reason_codes") or [])
            if str(code).strip()
        ],
        "valuation_consistency_status": str(row.get("valuation_consistency_status") or UNKNOWN),
        "valuation_consistency_reason_codes": [
            str(code)
            for code in (row.get("valuation_consistency_reason_codes") or [])
            if str(code).strip()
        ],
        "valuation_uniformity_group_id": (
            str(row.get("valuation_uniformity_group_id"))
            if str(row.get("valuation_uniformity_group_id") or "").strip()
            else None
        ),
        "valuation_uniformity_reason_codes": [
            str(code)
            for code in (row.get("valuation_uniformity_reason_codes") or [])
            if str(code).strip()
        ],
        "evidence_sufficiency_class": str(row.get("evidence_sufficiency_class") or UNKNOWN),
        "evidence_sufficiency_reason_codes": [
            str(code)
            for code in (row.get("evidence_sufficiency_reason_codes") or [])
            if str(code).strip()
        ],
        "mos_assessment_status": str(row.get("mos_assessment_status") or UNKNOWN),
        "mos_guardrail_reason_codes": [
            str(code)
            for code in (row.get("mos_guardrail_reason_codes") or [])
            if str(code).strip()
        ],
        "investment_readiness_class": str(row.get("investment_readiness_class") or UNKNOWN),
        "investment_readiness_reason_codes": [
            str(code)
            for code in (row.get("investment_readiness_reason_codes") or [])
            if str(code).strip()
        ],
        "blocker_stack_primary": str(row.get("blocker_stack_primary") or UNKNOWN),
        "blocker_stack_secondary": (
            str(row.get("blocker_stack_secondary"))
            if str(row.get("blocker_stack_secondary") or "").strip()
            else None
        ),
        "blocker_stack_all": [
            str(code)
            for code in (row.get("blocker_stack_all") or [])
            if str(code).strip()
        ],
        "blocker_stack_retryable": bool(row.get("blocker_stack_retryable", False)),
        "blocker_stack_structural": bool(row.get("blocker_stack_structural", False)),
        "readiness_support_present": [
            str(code)
            for code in (row.get("readiness_support_present") or [])
            if str(code).strip()
        ],
        "readiness_support_missing": [
            str(code)
            for code in (row.get("readiness_support_missing") or [])
            if str(code).strip()
        ],
        "readiness_support_headwinds": [
            str(code)
            for code in (row.get("readiness_support_headwinds") or [])
            if str(code).strip()
        ],
        "primary_next_step": str(row.get("primary_next_step") or UNKNOWN),
        "primary_next_step_reason": str(row.get("primary_next_step_reason") or UNKNOWN),
        "value_type_primary": str(row.get("value_type_primary") or UNKNOWN),
        "value_type_secondary": (
            str(row.get("value_type_secondary"))
            if str(row.get("value_type_secondary") or "").strip()
            else None
        ),
        "value_type_reason_codes": [
            str(code)
            for code in (row.get("value_type_reason_codes") or [])
            if str(code).strip()
        ],
        "value_type_support_summary": str(row.get("value_type_support_summary") or ""),
        "intangible_economics_reason_codes": [
            str(code)
            for code in (row.get("intangible_economics_reason_codes") or [])
            if str(code).strip()
        ],
        "facts_blocker_class": str(row.get("facts_blocker_class") or "FACTS_OK"),
        "facts_blocker_retryable": bool(row.get("facts_blocker_retryable", False)),
        "facts_blocker_terminal": bool(row.get("facts_blocker_terminal", False)),
        "facts_blocker_partial_usable": bool(row.get("facts_blocker_partial_usable", False)),
        "facts_missing_key_inputs": [str(value) for value in (row.get("facts_missing_key_inputs") or []) if str(value).strip()],
        "facts_retry_recommended": bool(row.get("facts_retry_recommended", False)),
        "facts_blocker_reason_codes": [str(code) for code in (row.get("facts_blocker_reason_codes") or []) if str(code).strip()],
        "facts_recommended_action": str(row.get("facts_recommended_action") or "NONE"),
        "fail_due_to_missing_evidence": bool(row.get("fail_due_to_missing_evidence", False)),
        "fail_due_to_economic_weakness": bool(row.get("fail_due_to_economic_weakness", False)),
        "primary_fail_domain": str(row.get("primary_fail_domain") or "NONE"),
        "composite_status": str(row.get("composite_status") or "UNKNOWN"),
        "composite_reason_code": str(row.get("composite_reason_code") or "UNKNOWN"),
        "primary_blocker_category": str(row.get("primary_blocker_category") or "NONE"),
        "delta_to_pass": row.get("delta_to_pass", UNKNOWN),
        "mos_epv": row.get("mos_epv", metric_values.get("mos_epv", UNKNOWN)),
        "mos_netnet": row.get("mos_netnet", metric_values.get("mos_netnet", UNKNOWN)),
        "yield_gate_value_used": row.get("yield_gate_value_used", UNKNOWN),
        "yield_metric_used": str(row.get("yield_metric_used") or UNKNOWN),
        "yield_denominator_used": str(row.get("yield_denominator_used") or UNKNOWN),
        "gd_value_status": str(row.get("gd_value_status") or "UNKNOWN"),
        "gd_primary_reason_code": str(row.get("gd_primary_reason_code") or UNKNOWN),
        "reasons": list(row.get("reasons") or [])[:8],
        "derived_from": list(row.get("derived_from") or []),
    }


def _top_shortlist_candidates(
    scoreboard_rows: list[dict[str, Any]],
    *,
    top_n: int,
    max_per_sector: int,
    sector_map: dict[str, str],
) -> dict[str, Any]:
    filtered = [
        row for row in scoreboard_rows if str(row.get("scout_status") or "").upper() in {PASS, WATCH}
    ]
    filtered = sorted(filtered, key=ranking_sort_key)
    top_n_eff = max(1, int(top_n))
    max_sector_eff = max(1, int(max_per_sector))

    overall_rows = filtered[:top_n_eff]
    top_candidates = [_serialize_rank_row(row, sector_map) for row in overall_rows]

    by_sector: dict[str, list[dict[str, Any]]] = {}
    for row in filtered:
        sector = _row_sector(row, sector_map)
        bucket = by_sector.setdefault(sector, [])
        if len(bucket) >= max_sector_eff:
            continue
        bucket.append(_serialize_rank_row(row, sector_map))
    top_candidates_by_sector = [
        {"sector": sector, "candidates": rows}
        for sector, rows in sorted(
            by_sector.items(),
            key=lambda item: (item[0] == "UNKNOWN_SECTOR", item[0]),
        )
    ]
    return {
        "top_candidates": top_candidates,
        "top_candidates_by_sector": top_candidates_by_sector,
    }


def _build_rankings_payload(
    *,
    run_id: str,
    as_of_date: str,
    ranked_rows: list[dict[str, Any]],
    top_n: int,
    thresholds_effective: dict[str, Any],
    yield_rows: list[dict[str, Any]],
    coverage_summary: dict[str, Any],
) -> dict[str, Any]:
    top_n_eff = max(1, int(top_n))
    top_overall = ranked_rows[:top_n_eff]
    top_pass = [row for row in ranked_rows if str(row.get("scout_status") or "").upper() == PASS][:top_n_eff]
    top_watch = [row for row in ranked_rows if str(row.get("scout_status") or "").upper() == WATCH][:top_n_eff]
    denominator_counts: dict[str, int] = {}
    for row in yield_rows:
        denom = str(row.get("yield_denominator_used") or UNKNOWN).upper()
        denominator_counts[denom] = denominator_counts.get(denom, 0) + 1
    denominator_counts = dict(sorted(denominator_counts.items(), key=lambda kv: (-kv[1], kv[0])))
    coverage_stats = {
        "unknown_reason_counts": coverage_summary.get("unknown_reason_counts")
        if isinstance(coverage_summary.get("unknown_reason_counts"), dict)
        else {},
        "blocker_counts": coverage_summary.get("blocker_counts")
        if isinstance(coverage_summary.get("blocker_counts"), dict)
        else {},
        "facts_blocker_histogram": coverage_summary.get("facts_blocker_histogram")
        if isinstance(coverage_summary.get("facts_blocker_histogram"), dict)
        else {},
    }
    return {
        "run_id": run_id,
        "as_of_date": as_of_date,
        "ticker_count": len(ranked_rows),
        "thresholds_effective": thresholds_effective,
        "denominator_counts": denominator_counts,
        "coverage_stats": coverage_stats,
        "ranked_rows": ranked_rows,
        "top_overall": top_overall,
        "top_pass": top_pass,
        "top_watch": top_watch,
        "generated_at": utc_now_iso(),
        "derived_from": [
            "universe_scoreboard.rows[*]",
            "yield_coverage.rows[*].yield_denominator_used",
            "universe_coverage.unknown_reason_counts",
            "universe_coverage.blocker_counts",
        ],
    }


def _build_depth_queue_payload(
    *,
    run_id: str,
    as_of_date: str,
    ranked_rows: list[dict[str, Any]],
    cfg,
) -> dict[str, Any]:
    chunk_size = max(1, int(getattr(cfg, "universe_depth_queue_chunk_size", 10)))
    iterations = max(1, int(getattr(cfg, "universe_depth_queue_iterations_default", 2)))
    top_k_default = max(1, int(getattr(cfg, "universe_depth_queue_top_k_default", 10)))
    dossiers_default = max(1, int(getattr(cfg, "universe_depth_queue_limit_dossiers_default", 10)))
    prioritized_candidates = [
        row
        for row in ranked_rows
        if str(row.get("scout_status") or "").upper() in {PASS, WATCH}
    ]
    fallback_used = False
    candidates = prioritized_candidates
    if not candidates:
        candidates = list(ranked_rows)
        fallback_used = True
    entries: list[dict[str, Any]] = []
    for idx, start in enumerate(range(0, len(candidates), chunk_size), start=1):
        chunk = candidates[start : start + chunk_size]
        if not chunk:
            continue
        ticker_list = [str(row.get("ticker") or "") for row in chunk if str(row.get("ticker") or "").strip()]
        if not ticker_list:
            continue
        sector_counts: dict[str, int] = {}
        for row in chunk:
            sector = str(row.get("sector") or "UNKNOWN_SECTOR")
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        sector_suggested = sorted(
            sector_counts.items(),
            key=lambda kv: (-int(kv[1]), str(kv[0])),
        )[0][0]
        top_k = max(1, min(top_k_default, len(ticker_list)))
        limit_dossiers = max(1, min(dossiers_default, len(ticker_list)))
        min_peers_dossierable = max(1, min(len(ticker_list), top_k))
        status_counts = {
            PASS: len([row for row in chunk if str(row.get("scout_status") or "").upper() == PASS]),
            WATCH: len([row for row in chunk if str(row.get("scout_status") or "").upper() == WATCH]),
            FAIL: len([row for row in chunk if str(row.get("scout_status") or "").upper() == FAIL]),
        }
        avg_composite = round(
            sum(float(row.get("composite_score_total") or 0.0) for row in chunk if _is_num(row.get("composite_score_total")))
            / max(1, len([row for row in chunk if _is_num(row.get("composite_score_total"))])),
            6,
        )
        run_id_suggested = f"depth_from_{run_id}_{idx:03d}"
        recommended_flags = {
            "mode": "depth",
            "sector": sector_suggested,
            "iterations": iterations,
            "peer_limit": len(ticker_list),
            "min_peers_dossierable": min_peers_dossierable,
            "limit_dossiers": limit_dossiers,
            "top_k": top_k,
        }
        command = (
            ".venv/bin/python -m app.cli sector-rlm "
            f"--mode depth --sector {sector_suggested} --run-id {run_id_suggested} "
            f"--iterations {iterations} --peer-limit {len(ticker_list)} "
            f"--min-peers-dossierable {min_peers_dossierable} --limit-dossiers {limit_dossiers} "
            f"--top-k {top_k} --tickers {','.join(ticker_list)}"
        )
        entries.append(
            {
                "rank": idx,
                "run_id_suggested": run_id_suggested,
                "tickers": ticker_list,
                "sector_suggested": sector_suggested,
                "status_counts": status_counts,
                "avg_composite_score": avg_composite,
                "rationale": {
                    "top_ticker": ticker_list[0],
                    "top_composite_score": chunk[0].get("composite_score_total", UNKNOWN),
                    "selection_mode": "PASS_WATCH" if not fallback_used else "TOP_OVERALL_FALLBACK",
                    "primary_watch_blockers": sorted(
                        {
                            str(row.get("primary_blocker_category") or "NONE")
                            for row in chunk
                            if str(row.get("scout_status") or "").upper() == WATCH
                        }
                    ),
                },
                "recommended_flags": recommended_flags,
                "command": command,
                "derived_from": [f"universe_rankings.ranked_rows[{row.get('ticker')}]" for row in chunk],
            }
        )
    return {
        "run_id": run_id,
        "as_of_date": as_of_date,
        "selection_mode": "PASS_WATCH" if not fallback_used else "TOP_OVERALL_FALLBACK",
        "queue_count": len(entries),
        "entries": entries,
        "generated_at": utc_now_iso(),
        "derived_from": ["universe_rankings.ranked_rows[*]"],
    }


def _missing_input_breakdown(coverage_rows: list[dict[str, Any]]) -> dict[str, int]:
    breakdown = {
        "price": 0,
        "facts": 0,
        "shares": 0,
        "fcf": 0,
        "cfo": 0,
        "capex": 0,
        "ev": 0,
        "gd_inputs": 0,
    }
    for row in coverage_rows:
        category = str(row.get("primary_blocker_category") or "").upper()
        categories = {str(code).upper() for code in (row.get("blocker_categories") or [])}
        if category == BLOCKER_MISSING_PRICE or BLOCKER_MISSING_PRICE in categories:
            breakdown["price"] += 1
        if category == BLOCKER_MISSING_FACTS or BLOCKER_MISSING_FACTS in categories:
            breakdown["facts"] += 1
        if category == BLOCKER_MISSING_SHARES or BLOCKER_MISSING_SHARES in categories:
            breakdown["shares"] += 1
        if category == BLOCKER_MISSING_FCF or BLOCKER_MISSING_FCF in categories:
            breakdown["fcf"] += 1
        if category == BLOCKER_MISSING_CFO or BLOCKER_MISSING_CFO in categories:
            breakdown["cfo"] += 1
        if category == BLOCKER_MISSING_CAPEX or BLOCKER_MISSING_CAPEX in categories:
            breakdown["capex"] += 1
        if category == BLOCKER_MISSING_EV or BLOCKER_MISSING_EV in categories:
            breakdown["ev"] += 1
        if category == BLOCKER_MISSING_GD_INPUTS or BLOCKER_MISSING_GD_INPUTS in categories:
            breakdown["gd_inputs"] += 1
    return dict(sorted(breakdown.items(), key=lambda kv: (-kv[1], kv[0])))


def _delta_sort_key(value: Any) -> tuple[int, float]:
    if _is_num(value):
        return (0, float(value))
    return (1, float("inf"))


def _top_near_misses(scoreboard_rows: list[dict[str, Any]], top_n: int = 10) -> list[dict[str, Any]]:
    watches = [
        row
        for row in scoreboard_rows
        if str(row.get("scout_status") or "").upper() == WATCH
    ]
    watches.sort(
        key=lambda row: (
            _delta_sort_key(row.get("delta_to_pass")),
            str(row.get("ticker") or ""),
        )
    )
    out: list[dict[str, Any]] = []
    for row in watches[: max(1, int(top_n))]:
        out.append(
            {
                "ticker": str(row.get("ticker") or ""),
                "primary_blocker_category": str(row.get("primary_blocker_category") or BLOCKER_OTHER_UNKNOWN),
                "delta_to_pass": row.get("delta_to_pass", UNKNOWN),
                "yield_metric_used": str(row.get("yield_metric_used") or UNKNOWN),
                "yield_denominator_used": str(row.get("yield_denominator_used") or UNKNOWN),
                "ev_status": str(row.get("ev_status") or UNKNOWN),
                "mos_epv": row.get("mos_epv", UNKNOWN),
                "mos_netnet": row.get("mos_netnet", UNKNOWN),
                "gd_primary_reason_code": str(row.get("gd_primary_reason_code") or UNKNOWN),
                "yield_delta_to_pass": row.get("yield_delta_to_pass", UNKNOWN),
                "near_miss_fields": list(row.get("near_miss_fields") or []),
                "score_total": float(row.get("score_total") or 0.0),
            }
        )
    return out


def _threshold_adjustment_suggestions(
    *,
    scoreboard_rows: list[dict[str, Any]],
    thresholds: dict[str, float],
) -> list[str]:
    suggestions: list[str] = []

    def _best_mos_value(row: dict[str, Any]) -> float | str:
        epv = row.get("mos_epv", UNKNOWN)
        netnet = row.get("mos_netnet", UNKNOWN)
        if _is_num(epv) and _is_num(netnet):
            return max(float(epv), float(netnet))
        if _is_num(epv):
            return float(epv)
        if _is_num(netnet):
            return float(netnet)
        return (row.get("metric_values") or {}).get("valuation_gap", UNKNOWN)

    mos_candidates = [
        row
        for row in scoreboard_rows
        if str(row.get("scout_status") or "").upper() == WATCH
        and str(row.get("primary_blocker_category") or "").upper()
        in {BLOCKER_INSUFFICIENT_MOS, BLOCKER_INSUFFICIENT_MOS_EPV, BLOCKER_INSUFFICIENT_MOS_NETNET}
        and _is_num(_best_mos_value(row))
    ]
    if mos_candidates:
        target = max(float(_best_mos_value(row)) for row in mos_candidates)
        if target < float(thresholds["scout_mos_min"]):
            upgraded = len(
                [
                    row
                    for row in mos_candidates
                    if float(_best_mos_value(row)) >= target
                ]
            )
            suggestions.append(
                f"If you reduce scout_mos_min from {thresholds['scout_mos_min']:.4f} to {target:.4f}, PASS could increase by {upgraded}"
            )

    valuation_candidates = [
        row
        for row in scoreboard_rows
        if str(row.get("scout_status") or "").upper() == FAIL
        and str(row.get("primary_blocker_category") or "").upper() == BLOCKER_INSUFFICIENT_MOS
        and _is_num((row.get("metric_values") or {}).get("valuation_gap"))
    ]
    if valuation_candidates:
        target = max(float((row.get("metric_values") or {}).get("valuation_gap")) for row in valuation_candidates)
        if target < float(thresholds["scout_valuation_gap_min"]):
            upgraded = len(
                [
                    row
                    for row in valuation_candidates
                    if float((row.get("metric_values") or {}).get("valuation_gap")) >= target
                ]
            )
            suggestions.append(
                "If you reduce scout_valuation_gap_min from "
                f"{thresholds['scout_valuation_gap_min']:.4f} to {target:.4f}, WATCH could increase by {upgraded}"
            )

    gd_candidates = [
        row
        for row in scoreboard_rows
        if str(row.get("scout_status") or "").upper() in {WATCH, FAIL}
        and str(row.get("primary_blocker_category") or "").upper()
        in {BLOCKER_INSUFFICIENT_MOS_EPV, BLOCKER_INSUFFICIENT_MOS_NETNET}
        and _is_num(_best_mos_value(row))
    ]
    if gd_candidates:
        target = max(float(_best_mos_value(row)) for row in gd_candidates)
        if target < float(thresholds["scout_mos_min"]):
            upgraded = len([row for row in gd_candidates if float(_best_mos_value(row)) >= target])
            suggestions.append(
                f"If you reduce scout_mos_min from {thresholds['scout_mos_min']:.4f} to {target:.4f}, PASS could increase by {upgraded}"
            )

    fcf_yield_candidates = [
        row
        for row in scoreboard_rows
        if str(row.get("scout_status") or "").upper() in {WATCH, FAIL}
        and str(row.get("primary_blocker_category") or "").upper()
        in {
            BLOCKER_LOW_YIELD_OWNER_EARNINGS,
            BLOCKER_LOW_YIELD_FCF,
            BLOCKER_LOW_YIELD_OWNER_EARNINGS_EV,
            BLOCKER_LOW_YIELD_FCF_EV,
        }
        and _is_num(row.get("yield_gate_value_used"))
    ]
    if fcf_yield_candidates:
        target = max(float(row.get("yield_gate_value_used")) for row in fcf_yield_candidates)
        if target < float(thresholds["scout_fcf_yield_min"]):
            upgraded = len(
                [
                    row
                    for row in fcf_yield_candidates
                    if float(row.get("yield_gate_value_used")) >= target
                ]
            )
            suggestions.append(
                "If you reduce scout_fcf_yield_min from "
                f"{thresholds['scout_fcf_yield_min']:.4f} to {target:.4f}, PASS could increase by {upgraded}"
            )

    net_debt_candidates = [
        row
        for row in scoreboard_rows
        if str(row.get("scout_status") or "").upper() in {WATCH, FAIL}
        and str(row.get("primary_blocker_category") or "").upper() == BLOCKER_EXCESS_NET_DEBT
        and _is_num((row.get("metric_values") or {}).get("net_debt_to_cfo"))
    ]
    if net_debt_candidates:
        target = min(float((row.get("metric_values") or {}).get("net_debt_to_cfo")) for row in net_debt_candidates)
        if target > float(thresholds["scout_net_debt_to_cfo_max"]):
            upgraded = len(
                [
                    row
                    for row in net_debt_candidates
                    if float((row.get("metric_values") or {}).get("net_debt_to_cfo")) <= target
                ]
            )
            suggestions.append(
                "If you increase scout_net_debt_to_cfo_max from "
                f"{thresholds['scout_net_debt_to_cfo_max']:.4f} to {target:.4f}, PASS could increase by {upgraded}"
            )

    dilution_candidates = [
        row
        for row in scoreboard_rows
        if str(row.get("scout_status") or "").upper() in {WATCH, FAIL}
        and str(row.get("primary_blocker_category") or "").upper() == BLOCKER_EXCESS_DILUTION
        and _is_num((row.get("metric_values") or {}).get("dilution_rate"))
    ]
    if dilution_candidates:
        target = min(float((row.get("metric_values") or {}).get("dilution_rate")) for row in dilution_candidates)
        if target > float(thresholds["scout_dilution_max"]):
            upgraded = len(
                [
                    row
                    for row in dilution_candidates
                    if float((row.get("metric_values") or {}).get("dilution_rate")) <= target
                ]
            )
            suggestions.append(
                "If you increase scout_dilution_max from "
                f"{thresholds['scout_dilution_max']:.4f} to {target:.4f}, PASS could increase by {upgraded}"
            )

    return suggestions


def _build_scout_calibration_payload(
    *,
    run_id: str,
    as_of_date: str,
    scoreboard_rows: list[dict[str, Any]],
    coverage_rows: list[dict[str, Any]],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    status_counts = {
        PASS: len([row for row in scoreboard_rows if str(row.get("scout_status") or "").upper() == PASS]),
        WATCH: len([row for row in scoreboard_rows if str(row.get("scout_status") or "").upper() == WATCH]),
        FAIL: len([row for row in scoreboard_rows if str(row.get("scout_status") or "").upper() == FAIL]),
    }
    blocker_counts: dict[str, int] = {}
    yield_blocker_subreason_counts: dict[str, int] = {}
    row_entries: list[dict[str, Any]] = []
    for row in sorted(scoreboard_rows, key=lambda item: str(item.get("ticker") or "")):
        category = str(row.get("primary_blocker_category") or BLOCKER_OTHER_UNKNOWN)
        recommendation = str(row.get("recommendation") or _recommendation_for_blocker(category))
        if str(row.get("yield_calibration_note") or "") == "EV_UNKNOWN_MARKET_CAP_FALLBACK":
            recommendation = f"{recommendation}; hydrate facts (net debt proxy) to enable EV-based yield"
        blocker_counts[category] = blocker_counts.get(category, 0) + 1
        yield_sub = str(row.get("yield_blocker_subreason") or "UNKNOWN")
        if str(category).upper() in {
            BLOCKER_LOW_YIELD_OWNER_EARNINGS,
            BLOCKER_LOW_YIELD_FCF,
            BLOCKER_LOW_YIELD_OWNER_EARNINGS_EV,
            BLOCKER_LOW_YIELD_FCF_EV,
        }:
            yield_blocker_subreason_counts[yield_sub] = yield_blocker_subreason_counts.get(yield_sub, 0) + 1
        row_entries.append(
            {
                "ticker": str(row.get("ticker") or ""),
                "gate_status": str(row.get("scout_status") or WATCH),
                "primary_blocker_category": category,
                "yield_metric_used": str(row.get("yield_metric_used") or UNKNOWN),
                "yield_denominator_used": str(row.get("yield_denominator_used") or UNKNOWN),
                "ev_status": str(row.get("ev_status") or UNKNOWN),
                "gd_value_status": str(row.get("gd_value_status") or UNKNOWN),
                "gd_primary_reason_code": str(row.get("gd_primary_reason_code") or UNKNOWN),
                "mos_epv": row.get("mos_epv", UNKNOWN),
                "mos_netnet": row.get("mos_netnet", UNKNOWN),
                "yield_delta_to_pass": row.get("yield_delta_to_pass", UNKNOWN),
                "yield_blocker_subreason": str(row.get("yield_blocker_subreason") or "UNKNOWN"),
                "near_miss_fields": list(row.get("near_miss_fields") or []),
                "delta_to_pass": row.get("delta_to_pass", UNKNOWN),
                "yield_calibration_note": str(row.get("yield_calibration_note") or ""),
                "recommendation": recommendation,
            }
        )
    blocker_counts = dict(sorted(blocker_counts.items(), key=lambda kv: (-kv[1], kv[0])))
    yield_blocker_subreason_counts = dict(
        sorted(yield_blocker_subreason_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    )
    missing_inputs = _missing_input_breakdown(coverage_rows)
    near_misses = _top_near_misses(scoreboard_rows, top_n=10)
    adjustments = _threshold_adjustment_suggestions(scoreboard_rows=scoreboard_rows, thresholds=thresholds)

    calibration_required = int(status_counts[PASS]) == 0
    top_blockers = [name for name, _count in list(blocker_counts.items())[:3]]
    return {
        "run_id": run_id,
        "as_of_date": as_of_date,
        "generated_at": utc_now_iso(),
        "counts": status_counts,
        "thresholds_effective": {
            **{key: float(thresholds[key]) for key in _THRESHOLD_KEYS},
            "scout_require_ev_yield": bool(thresholds.get("scout_require_ev_yield", False)),
            "scout_use_graham_dodd": bool(thresholds.get("scout_use_graham_dodd", True)),
        },
        "blocker_counts": blocker_counts,
        "yield_blocker_subreason_counts": yield_blocker_subreason_counts,
        "missing_input_breakdown": missing_inputs,
        "top_near_misses": near_misses,
        "suggested_threshold_adjustments": adjustments,
        "rows": row_entries,
        "calibration_required": {
            "required": calibration_required,
            "note_code": "CALIBRATION_REQUIRED" if calibration_required else "NONE",
            "summary": (
                "No PASS tickers in universe scout; inspect blockers and near misses with universe-scout-calibration-open."
                if calibration_required
                else "PASS tickers present."
            ),
            "top_blockers": top_blockers,
            "missing_input_breakdown": missing_inputs,
        },
        "derived_from": [
            "universe_scoreboard.rows[*]",
            "universe_coverage.rows[*]",
        ],
    }


def _json_write(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_json(path, payload, indent=2)


def _safe_json_rows(path: Path, key: str = "rows") -> list[dict[str, Any]]:
    payload = _safe_json(path)
    return [row for row in (payload.get(key) or []) if isinstance(row, dict)]


def _rows_by_ticker(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        ticker = str(row.get("ticker") or "").strip().upper()
        if ticker:
            out[ticker] = row
    return out


def _batch_slices(tickers: list[str], batch_size: int) -> list[list[str]]:
    size = max(1, int(batch_size))
    return [tickers[index : index + size] for index in range(0, len(tickers), size)]


def _effective_thresholds(thresholds: dict[str, Any]) -> dict[str, Any]:
    return {
        **{key: float(thresholds[key]) for key in _THRESHOLD_KEYS},
        "scout_require_ev_yield": bool(thresholds.get("scout_require_ev_yield", False)),
        "scout_use_graham_dodd": bool(thresholds.get("scout_use_graham_dodd", True)),
    }


def _run_paths(*, cfg, run_id: str) -> dict[str, Path]:
    run_dir = cfg.sectors_dir / run_id
    universe_dir = cfg.outputs_dir / "universe" / run_id
    return {
        "run_dir": run_dir,
        "universe_dir": universe_dir,
        "batch_dir": universe_dir / "batches",
        "state_path": universe_dir / "scout_state.json",
        "universe_input_path": universe_dir / "universe_input.json",
        "summary_path": run_dir / "universe_summary.json",
        "scoreboard_path": run_dir / "universe_scoreboard.json",
        "shortlist_path": run_dir / "universe_shortlist.json",
        "coverage_path": run_dir / "universe_coverage.json",
        "calibration_path": run_dir / "universe_scout_calibration.json",
        "yield_path": universe_dir / "yield_coverage.json",
        "net_debt_path": universe_dir / "net_debt_coverage.json",
        "owner_earnings_quality_path": universe_dir / "owner_earnings_quality.json",
        "maintenance_capex_discipline_path": universe_dir / "maintenance_capex_discipline.json",
        "accounting_quality_path": universe_dir / "accounting_quality.json",
        "balance_sheet_stress_path": universe_dir / "balance_sheet_stress.json",
        "returns_persistence_path": universe_dir / "returns_persistence.json",
        "revenue_dependence_path": universe_dir / "revenue_dependence.json",
        "intangible_economics_path": universe_dir / "intangible_economics.json",
        "reinvestment_efficiency_path": universe_dir / "reinvestment_efficiency.json",
        "intrinsic_discipline_path": universe_dir / "intrinsic_discipline.json",
        "evidence_sufficiency_path": universe_dir / "evidence_sufficiency.json",
        "valuation_confidence_path": universe_dir / "valuation_confidence.json",
        "valuation_integrity_path": universe_dir / "valuation_integrity.json",
        "fundamental_regression_analytics_path": universe_dir / "fundamental_regression_analytics.json",
        "fundamental_regression_analytics_md_path": universe_dir / "fundamental_regression_analytics.md",
        "investment_readiness_path": universe_dir / "investment_readiness.json",
        "value_type_path": universe_dir / "value_type.json",
        "cyclical_normalization_path": universe_dir / "cyclical_normalization.json",
        "impairment_classification_path": universe_dir / "impairment_classification.json",
        "normalization_credibility_path": universe_dir / "normalization_credibility.json",
        "capital_allocation_discipline_path": universe_dir / "capital_allocation_discipline.json",
        "graham_dodd_summary_path": universe_dir / "graham_dodd_summary.json",
        "rankings_path": universe_dir / "universe_rankings.json",
        "depth_queue_path": universe_dir / "depth_queue.json",
    }


def _graham_dodd_entry_from_row(row: dict[str, Any], as_of_date: str) -> dict[str, Any]:
    detail = row.get("graham_dodd_detail")
    if isinstance(detail, dict):
        payload = dict(detail)
        payload["ticker"] = str(payload.get("ticker") or row.get("ticker") or "").upper()
        payload["as_of_date"] = str(payload.get("as_of_date") or as_of_date)
        return payload
    ticker = str(row.get("ticker") or "").upper()
    return {
        "ticker": ticker,
        "as_of_date": as_of_date,
        "epv_status": GD_STATUS_OK if _is_num(row.get("epv_per_share")) else "UNKNOWN",
        "epv_reason_code": GD_REASON_OK if _is_num(row.get("epv_per_share")) else UNKNOWN,
        "epv_value": UNKNOWN,
        "epv_per_share": row.get("epv_per_share", UNKNOWN),
        "netnet_status": GD_STATUS_OK if _is_num(row.get("netnet_per_share")) else "UNKNOWN",
        "netnet_reason_code": GD_REASON_OK if _is_num(row.get("netnet_per_share")) else UNKNOWN,
        "netnet_value": UNKNOWN,
        "netnet_per_share": row.get("netnet_per_share", UNKNOWN),
        "mos_epv": row.get("mos_epv", UNKNOWN),
        "mos_netnet": row.get("mos_netnet", UNKNOWN),
        "gd_value_status": str(row.get("gd_value_status") or "UNKNOWN").upper(),
        "gd_primary_reason_code": str(row.get("gd_primary_reason_code") or UNKNOWN).upper(),
        "inputs_used": {},
        "thresholds_used": {},
        "derived_from": [str(ref) for ref in (row.get("derived_from") or []) if str(ref).strip()],
        "generated_at": utc_now_iso(),
    }


def _write_graham_dodd_artifacts(
    *,
    run_id: str,
    as_of_date: str,
    scoreboard_rows: list[dict[str, Any]],
    summary_path: Path,
    universe_dir: Path,
) -> dict[str, Any]:
    entries = [
        _graham_dodd_entry_from_row(row, as_of_date=as_of_date)
        for row in sorted(scoreboard_rows, key=lambda item: str(item.get("ticker") or ""))
    ]
    known_count = len([row for row in entries if str(row.get("gd_value_status") or "").upper() == GD_STATUS_OK])
    unknown_count = len(entries) - known_count
    reason_counts: dict[str, int] = {}
    for row in entries:
        reason = str(row.get("gd_primary_reason_code") or UNKNOWN).upper()
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    reason_counts = dict(sorted(reason_counts.items(), key=lambda kv: (-kv[1], kv[0])))

    def _top_rows(field: str, *, top_n: int = 10) -> list[dict[str, Any]]:
        ranked = [row for row in entries if _is_num(row.get(field))]
        ranked.sort(key=lambda row: (-float(row.get(field) or 0.0), str(row.get("ticker") or "")))
        return [
            {
                "ticker": str(row.get("ticker") or ""),
                field: _to_num(row.get(field)),
                "gd_value_status": str(row.get("gd_value_status") or "UNKNOWN"),
                "gd_primary_reason_code": str(row.get("gd_primary_reason_code") or UNKNOWN),
            }
            for row in ranked[: max(1, int(top_n))]
        ]

    summary_payload = {
        "run_id": run_id,
        "as_of_date": as_of_date,
        "ticker_count": len(entries),
        "gd_known_count": int(known_count),
        "gd_unknown_count": int(unknown_count),
        "reason_counts": reason_counts,
        "top_mos_epv": _top_rows("mos_epv"),
        "top_mos_netnet": _top_rows("mos_netnet"),
        "top_netnet_situations": _top_rows("mos_netnet"),
        "rows": entries,
        "generated_at": utc_now_iso(),
        "derived_from": ["universe_scoreboard.rows[*].graham_dodd_detail"],
    }
    _json_write(summary_path, summary_payload)
    universe_dir.mkdir(parents=True, exist_ok=True)
    for row in entries:
        ticker = str(row.get("ticker") or "").upper()
        if not ticker:
            continue
        _json_write(universe_dir / f"graham_dodd_{ticker}.json", row)
    return summary_payload


def _as_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    if not _is_num(value):
        return None
    return max(0, int(value))


def _batch_counts(
    *,
    scoreboard_rows: list[dict[str, Any]],
    coverage_rows: list[dict[str, Any]],
    yield_rows: list[dict[str, Any]],
) -> dict[str, int]:
    return {
        "price_ok": len([row for row in coverage_rows if str(row.get("price_status") or "").upper() == "OK"]),
        "facts_ok": len([row for row in coverage_rows if str(row.get("facts_status") or "").upper() == "OK"]),
        "yields_ok": len([row for row in yield_rows if str(row.get("yield_status") or "").upper() in {"OK", "LOW"}]),
        PASS: len([row for row in scoreboard_rows if str(row.get("scout_status") or "").upper() == PASS]),
        WATCH: len([row for row in scoreboard_rows if str(row.get("scout_status") or "").upper() == WATCH]),
        FAIL: len([row for row in scoreboard_rows if str(row.get("scout_status") or "").upper() == FAIL]),
    }


def _build_budget_skipped_rows(
    *,
    ticker: str,
    reason_code: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    ticker_norm = str(ticker).upper()
    reason = str(reason_code).upper()
    derived = [f"scout_state.stop_reason_code={reason}"]
    owner_quality_detail = {
        "ticker": ticker_norm,
        "as_of_date": None,
        "source_kind": "SCOUT_BUDGET_SKIPPED",
        "owner_earnings_positive_years_5y": UNKNOWN,
        "owner_earnings_volatility_5y": UNKNOWN,
        "owner_earnings_stability_score": UNKNOWN,
        "owner_earnings_stability_reason_codes": ["OWNER_EARNINGS_UNKNOWN"],
        "dilution_rate_shares_cagr": UNKNOWN,
        "capex_burden_vs_cfo": UNKNOWN,
        "net_debt_change_proxy_3y": UNKNOWN,
        "capital_allocation_score": UNKNOWN,
        "capital_allocation_reason_codes": ["UNKNOWN_CAPITAL_ALLOCATION"],
        "cfo_margin_proxy": UNKNOWN,
        "fcf_conversion_proxy": UNKNOWN,
        "cash_conversion_score": UNKNOWN,
        "cash_conversion_reason_codes": ["CASH_CONVERSION_UNKNOWN"],
        "oe_quality_total": UNKNOWN,
        "oe_quality_reason_codes": ["OWNER_EARNINGS_UNKNOWN", "UNKNOWN_CAPITAL_ALLOCATION", "CASH_CONVERSION_UNKNOWN"],
        "claims": {},
        "derived_from": derived,
        "generated_at": utc_now_iso(),
    }
    intangible_detail = {
        "ticker": ticker_norm,
        "as_of_date": None,
        "source_kind": "SCOUT_BUDGET_SKIPPED",
        "gross_margin_avg_5y": UNKNOWN,
        "gross_margin_volatility_5y": UNKNOWN,
        "gross_margin_floor_5y": UNKNOWN,
        "gross_margin_durability_score": UNKNOWN,
        "gross_margin_durability_reason_codes": ["GROSS_MARGIN_DURABILITY_UNKNOWN"],
        "net_debt_proxy": UNKNOWN,
        "net_debt_to_cfo_proxy": UNKNOWN,
        "cash_pct_revenue": UNKNOWN,
        "balance_sheet_optionality_score": UNKNOWN,
        "balance_sheet_optionality_reason_codes": ["BALANCE_SHEET_OPTIONALITY_UNKNOWN"],
        "margin_volatility_5y": UNKNOWN,
        "fcf_cfo_conversion_median_3y": UNKNOWN,
        "fcf_cfo_conversion_volatility_3y": UNKNOWN,
        "cycle_resilience_score": UNKNOWN,
        "cycle_resilience_reason_codes": ["CYCLE_RESILIENCE_UNKNOWN"],
        "rnd_to_revenue_avg_5y": UNKNOWN,
        "revenue_per_rnd_proxy": UNKNOWN,
        "gross_profit_per_rnd_proxy": UNKNOWN,
        "owner_earnings_per_rnd_proxy": UNKNOWN,
        "rnd_productivity_score": UNKNOWN,
        "rnd_productivity_reason_codes": ["RND_PRODUCTIVITY_UNKNOWN"],
        "sga_to_revenue_avg_5y": UNKNOWN,
        "sga_growth_vs_revenue_growth_proxy": UNKNOWN,
        "operating_leverage_proxy": UNKNOWN,
        "sga_leverage_score": UNKNOWN,
        "sga_leverage_reason_codes": ["SGA_LEVERAGE_UNKNOWN"],
        "dilution_rate_shares_cagr": UNKNOWN,
        "revenue_per_share_cagr_proxy": UNKNOWN,
        "fcf_per_share_cagr_proxy": UNKNOWN,
        "owner_earnings_per_share_cagr_proxy": UNKNOWN,
        "owner_value_capture_score": UNKNOWN,
        "owner_value_capture_reason_codes": ["OWNER_VALUE_CAPTURE_UNKNOWN"],
        "intangible_economics_total": UNKNOWN,
        "intangible_economics_reason_codes": [
            "GROSS_MARGIN_DURABILITY_UNKNOWN",
            "BALANCE_SHEET_OPTIONALITY_UNKNOWN",
            "CYCLE_RESILIENCE_UNKNOWN",
            "RND_PRODUCTIVITY_UNKNOWN",
            "SGA_LEVERAGE_UNKNOWN",
            "OWNER_VALUE_CAPTURE_UNKNOWN",
        ],
        "claims": {},
        "derived_from": derived,
        "generated_at": utc_now_iso(),
    }
    reinvestment_detail = {
        "ticker": ticker_norm,
        "as_of_date": None,
        "revenue_cagr_5y_proxy": UNKNOWN,
        "cfo_cagr_5y_proxy": UNKNOWN,
        "fcf_cagr_5y_proxy": UNKNOWN,
        "shares_cagr_5y_proxy": UNKNOWN,
        "capex_burden_vs_cfo_median_3y": UNKNOWN,
        "reinvestment_efficiency_class": "REINVESTMENT_EFFICIENCY_UNKNOWN",
        "reinvestment_efficiency_reason_codes": [
            "MISSING_REINVESTMENT_INPUTS",
            "REINVESTMENT_EVIDENCE_THIN",
        ],
        "reinvestment_support_signals": [],
        "reinvestment_headwind_signals": ["REINVESTMENT_PRODUCTIVITY_UNCLEAR"],
        "primary_reinvestment_caution": "REINVESTMENT_UNCLEAR",
        "reinvestment_efficiency_summary": "evidence too thin to assess reinvestment productivity responsibly",
        "derived_from": derived,
        "generated_at": utc_now_iso(),
    }
    maintenance_capex_detail = {
        "ticker": ticker_norm,
        "as_of_date": None,
        "asset_intensity_class": "ASSET_INTENSITY_UNKNOWN",
        "asset_intensity_reason_codes": [
            "MISSING_MAINTENANCE_CAPEX_INPUTS",
            "MAINTENANCE_CAPEX_DISCIPLINE_UNKNOWN",
        ],
        "maintenance_capex_credibility_class": "MAINTENANCE_CAPEX_CREDIBILITY_UNKNOWN",
        "maintenance_capex_credibility_reason_codes": [
            "MISSING_MAINTENANCE_CAPEX_INPUTS",
            "MAINTENANCE_CAPEX_DISCIPLINE_UNKNOWN",
        ],
        "maintenance_capex_support_signals": [],
        "maintenance_capex_headwind_signals": ["MAINTENANCE_CAPEX_EVIDENCE_THIN"],
        "primary_maintenance_capex_caution": "OWNER_EARNINGS_UNCLEAR",
        "maintenance_capex_discipline_summary": "evidence too thin to judge whether owner earnings are flattered by sustaining capital needs",
        "derived_from": derived,
        "generated_at": utc_now_iso(),
    }
    returns_persistence_detail = {
        "ticker": ticker_norm,
        "as_of_date": None,
        "returns_persistence_class": "RETURNS_PERSISTENCE_UNKNOWN",
        "returns_persistence_reason_codes": [
            "MISSING_RETURNS_INPUTS",
            "RETURNS_DURABILITY_UNKNOWN",
        ],
        "returns_support_signals": [],
        "returns_headwind_signals": ["RETURNS_EVIDENCE_THIN"],
        "primary_returns_caution": "RETURNS_DURABILITY_UNCLEAR",
        "economic_durability_summary": "evidence too thin to judge whether returns on capital are durable",
        "derived_from": derived,
        "generated_at": utc_now_iso(),
    }
    revenue_dependence_detail = {
        "ticker": ticker_norm,
        "as_of_date": None,
        "revenue_dependence_risk_class": "REVENUE_DEPENDENCE_UNKNOWN",
        "revenue_dependence_risk_reason_codes": [
            "MISSING_REVENUE_DEPENDENCE_INPUTS",
            "REVENUE_DEPENDENCE_UNKNOWN",
        ],
        "revenue_dependence_support_signals": [],
        "revenue_dependence_headwind_signals": ["CONCENTRATION_DISCLOSURE_THIN"],
        "primary_revenue_dependence_caution": "REVENUE_BASE_UNCLEAR",
        "revenue_fragility_summary": "evidence too thin to judge whether revenue depends on a narrow customer, channel, or end market base",
        "derived_from": derived,
        "generated_at": utc_now_iso(),
    }
    accounting_quality_detail = {
        "ticker": ticker_norm,
        "as_of_date": None,
        "accounting_quality_class": "ACCOUNTING_QUALITY_UNKNOWN",
        "accounting_quality_reason_codes": ["MISSING_ACCOUNTING_INPUTS", "ACCOUNTING_EVIDENCE_THIN"],
        "cash_earnings_support_signals": [],
        "cash_earnings_headwind_signals": ["CASH_EARNINGS_DIVERGENCE"],
        "primary_accounting_caution": "ACCOUNTING_QUALITY_UNCLEAR",
        "cash_earnings_discipline_summary": "evidence too thin to judge whether reported earnings are cash-backed",
        "derived_from": derived,
        "generated_at": utc_now_iso(),
    }
    balance_sheet_stress_detail = {
        "ticker": ticker_norm,
        "as_of_date": None,
        "balance_sheet_stress_class": "BALANCE_SHEET_STRESS_UNKNOWN",
        "balance_sheet_stress_reason_codes": [
            "INSUFFICIENT_DEBT_INPUTS",
            "BALANCE_SHEET_EVIDENCE_THIN",
        ],
        "refinancing_risk_class": "REFINANCING_RISK_UNKNOWN",
        "refinancing_risk_reason_codes": [
            "INSUFFICIENT_DEBT_INPUTS",
            "BALANCE_SHEET_EVIDENCE_THIN",
        ],
        "balance_sheet_support_signals": [],
        "balance_sheet_headwind_signals": ["DEBT_INPUTS_THIN"],
        "primary_balance_sheet_caution": "BALANCE_SHEET_UNCLEAR",
        "balance_sheet_discipline_summary": "evidence too thin to judge whether capital structure supports value realization",
        "derived_from": derived,
        "generated_at": utc_now_iso(),
    }
    scoreboard_row = {
        "ticker": ticker_norm,
        "scout_status": WATCH,
        "score_total": 0.0,
        "score_components": {
            "valuation_mos": 0.0,
            "fcf_yield": 0.0,
            "balance_sheet": 0.0,
            "dilution_discipline": 0.0,
            "quality_stability": 0.0,
            "unknown_penalty": 0.0,
        },
        "primary_blocker": BLOCKER_OTHER_UNKNOWN,
        "primary_blocker_category": BLOCKER_OTHER_UNKNOWN,
        "blocker_categories": [BLOCKER_OTHER_UNKNOWN],
        "delta_to_pass": UNKNOWN,
        "near_miss_fields": [],
        "recommendation": "resume universe-scout with higher budget limits",
        "yield_metric_used": UNKNOWN,
        "yield_gate_value_used": UNKNOWN,
        "yield_status": UNKNOWN,
        "yield_reason_code": reason,
        "yield_blocker_subreason": reason,
        "yield_denominator_used": UNKNOWN,
        "yield_calibration_note": "SKIPPED_BUDGET",
        "yield_delta_to_pass": UNKNOWN,
        "ev_status": UNKNOWN,
        "ev_reason_code": "MISSING_MARKET_CAP",
        "owner_earnings_stability_score": UNKNOWN,
        "capital_allocation_score": UNKNOWN,
        "cash_conversion_score": UNKNOWN,
        "oe_quality_total": UNKNOWN,
        "oe_quality_reason_codes": list(owner_quality_detail["oe_quality_reason_codes"]),
        "gross_margin_durability_score": UNKNOWN,
        "balance_sheet_optionality_score": UNKNOWN,
        "cycle_resilience_score": UNKNOWN,
        "rnd_productivity_score": UNKNOWN,
        "sga_leverage_score": UNKNOWN,
        "owner_value_capture_score": UNKNOWN,
        "intangible_economics_total": UNKNOWN,
        "rnd_productivity_reason_codes": list(intangible_detail["rnd_productivity_reason_codes"]),
        "sga_leverage_reason_codes": list(intangible_detail["sga_leverage_reason_codes"]),
        "owner_value_capture_reason_codes": list(intangible_detail["owner_value_capture_reason_codes"]),
        "intangible_economics_reason_codes": list(intangible_detail["intangible_economics_reason_codes"]),
        "reinvestment_efficiency_class": "REINVESTMENT_EFFICIENCY_UNKNOWN",
        "reinvestment_efficiency_reason_codes": list(reinvestment_detail["reinvestment_efficiency_reason_codes"]),
        "reinvestment_support_signals": [],
        "reinvestment_headwind_signals": list(reinvestment_detail["reinvestment_headwind_signals"]),
        "primary_reinvestment_caution": "REINVESTMENT_UNCLEAR",
        "reinvestment_efficiency_summary": str(reinvestment_detail["reinvestment_efficiency_summary"]),
        "asset_intensity_class": "ASSET_INTENSITY_UNKNOWN",
        "asset_intensity_reason_codes": list(maintenance_capex_detail["asset_intensity_reason_codes"]),
        "maintenance_capex_credibility_class": "MAINTENANCE_CAPEX_CREDIBILITY_UNKNOWN",
        "maintenance_capex_credibility_reason_codes": list(
            maintenance_capex_detail["maintenance_capex_credibility_reason_codes"]
        ),
        "maintenance_capex_support_signals": [],
        "maintenance_capex_headwind_signals": list(maintenance_capex_detail["maintenance_capex_headwind_signals"]),
        "primary_maintenance_capex_caution": "OWNER_EARNINGS_UNCLEAR",
        "maintenance_capex_discipline_summary": str(
            maintenance_capex_detail["maintenance_capex_discipline_summary"]
        ),
        "revenue_dependence_risk_class": "REVENUE_DEPENDENCE_UNKNOWN",
        "revenue_dependence_risk_reason_codes": list(revenue_dependence_detail["revenue_dependence_risk_reason_codes"]),
        "revenue_dependence_support_signals": [],
        "revenue_dependence_headwind_signals": list(revenue_dependence_detail["revenue_dependence_headwind_signals"]),
        "primary_revenue_dependence_caution": "REVENUE_BASE_UNCLEAR",
        "revenue_fragility_summary": str(revenue_dependence_detail["revenue_fragility_summary"]),
        "returns_persistence_class": "RETURNS_PERSISTENCE_UNKNOWN",
        "returns_persistence_reason_codes": list(returns_persistence_detail["returns_persistence_reason_codes"]),
        "returns_support_signals": [],
        "returns_headwind_signals": list(returns_persistence_detail["returns_headwind_signals"]),
        "primary_returns_caution": "RETURNS_DURABILITY_UNCLEAR",
        "economic_durability_summary": str(returns_persistence_detail["economic_durability_summary"]),
        "accounting_quality_class": "ACCOUNTING_QUALITY_UNKNOWN",
        "accounting_quality_reason_codes": list(accounting_quality_detail["accounting_quality_reason_codes"]),
        "cash_earnings_support_signals": [],
        "cash_earnings_headwind_signals": list(accounting_quality_detail["cash_earnings_headwind_signals"]),
        "primary_accounting_caution": "ACCOUNTING_QUALITY_UNCLEAR",
        "cash_earnings_discipline_summary": str(accounting_quality_detail["cash_earnings_discipline_summary"]),
        "balance_sheet_stress_class": "BALANCE_SHEET_STRESS_UNKNOWN",
        "balance_sheet_stress_reason_codes": list(balance_sheet_stress_detail["balance_sheet_stress_reason_codes"]),
        "refinancing_risk_class": "REFINANCING_RISK_UNKNOWN",
        "refinancing_risk_reason_codes": list(balance_sheet_stress_detail["refinancing_risk_reason_codes"]),
        "balance_sheet_support_signals": [],
        "balance_sheet_headwind_signals": list(balance_sheet_stress_detail["balance_sheet_headwind_signals"]),
        "primary_balance_sheet_caution": "BALANCE_SHEET_UNCLEAR",
        "balance_sheet_discipline_summary": str(balance_sheet_stress_detail["balance_sheet_discipline_summary"]),
        "reasons": ["SKIPPED_BUDGET", reason],
        "metric_values": {
            "current_price": UNKNOWN,
            "market_cap": UNKNOWN,
            "ev": UNKNOWN,
            "cfo_value": UNKNOWN,
            "capex_value": UNKNOWN,
            "fcf_value": UNKNOWN,
            "fcf_yield": UNKNOWN,
            "fcf_yield_3y": UNKNOWN,
            "fcf_yield_ev": UNKNOWN,
            "fcf_yield_ev_3y": UNKNOWN,
            "owner_earnings_yield_3y": UNKNOWN,
            "owner_earnings_yield_latest": UNKNOWN,
            "owner_earnings_yield_ev_3y": UNKNOWN,
            "owner_earnings_yield_ev_latest": UNKNOWN,
            "net_debt_proxy": UNKNOWN,
            "total_debt": UNKNOWN,
            "cash_equivalents": UNKNOWN,
            "net_debt_to_cfo": UNKNOWN,
            "net_debt_to_fcf": UNKNOWN,
            "dilution_rate": UNKNOWN,
            "fcf_stability_score": UNKNOWN,
            "intrinsic_per_share_proxy": UNKNOWN,
            "valuation_gap": UNKNOWN,
            "epv_per_share": UNKNOWN,
            "netnet_per_share": UNKNOWN,
            "mos_epv": UNKNOWN,
            "mos_netnet": UNKNOWN,
            "owner_earnings_positive_years_5y": UNKNOWN,
            "owner_earnings_volatility_5y": UNKNOWN,
            "owner_earnings_stability_score": UNKNOWN,
            "capex_burden_vs_cfo": UNKNOWN,
            "capital_allocation_score": UNKNOWN,
            "cfo_margin_proxy": UNKNOWN,
            "fcf_conversion_proxy": UNKNOWN,
            "cash_conversion_score": UNKNOWN,
            "oe_quality_total": UNKNOWN,
            "gross_margin_avg_5y": UNKNOWN,
            "gross_margin_volatility_5y": UNKNOWN,
            "gross_margin_floor_5y": UNKNOWN,
            "gross_margin_durability_score": UNKNOWN,
            "rnd_to_revenue_avg_5y": UNKNOWN,
            "revenue_per_rnd_proxy": UNKNOWN,
            "gross_profit_per_rnd_proxy": UNKNOWN,
            "owner_earnings_per_rnd_proxy": UNKNOWN,
            "rnd_productivity_score": UNKNOWN,
            "net_debt_to_cfo_proxy": UNKNOWN,
            "cash_pct_revenue": UNKNOWN,
            "balance_sheet_optionality_score": UNKNOWN,
            "margin_volatility_5y": UNKNOWN,
            "fcf_cfo_conversion_median_3y": UNKNOWN,
            "fcf_cfo_conversion_volatility_3y": UNKNOWN,
            "cycle_resilience_score": UNKNOWN,
            "sga_to_revenue_avg_5y": UNKNOWN,
            "sga_growth_vs_revenue_growth_proxy": UNKNOWN,
            "operating_leverage_proxy": UNKNOWN,
            "sga_leverage_score": UNKNOWN,
            "revenue_per_share_cagr_proxy": UNKNOWN,
            "fcf_per_share_cagr_proxy": UNKNOWN,
            "owner_earnings_per_share_cagr_proxy": UNKNOWN,
            "owner_value_capture_score": UNKNOWN,
            "intangible_economics_total": UNKNOWN,
            "reinvestment_efficiency_class": "REINVESTMENT_EFFICIENCY_UNKNOWN",
            "asset_intensity_class": "ASSET_INTENSITY_UNKNOWN",
            "maintenance_capex_credibility_class": "MAINTENANCE_CAPEX_CREDIBILITY_UNKNOWN",
            "revenue_dependence_risk_class": "REVENUE_DEPENDENCE_UNKNOWN",
            "returns_persistence_class": "RETURNS_PERSISTENCE_UNKNOWN",
            "accounting_quality_class": "ACCOUNTING_QUALITY_UNKNOWN",
            "balance_sheet_stress_class": "BALANCE_SHEET_STRESS_UNKNOWN",
            "refinancing_risk_class": "REFINANCING_RISK_UNKNOWN",
        },
        "epv_per_share": UNKNOWN,
        "mos_epv": UNKNOWN,
        "netnet_per_share": UNKNOWN,
        "mos_netnet": UNKNOWN,
        "gd_value_status": "UNKNOWN",
        "gd_primary_reason_code": reason,
        "graham_dodd_detail": {
            "ticker": ticker_norm,
            "as_of_date": None,
            "epv_status": "UNKNOWN",
            "epv_reason_code": reason,
            "epv_value": UNKNOWN,
            "epv_per_share": UNKNOWN,
            "netnet_status": "UNKNOWN",
            "netnet_reason_code": reason,
            "netnet_value": UNKNOWN,
            "netnet_per_share": UNKNOWN,
            "mos_epv": UNKNOWN,
            "mos_netnet": UNKNOWN,
            "gd_value_status": "UNKNOWN",
            "gd_primary_reason_code": reason,
            "inputs_used": {},
            "thresholds_used": {},
            "derived_from": derived,
            "generated_at": utc_now_iso(),
        },
        "inputs_used": {},
        "owner_earnings_quality_detail": owner_quality_detail,
        "maintenance_capex_discipline_detail": maintenance_capex_detail,
        "accounting_quality_detail": accounting_quality_detail,
        "balance_sheet_stress_detail": balance_sheet_stress_detail,
        "intangible_economics_detail": intangible_detail,
        "reinvestment_efficiency_detail": reinvestment_detail,
        "revenue_dependence_detail": revenue_dependence_detail,
        "returns_persistence_detail": returns_persistence_detail,
        "derived_from": derived,
    }
    coverage_row = {
        "ticker": ticker_norm,
        "scout_status": WATCH,
        "primary_blocker": BLOCKER_OTHER_UNKNOWN,
        "primary_blocker_category": BLOCKER_OTHER_UNKNOWN,
        "blocker_categories": [BLOCKER_OTHER_UNKNOWN],
        "near_miss_fields": [],
        "recommendation": "resume universe-scout with higher budget limits",
        "yield_metric_used": UNKNOWN,
        "yield_status": UNKNOWN,
        "yield_reason_code": reason,
        "yield_blocker_subreason": reason,
        "yield_denominator_used": UNKNOWN,
        "yield_calibration_note": "SKIPPED_BUDGET",
        "yield_delta_to_pass": UNKNOWN,
        "epv_per_share": UNKNOWN,
        "mos_epv": UNKNOWN,
        "netnet_per_share": UNKNOWN,
        "mos_netnet": UNKNOWN,
        "gd_value_status": "UNKNOWN",
        "gd_primary_reason_code": reason,
        "price_status": UNKNOWN,
        "price_reason_code": reason,
        "facts_status": UNKNOWN,
        "fetch_reason_code": reason,
        "shares_status": UNKNOWN,
        "shares_reason_code": reason,
        "cfo_status": UNKNOWN,
        "cfo_reason_code": reason,
        "capex_status": UNKNOWN,
        "capex_reason_code": reason,
        "fcf_status": UNKNOWN,
        "fcf_reason_code": reason,
        "net_debt_status": UNKNOWN,
        "net_debt_reason_code": reason,
        "market_cap_status": UNKNOWN,
        "market_cap_reason_code": reason,
        "ev_status": UNKNOWN,
        "ev_reason_code": "MISSING_MARKET_CAP",
        "require_ev_yield": False,
        "dilution_status": UNKNOWN,
        "dilution_reason_code": reason,
        "fcf_stability_status": UNKNOWN,
        "fcf_stability_reason_code": reason,
        "owner_earnings_stability_score": UNKNOWN,
        "capital_allocation_score": UNKNOWN,
        "cash_conversion_score": UNKNOWN,
        "oe_quality_total": UNKNOWN,
        "oe_quality_reason_codes": list(owner_quality_detail["oe_quality_reason_codes"]),
        "gross_margin_durability_score": UNKNOWN,
        "balance_sheet_optionality_score": UNKNOWN,
        "cycle_resilience_score": UNKNOWN,
        "rnd_productivity_score": UNKNOWN,
        "sga_leverage_score": UNKNOWN,
        "owner_value_capture_score": UNKNOWN,
        "intangible_economics_total": UNKNOWN,
        "rnd_productivity_reason_codes": list(intangible_detail["rnd_productivity_reason_codes"]),
        "sga_leverage_reason_codes": list(intangible_detail["sga_leverage_reason_codes"]),
        "owner_value_capture_reason_codes": list(intangible_detail["owner_value_capture_reason_codes"]),
        "intangible_economics_reason_codes": list(intangible_detail["intangible_economics_reason_codes"]),
        "reinvestment_efficiency_class": "REINVESTMENT_EFFICIENCY_UNKNOWN",
        "reinvestment_efficiency_reason_codes": list(reinvestment_detail["reinvestment_efficiency_reason_codes"]),
        "reinvestment_support_signals": [],
        "reinvestment_headwind_signals": list(reinvestment_detail["reinvestment_headwind_signals"]),
        "primary_reinvestment_caution": "REINVESTMENT_UNCLEAR",
        "reinvestment_efficiency_summary": str(reinvestment_detail["reinvestment_efficiency_summary"]),
        "asset_intensity_class": "ASSET_INTENSITY_UNKNOWN",
        "asset_intensity_reason_codes": list(maintenance_capex_detail["asset_intensity_reason_codes"]),
        "maintenance_capex_credibility_class": "MAINTENANCE_CAPEX_CREDIBILITY_UNKNOWN",
        "maintenance_capex_credibility_reason_codes": list(
            maintenance_capex_detail["maintenance_capex_credibility_reason_codes"]
        ),
        "maintenance_capex_support_signals": [],
        "maintenance_capex_headwind_signals": list(maintenance_capex_detail["maintenance_capex_headwind_signals"]),
        "primary_maintenance_capex_caution": "OWNER_EARNINGS_UNCLEAR",
        "maintenance_capex_discipline_summary": str(
            maintenance_capex_detail["maintenance_capex_discipline_summary"]
        ),
        "revenue_dependence_risk_class": "REVENUE_DEPENDENCE_UNKNOWN",
        "revenue_dependence_risk_reason_codes": list(revenue_dependence_detail["revenue_dependence_risk_reason_codes"]),
        "revenue_dependence_support_signals": [],
        "revenue_dependence_headwind_signals": list(revenue_dependence_detail["revenue_dependence_headwind_signals"]),
        "primary_revenue_dependence_caution": "REVENUE_BASE_UNCLEAR",
        "revenue_fragility_summary": str(revenue_dependence_detail["revenue_fragility_summary"]),
        "returns_persistence_class": "RETURNS_PERSISTENCE_UNKNOWN",
        "returns_persistence_reason_codes": list(returns_persistence_detail["returns_persistence_reason_codes"]),
        "returns_support_signals": [],
        "returns_headwind_signals": list(returns_persistence_detail["returns_headwind_signals"]),
        "primary_returns_caution": "RETURNS_DURABILITY_UNCLEAR",
        "economic_durability_summary": str(returns_persistence_detail["economic_durability_summary"]),
        "accounting_quality_class": "ACCOUNTING_QUALITY_UNKNOWN",
        "accounting_quality_reason_codes": list(accounting_quality_detail["accounting_quality_reason_codes"]),
        "cash_earnings_support_signals": [],
        "cash_earnings_headwind_signals": list(accounting_quality_detail["cash_earnings_headwind_signals"]),
        "primary_accounting_caution": "ACCOUNTING_QUALITY_UNCLEAR",
        "cash_earnings_discipline_summary": str(accounting_quality_detail["cash_earnings_discipline_summary"]),
        "balance_sheet_stress_class": "BALANCE_SHEET_STRESS_UNKNOWN",
        "balance_sheet_stress_reason_codes": list(balance_sheet_stress_detail["balance_sheet_stress_reason_codes"]),
        "refinancing_risk_class": "REFINANCING_RISK_UNKNOWN",
        "refinancing_risk_reason_codes": list(balance_sheet_stress_detail["refinancing_risk_reason_codes"]),
        "balance_sheet_support_signals": [],
        "balance_sheet_headwind_signals": list(balance_sheet_stress_detail["balance_sheet_headwind_signals"]),
        "primary_balance_sheet_caution": "BALANCE_SHEET_UNCLEAR",
        "balance_sheet_discipline_summary": str(balance_sheet_stress_detail["balance_sheet_discipline_summary"]),
        "valuation_gap_status": UNKNOWN,
        "valuation_gap_reason_code": reason,
        "unknown_reasons": ["SKIPPED_BUDGET", reason],
        "fcf_history_values": [],
        "graham_dodd_detail": scoreboard_row["graham_dodd_detail"],
        "derived_from": derived,
    }
    yield_row = {
        "ticker": ticker_norm,
        "scout_status": WATCH,
        "yield_status": UNKNOWN,
        "yield_reason_code": reason,
        "yield_blocker_subreason": reason,
        "yield_metric_used": UNKNOWN,
        "yield_metric_type": UNKNOWN,
        "yield_denominator_used": UNKNOWN,
        "yield_calibration_note": "SKIPPED_BUDGET",
        "yield_gate_value_used": UNKNOWN,
        "yield_delta_to_pass": UNKNOWN,
        "owner_earnings_yield_3y": UNKNOWN,
        "fcf_yield_3y": UNKNOWN,
        "owner_earnings_yield_ev_3y": UNKNOWN,
        "fcf_yield_ev_3y": UNKNOWN,
        "ev": UNKNOWN,
        "ev_value": UNKNOWN,
        "ev_status": UNKNOWN,
        "ev_reason_code": "MISSING_MARKET_CAP",
        "ev_used": False,
        "net_debt_proxy_used": UNKNOWN,
        "net_debt_proxy_reason_code": reason,
        "denominator_used": UNKNOWN,
        "primary_blocker_category": BLOCKER_OTHER_UNKNOWN,
        "epv_per_share": UNKNOWN,
        "mos_epv": UNKNOWN,
        "netnet_per_share": UNKNOWN,
        "mos_netnet": UNKNOWN,
        "gd_value_status": "UNKNOWN",
        "gd_primary_reason_code": reason,
        "owner_earnings_stability_score": UNKNOWN,
        "capital_allocation_score": UNKNOWN,
        "cash_conversion_score": UNKNOWN,
        "oe_quality_total": UNKNOWN,
        "oe_quality_reason_codes": list(owner_quality_detail["oe_quality_reason_codes"]),
        "gross_margin_durability_score": UNKNOWN,
        "balance_sheet_optionality_score": UNKNOWN,
        "cycle_resilience_score": UNKNOWN,
        "rnd_productivity_score": UNKNOWN,
        "sga_leverage_score": UNKNOWN,
        "owner_value_capture_score": UNKNOWN,
        "intangible_economics_total": UNKNOWN,
        "rnd_productivity_reason_codes": list(intangible_detail["rnd_productivity_reason_codes"]),
        "sga_leverage_reason_codes": list(intangible_detail["sga_leverage_reason_codes"]),
        "owner_value_capture_reason_codes": list(intangible_detail["owner_value_capture_reason_codes"]),
        "intangible_economics_reason_codes": list(intangible_detail["intangible_economics_reason_codes"]),
        "reinvestment_efficiency_class": "REINVESTMENT_EFFICIENCY_UNKNOWN",
        "reinvestment_efficiency_reason_codes": list(reinvestment_detail["reinvestment_efficiency_reason_codes"]),
        "reinvestment_support_signals": [],
        "reinvestment_headwind_signals": list(reinvestment_detail["reinvestment_headwind_signals"]),
        "primary_reinvestment_caution": "REINVESTMENT_UNCLEAR",
        "reinvestment_efficiency_summary": str(reinvestment_detail["reinvestment_efficiency_summary"]),
        "asset_intensity_class": "ASSET_INTENSITY_UNKNOWN",
        "asset_intensity_reason_codes": list(maintenance_capex_detail["asset_intensity_reason_codes"]),
        "maintenance_capex_credibility_class": "MAINTENANCE_CAPEX_CREDIBILITY_UNKNOWN",
        "maintenance_capex_credibility_reason_codes": list(
            maintenance_capex_detail["maintenance_capex_credibility_reason_codes"]
        ),
        "maintenance_capex_support_signals": [],
        "maintenance_capex_headwind_signals": list(maintenance_capex_detail["maintenance_capex_headwind_signals"]),
        "primary_maintenance_capex_caution": "OWNER_EARNINGS_UNCLEAR",
        "maintenance_capex_discipline_summary": str(
            maintenance_capex_detail["maintenance_capex_discipline_summary"]
        ),
        "revenue_dependence_risk_class": "REVENUE_DEPENDENCE_UNKNOWN",
        "revenue_dependence_risk_reason_codes": list(revenue_dependence_detail["revenue_dependence_risk_reason_codes"]),
        "revenue_dependence_support_signals": [],
        "revenue_dependence_headwind_signals": list(revenue_dependence_detail["revenue_dependence_headwind_signals"]),
        "primary_revenue_dependence_caution": "REVENUE_BASE_UNCLEAR",
        "revenue_fragility_summary": str(revenue_dependence_detail["revenue_fragility_summary"]),
        "returns_persistence_class": "RETURNS_PERSISTENCE_UNKNOWN",
        "returns_persistence_reason_codes": list(returns_persistence_detail["returns_persistence_reason_codes"]),
        "returns_support_signals": [],
        "returns_headwind_signals": list(returns_persistence_detail["returns_headwind_signals"]),
        "primary_returns_caution": "RETURNS_DURABILITY_UNCLEAR",
        "economic_durability_summary": str(returns_persistence_detail["economic_durability_summary"]),
        "accounting_quality_class": "ACCOUNTING_QUALITY_UNKNOWN",
        "accounting_quality_reason_codes": list(accounting_quality_detail["accounting_quality_reason_codes"]),
        "cash_earnings_support_signals": [],
        "cash_earnings_headwind_signals": list(accounting_quality_detail["cash_earnings_headwind_signals"]),
        "primary_accounting_caution": "ACCOUNTING_QUALITY_UNCLEAR",
        "cash_earnings_discipline_summary": str(accounting_quality_detail["cash_earnings_discipline_summary"]),
        "balance_sheet_stress_class": "BALANCE_SHEET_STRESS_UNKNOWN",
        "balance_sheet_stress_reason_codes": list(balance_sheet_stress_detail["balance_sheet_stress_reason_codes"]),
        "refinancing_risk_class": "REFINANCING_RISK_UNKNOWN",
        "refinancing_risk_reason_codes": list(balance_sheet_stress_detail["refinancing_risk_reason_codes"]),
        "balance_sheet_support_signals": [],
        "balance_sheet_headwind_signals": list(balance_sheet_stress_detail["balance_sheet_headwind_signals"]),
        "primary_balance_sheet_caution": "BALANCE_SHEET_UNCLEAR",
        "balance_sheet_discipline_summary": str(balance_sheet_stress_detail["balance_sheet_discipline_summary"]),
        "owner_earnings_summary": {},
        "maintenance_capex_ratio": float(DEFAULT_MAINT_CAPEX_RATIO),
        "proxy_flags": {"maintenance_capex_proxy": True, "owner_earnings_proxy": True},
        "inputs_used": {},
        "derived_from": derived,
    }
    net_debt_row = {
        "ticker": ticker_norm,
        "as_of_date": None,
        "status": UNKNOWN,
        "reason_code": reason,
        "total_debt": {"value": UNKNOWN, "tag": None, "date": None, "derived_from": []},
        "cash_equivalents": {"value": UNKNOWN, "tag": None, "date": None, "derived_from": []},
        "net_debt_proxy": UNKNOWN,
        "derived_from": derived,
    }
    facts_blocker_fields = enrich_facts_blocker_fields(
        {
            **coverage_row,
            "primary_blocker": scoreboard_row["primary_blocker"],
            "primary_blocker_category": scoreboard_row["primary_blocker_category"],
            "blocker_categories": list(scoreboard_row.get("blocker_categories") or []),
        }
    )
    scoreboard_row.update(
        {
            "facts_blocker_class": facts_blocker_fields["facts_blocker_class"],
            "facts_blocker_retryable": facts_blocker_fields["facts_blocker_retryable"],
            "facts_blocker_terminal": facts_blocker_fields["facts_blocker_terminal"],
            "facts_blocker_partial_usable": facts_blocker_fields["facts_blocker_partial_usable"],
            "facts_missing_key_inputs": list(facts_blocker_fields["facts_missing_key_inputs"]),
            "facts_retry_recommended": facts_blocker_fields["facts_retry_recommended"],
            "facts_blocker_reason_codes": list(facts_blocker_fields["facts_blocker_reason_codes"]),
            "facts_recommended_action": facts_blocker_fields["facts_recommended_action"],
            "fail_due_to_missing_evidence": facts_blocker_fields["fail_due_to_missing_evidence"],
            "fail_due_to_economic_weakness": facts_blocker_fields["fail_due_to_economic_weakness"],
            "primary_fail_domain": facts_blocker_fields["primary_fail_domain"],
        }
    )
    coverage_row.update(facts_blocker_fields)
    yield_row.update(
        {
            "facts_blocker_class": facts_blocker_fields["facts_blocker_class"],
            "facts_blocker_retryable": facts_blocker_fields["facts_blocker_retryable"],
            "facts_blocker_terminal": facts_blocker_fields["facts_blocker_terminal"],
            "facts_blocker_partial_usable": facts_blocker_fields["facts_blocker_partial_usable"],
            "facts_missing_key_inputs": list(facts_blocker_fields["facts_missing_key_inputs"]),
            "facts_retry_recommended": facts_blocker_fields["facts_retry_recommended"],
            "facts_blocker_reason_codes": list(facts_blocker_fields["facts_blocker_reason_codes"]),
            "facts_recommended_action": facts_blocker_fields["facts_recommended_action"],
            "fail_due_to_missing_evidence": facts_blocker_fields["fail_due_to_missing_evidence"],
            "fail_due_to_economic_weakness": facts_blocker_fields["fail_due_to_economic_weakness"],
            "primary_fail_domain": facts_blocker_fields["primary_fail_domain"],
        }
    )
    return scoreboard_row, coverage_row, yield_row, net_debt_row


def _apply_facts_blocker_policy(
    score_row: dict[str, Any],
    coverage_row: dict[str, Any],
    yield_row: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    facts_blocker_fields = enrich_facts_blocker_fields(
        {
            **coverage_row,
            "scout_status": score_row.get("scout_status"),
            "primary_blocker": score_row.get("primary_blocker") or coverage_row.get("primary_blocker"),
            "primary_blocker_category": score_row.get("primary_blocker_category") or coverage_row.get("primary_blocker_category"),
            "blocker_categories": list(score_row.get("blocker_categories") or coverage_row.get("blocker_categories") or []),
        }
    )
    score_row.update(
        {
            "facts_blocker_class": facts_blocker_fields["facts_blocker_class"],
            "facts_blocker_retryable": facts_blocker_fields["facts_blocker_retryable"],
            "facts_blocker_terminal": facts_blocker_fields["facts_blocker_terminal"],
            "facts_blocker_partial_usable": facts_blocker_fields["facts_blocker_partial_usable"],
            "facts_missing_key_inputs": list(facts_blocker_fields["facts_missing_key_inputs"]),
            "facts_retry_recommended": facts_blocker_fields["facts_retry_recommended"],
            "facts_blocker_reason_codes": list(facts_blocker_fields["facts_blocker_reason_codes"]),
            "facts_recommended_action": facts_blocker_fields["facts_recommended_action"],
            "fail_due_to_missing_evidence": facts_blocker_fields["fail_due_to_missing_evidence"],
            "fail_due_to_economic_weakness": facts_blocker_fields["fail_due_to_economic_weakness"],
            "primary_fail_domain": facts_blocker_fields["primary_fail_domain"],
        }
    )
    coverage_row.update(facts_blocker_fields)
    yield_row.update(
        {
            "facts_blocker_class": facts_blocker_fields["facts_blocker_class"],
            "facts_blocker_retryable": facts_blocker_fields["facts_blocker_retryable"],
            "facts_blocker_terminal": facts_blocker_fields["facts_blocker_terminal"],
            "facts_blocker_partial_usable": facts_blocker_fields["facts_blocker_partial_usable"],
            "facts_missing_key_inputs": list(facts_blocker_fields["facts_missing_key_inputs"]),
            "facts_retry_recommended": facts_blocker_fields["facts_retry_recommended"],
            "facts_blocker_reason_codes": list(facts_blocker_fields["facts_blocker_reason_codes"]),
            "facts_recommended_action": facts_blocker_fields["facts_recommended_action"],
            "fail_due_to_missing_evidence": facts_blocker_fields["fail_due_to_missing_evidence"],
            "fail_due_to_economic_weakness": facts_blocker_fields["fail_due_to_economic_weakness"],
            "primary_fail_domain": facts_blocker_fields["primary_fail_domain"],
        }
    )
    return score_row, coverage_row, yield_row


def _build_scout_payloads(
    *,
    run_id: str,
    as_of_date: str,
    top_n: int,
    with_prices: bool,
    thresholds: dict[str, Any],
    universe_meta: dict[str, Any],
    scoreboard_rows: list[dict[str, Any]],
    coverage_rows: list[dict[str, Any]],
    yield_rows: list[dict[str, Any]],
    net_debt_payload: dict[str, Any],
    batch_progress: dict[str, Any],
    budget_progress: dict[str, Any],
    progress_state: dict[str, Any],
    run_status: str,
    stop_reason_code: str,
    stop_summary: str,
) -> dict[str, dict[str, Any]]:
    thresholds_effective = _effective_thresholds(thresholds)
    require_ev_yield = bool(thresholds_effective.get("scout_require_ev_yield", False))
    cfg = get_config()
    sector_map = _sector_map_from_taxonomy()
    shortlist_top_n = max(1, int(getattr(cfg, "universe_shortlist_top_n", top_n)))
    shortlist_max_per_sector = max(1, int(getattr(cfg, "universe_shortlist_max_per_sector", 10)))

    scoreboard_rows = sorted(scoreboard_rows, key=_score_sort_key)
    coverage_rows = sorted(coverage_rows, key=lambda row: str(row.get("ticker") or ""))
    yield_rows = sorted(yield_rows, key=lambda row: str(row.get("ticker") or ""))
    ranked_rows_full = _ranked_rows(scoreboard_rows)
    ranked_rows_serialized = [_serialize_rank_row(row, sector_map) for row in ranked_rows_full]

    status_counts = {PASS: 0, WATCH: 0, FAIL: 0}
    blocker_counts: dict[str, int] = {}
    for row in scoreboard_rows:
        status = str(row.get("scout_status") or WATCH).upper()
        if status not in status_counts:
            status = WATCH
        status_counts[status] += 1
        blocker = str(row.get("primary_blocker_category") or "UNKNOWN")
        blocker_counts[blocker] = blocker_counts.get(blocker, 0) + 1

    unknown_reason_counts: dict[str, int] = {}
    for row in coverage_rows:
        for reason in (row.get("unknown_reasons") or []):
            token = str(reason).upper().strip()
            if not token:
                continue
            unknown_reason_counts[token] = unknown_reason_counts.get(token, 0) + 1
    unknown_reason_counts = dict(sorted(unknown_reason_counts.items(), key=lambda kv: (-kv[1], kv[0])))
    facts_blocker_summary = summarize_facts_blockers(coverage_rows)

    yield_status_counts = {
        "known": len([row for row in yield_rows if str(row.get("yield_status") or "").upper() in {"OK", "LOW"}]),
        "unknown": len([row for row in yield_rows if str(row.get("yield_status") or "").upper() == "UNKNOWN"]),
        "low": len([row for row in yield_rows if str(row.get("yield_status") or "").upper() == "LOW"]),
    }
    ev_known_count = len([row for row in yield_rows if str(row.get("ev_status") or "").upper() == "OK"])
    ev_unknown_count = len(yield_rows) - ev_known_count
    yield_reason_counts: dict[str, int] = {}
    for row in yield_rows:
        code = str(row.get("yield_reason_code") or "UNKNOWN").upper()
        yield_reason_counts[code] = yield_reason_counts.get(code, 0) + 1
    yield_reason_counts = dict(sorted(yield_reason_counts.items(), key=lambda kv: (-kv[1], kv[0])))
    yield_blocker_breakdown: dict[str, int] = {}
    for row in yield_rows:
        code = str(row.get("yield_reason_code") or "UNKNOWN").upper()
        if code not in {
            BLOCKER_LOW_YIELD_OWNER_EARNINGS_EV,
            BLOCKER_LOW_YIELD_FCF_EV,
            BLOCKER_LOW_YIELD_OWNER_EARNINGS,
            BLOCKER_LOW_YIELD_FCF,
            BLOCKER_MISSING_EV,
            BLOCKER_NEGATIVE_CFO,
            BLOCKER_NEGATIVE_FCF,
        }:
            continue
        denominator = str(row.get("yield_denominator_used") or "UNKNOWN").upper()
        key = f"{denominator}:{code}:{str(row.get('yield_blocker_subreason') or 'UNKNOWN').upper()}"
        yield_blocker_breakdown[key] = yield_blocker_breakdown.get(key, 0) + 1
    yield_blocker_breakdown = dict(sorted(yield_blocker_breakdown.items(), key=lambda kv: (-kv[1], kv[0])))
    gd_known_count = len([row for row in scoreboard_rows if str(row.get("gd_value_status") or "").upper() == GD_STATUS_OK])
    gd_unknown_count = len(scoreboard_rows) - gd_known_count
    gd_reason_counts: dict[str, int] = {}
    for row in scoreboard_rows:
        reason = str(row.get("gd_primary_reason_code") or UNKNOWN).upper()
        gd_reason_counts[reason] = gd_reason_counts.get(reason, 0) + 1
    gd_reason_counts = dict(sorted(gd_reason_counts.items(), key=lambda kv: (-kv[1], kv[0])))

    calibration_payload = _build_scout_calibration_payload(
        run_id=run_id,
        as_of_date=as_of_date,
        scoreboard_rows=scoreboard_rows,
        coverage_rows=coverage_rows,
        thresholds=thresholds,
    )

    shortlist_bundle = _top_shortlist_candidates(
        scoreboard_rows,
        top_n=min(max(1, int(top_n)), shortlist_top_n),
        max_per_sector=shortlist_max_per_sector,
        sector_map=sector_map,
    )
    shortlist_rows = shortlist_bundle["top_candidates"]
    shortlist_rows_by_sector = shortlist_bundle["top_candidates_by_sector"]
    shortlist_payload = {
        "run_id": run_id,
        "as_of_date": as_of_date,
        "top_n": max(1, int(top_n)),
        "top_n_overall": min(max(1, int(top_n)), shortlist_top_n),
        "top_n_per_sector": shortlist_max_per_sector,
        "ticker_count": len(scoreboard_rows),
        "counts": status_counts,
        "thresholds_effective": thresholds_effective,
        "top_candidates": shortlist_rows,
        "top_candidates_overall": shortlist_rows,
        "top_candidates_by_sector": shortlist_rows_by_sector,
        "generated_at": utc_now_iso(),
        "derived_from": [
            "universe_rankings.ranked_rows[*]",
            "universe_scoreboard.rows[*].scout_status",
        ],
    }
    scoreboard_payload = {
        "run_id": run_id,
        "as_of_date": as_of_date,
        "ticker_count": len(scoreboard_rows),
        "counts": status_counts,
        "thresholds_effective": thresholds_effective,
        "rows": scoreboard_rows,
        "generated_at": utc_now_iso(),
    }
    coverage_payload = {
        "run_id": run_id,
        "as_of_date": as_of_date,
        "ticker_count": len(coverage_rows),
        "unknown_reason_counts": unknown_reason_counts,
        "blocker_counts": dict(sorted(blocker_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "facts_blocker_histogram": facts_blocker_summary.get("facts_blocker_histogram")
        if isinstance(facts_blocker_summary.get("facts_blocker_histogram"), dict)
        else {},
        "retryable_facts_blocker_count": int(facts_blocker_summary.get("retryable_facts_blocker_count") or 0),
        "terminal_facts_blocker_count": int(facts_blocker_summary.get("terminal_facts_blocker_count") or 0),
        "partial_usable_facts_count": int(facts_blocker_summary.get("partial_usable_facts_count") or 0),
        "top_retryable_facts_blockers": facts_blocker_summary.get("top_retryable_facts_blockers")
        if isinstance(facts_blocker_summary.get("top_retryable_facts_blockers"), list)
        else [],
        "top_terminal_facts_blockers": facts_blocker_summary.get("top_terminal_facts_blockers")
        if isinstance(facts_blocker_summary.get("top_terminal_facts_blockers"), list)
        else [],
        "top_partial_usable_facts": facts_blocker_summary.get("top_partial_usable_facts")
        if isinstance(facts_blocker_summary.get("top_partial_usable_facts"), list)
        else [],
        "recommended_next_action_counts": facts_blocker_summary.get("recommended_next_action_counts")
        if isinstance(facts_blocker_summary.get("recommended_next_action_counts"), dict)
        else {},
        "economic_fail_count_vs_evidence_fail_count": facts_blocker_summary.get("economic_fail_count_vs_evidence_fail_count")
        if isinstance(facts_blocker_summary.get("economic_fail_count_vs_evidence_fail_count"), dict)
        else {},
        "thresholds_effective": thresholds_effective,
        "rows": coverage_rows,
        "generated_at": utc_now_iso(),
    }
    yield_payload = {
        "run_id": run_id,
        "as_of_date": as_of_date,
        "ticker_count": len(yield_rows),
        "status_counts": yield_status_counts,
        "ev_known_count": int(ev_known_count),
        "ev_unknown_count": int(ev_unknown_count),
        "require_ev_yield": require_ev_yield,
        "yield_reason_counts": yield_reason_counts,
        "yield_blocker_breakdown": yield_blocker_breakdown,
        "thresholds_effective": thresholds_effective,
        "rows": yield_rows,
        "generated_at": utc_now_iso(),
    }

    summary_payload = {
        "run_id": run_id,
        "as_of_date": as_of_date,
        "ticker_count": len(scoreboard_rows),
        "top_n": max(1, int(top_n)),
        "with_prices": bool(with_prices),
        "counts": status_counts,
        "thresholds_effective": thresholds_effective,
        "coverage": {
            "unknown_reason_counts": unknown_reason_counts,
            "blocker_counts": dict(sorted(blocker_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        },
        "facts_blockers": {
            "facts_blocker_histogram": facts_blocker_summary.get("facts_blocker_histogram")
            if isinstance(facts_blocker_summary.get("facts_blocker_histogram"), dict)
            else {},
            "retryable_facts_blocker_count": int(facts_blocker_summary.get("retryable_facts_blocker_count") or 0),
            "terminal_facts_blocker_count": int(facts_blocker_summary.get("terminal_facts_blocker_count") or 0),
            "partial_usable_facts_count": int(facts_blocker_summary.get("partial_usable_facts_count") or 0),
            "top_retryable_facts_blockers": facts_blocker_summary.get("top_retryable_facts_blockers")
            if isinstance(facts_blocker_summary.get("top_retryable_facts_blockers"), list)
            else [],
            "top_terminal_facts_blockers": facts_blocker_summary.get("top_terminal_facts_blockers")
            if isinstance(facts_blocker_summary.get("top_terminal_facts_blockers"), list)
            else [],
            "top_partial_usable_facts": facts_blocker_summary.get("top_partial_usable_facts")
            if isinstance(facts_blocker_summary.get("top_partial_usable_facts"), list)
            else [],
            "recommended_next_action_counts": facts_blocker_summary.get("recommended_next_action_counts")
            if isinstance(facts_blocker_summary.get("recommended_next_action_counts"), dict)
            else {},
            "economic_fail_count_vs_evidence_fail_count": facts_blocker_summary.get("economic_fail_count_vs_evidence_fail_count")
            if isinstance(facts_blocker_summary.get("economic_fail_count_vs_evidence_fail_count"), dict)
            else {},
        },
        "yield_coverage": {
            "status_counts": yield_status_counts,
            "ev_known_count": int(ev_known_count),
            "ev_unknown_count": int(ev_unknown_count),
            "yield_reason_counts": yield_reason_counts,
            "yield_blocker_breakdown": yield_blocker_breakdown,
        },
        "graham_dodd": {
            "gd_known_count": int(gd_known_count),
            "gd_unknown_count": int(gd_unknown_count),
            "reason_counts": gd_reason_counts,
        },
        "net_debt_coverage": {
            "status_counts": net_debt_payload.get("status_counts") if isinstance(net_debt_payload.get("status_counts"), dict) else {},
            "reason_counts": net_debt_payload.get("reason_counts") if isinstance(net_debt_payload.get("reason_counts"), dict) else {},
            "net_debt_coverage_path": str(net_debt_payload.get("net_debt_coverage_path") or ""),
        },
        "top_blocker_categories": [
            {"primary_blocker_category": str(name), "count": int(count)}
            for name, count in list(dict(sorted(blocker_counts.items(), key=lambda kv: (-kv[1], kv[0]))).items())[:10]
        ],
        "calibration_required": calibration_payload.get("calibration_required"),
        "input_meta": universe_meta,
        "batch_progress": batch_progress,
        "budget_progress": budget_progress,
        "hydration_status": str(progress_state.get("hydration_status") or HYDRATION_STATUS_IDLE),
        "last_progress_phase": str(progress_state.get("last_progress_phase") or PHASE_NOT_STARTED),
        "primary_scout_blocker": str(progress_state.get("primary_scout_blocker") or ""),
        "stalled_reason_code": str(progress_state.get("stalled_reason_code") or ""),
        "hydration_progress": {
            "current_batch_index": progress_state.get("current_batch_index"),
            "current_batch_tickers": list(progress_state.get("current_batch_tickers") or []),
            "current_ticker": str(progress_state.get("current_ticker") or ""),
            "hydration_phase": str(progress_state.get("hydration_phase") or PHASE_NOT_STARTED),
            "hydration_started_at": progress_state.get("hydration_started_at"),
            "hydration_last_progress_at": progress_state.get("hydration_last_progress_at"),
            "last_completed_ticker": str(progress_state.get("last_completed_ticker") or ""),
            "facts_cache_hits": int(progress_state.get("facts_cache_hits") or 0),
            "facts_cache_misses": int(progress_state.get("facts_cache_misses") or 0),
            "companyfacts_fetch_attempts": int(progress_state.get("companyfacts_fetch_attempts") or 0),
            "companyfacts_failures": int(progress_state.get("companyfacts_failures") or 0),
            "companyfacts_timeouts": int(progress_state.get("companyfacts_timeouts") or 0),
            "price_stage_completed": bool(progress_state.get("price_stage_completed", False)),
            "facts_stage_completed": bool(progress_state.get("facts_stage_completed", False)),
            "scoring_stage_completed": bool(progress_state.get("scoring_stage_completed", False)),
            "finalization_started": bool(progress_state.get("finalization_started", False)),
            "tickers_completed": int(progress_state.get("tickers_completed") or 0),
        },
        "run_status": run_status,
        "stop_reason_code": stop_reason_code,
        "stop_summary": stop_summary,
        "generated_at": utc_now_iso(),
    }
    rankings_payload = _build_rankings_payload(
        run_id=run_id,
        as_of_date=as_of_date,
        ranked_rows=ranked_rows_serialized,
        top_n=max(1, int(top_n)),
        thresholds_effective=thresholds_effective,
        yield_rows=yield_rows,
        coverage_summary=coverage_payload,
    )
    depth_queue_payload = _build_depth_queue_payload(
        run_id=run_id,
        as_of_date=as_of_date,
        ranked_rows=rankings_payload.get("ranked_rows") if isinstance(rankings_payload.get("ranked_rows"), list) else [],
        cfg=cfg,
    )
    return {
        "summary": summary_payload,
        "scoreboard": scoreboard_payload,
        "shortlist": shortlist_payload,
        "coverage": coverage_payload,
        "calibration": calibration_payload,
        "yield": yield_payload,
        "rankings": rankings_payload,
        "depth_queue": depth_queue_payload,
    }


def run_universe_scout(
    *,
    run_id: str,
    as_of_date: str,
    top_n: int = 50,
    tickers: list[str] | None = None,
    sector_run_id: str | None = None,
    universe_csv: Path | None = None,
    universe_source: str | None = None,
    universe_limit: int | None = None,
    with_prices: bool = True,
    threshold_overrides: dict[str, Any] | None = None,
    batch_size: int = 200,
    max_batches: int | None = None,
    scout_sec_budget: int | None = None,
    scout_net_budget: int | None = None,
    scout_max_seconds: int | None = None,
    force_restart: bool = False,
) -> dict[str, Any]:
    cfg = get_config()
    paths = _run_paths(cfg=cfg, run_id=run_id)
    if bool(force_restart):
        if paths["run_dir"].exists():
            shutil.rmtree(paths["run_dir"])
        if paths["universe_dir"].exists():
            shutil.rmtree(paths["universe_dir"])
    paths["run_dir"].mkdir(parents=True, exist_ok=True)
    paths["universe_dir"].mkdir(parents=True, exist_ok=True)
    paths["batch_dir"].mkdir(parents=True, exist_ok=True)

    thresholds = normalize_scout_thresholds(threshold_overrides)
    require_ev_yield = bool(thresholds.get("scout_require_ev_yield", False))
    thresholds_effective = _effective_thresholds(thresholds)

    # Resume-gate read must distinguish a genuinely-absent state file (fresh run)
    # from a corrupt one (e.g. killed mid-write). A silent {} here would make the
    # gate see no RUNNING/PARTIAL status and restart the whole sweep from batch 0,
    # destroying prior progress. Read loudly so corruption raises instead.
    try:
        existing_state = read_json_safe(
            paths["state_path"], default={}, on_corrupt="raise"
        )
    except JsonCorruptError as exc:
        logger.error(
            "Scout state file is corrupt (not resumable): %s — %s. "
            "Refusing to silently restart from batch 0; pass force_restart=True to overwrite.",
            paths["state_path"],
            exc,
        )
        raise
    resume_existing = (
        not bool(force_restart)
        and isinstance(existing_state, dict)
        and str(existing_state.get("status") or "").upper() in {SCOUT_STATE_RUNNING, SCOUT_STATE_PARTIAL}
    )

    universe_tickers, universe_meta = _resolve_universe_tickers(
        tickers=tickers,
        sector_run_id=sector_run_id,
        universe_csv=universe_csv,
    )
    if universe_limit is not None and int(universe_limit) > 0:
        universe_tickers = universe_tickers[: int(universe_limit)]
    universe_meta = {
        **universe_meta,
        "count_used": len(universe_tickers),
    }
    if not universe_tickers:
        raise ValueError("Universe Scout received no valid tickers from inputs.")

    universe_input_payload = {
        "run_id": run_id,
        "as_of_date": as_of_date,
        "source": {
            "universe_source": str(universe_source or "custom"),
            "universe_file": str(universe_csv) if isinstance(universe_csv, Path) else None,
            "sector_run_id": str(sector_run_id) if sector_run_id else None,
            "explicit_tickers": len(tickers or []),
        },
        "canonical_tickers": list(universe_tickers),
        "normalization": universe_meta,
        "generated_at": utc_now_iso(),
    }
    _json_write(paths["universe_input_path"], universe_input_payload)

    scoreboard_map: dict[str, dict[str, Any]] = {}
    coverage_map: dict[str, dict[str, Any]] = {}
    yield_map: dict[str, dict[str, Any]] = {}
    net_debt_resolved_map: dict[str, dict[str, Any]] = {}
    done_batches: set[int] = set()
    if resume_existing:
        if int(existing_state.get("total_tickers") or 0) != len(universe_tickers):
            raise ValueError("Existing scout_state universe size differs from requested inputs; rerun with --force-restart.")
        scoreboard_map = _rows_by_ticker([enrich_facts_blocker_fields(row) for row in _safe_json_rows(paths["scoreboard_path"])])
        coverage_map = _rows_by_ticker([enrich_facts_blocker_fields(row) for row in _safe_json_rows(paths["coverage_path"])])
        yield_map = _rows_by_ticker(_safe_json_rows(paths["yield_path"]))
        for batch_payload_path in sorted(paths["batch_dir"].glob("batch_*.json")):
            batch_payload = _safe_json(batch_payload_path)
            if str(batch_payload.get("status") or "").upper() == "DONE":
                idx = int(batch_payload.get("batch_index") or -1)
                if idx >= 0:
                    done_batches.add(idx)
        for idx in existing_state.get("done_batches") or []:
            if _is_num(idx) and int(idx) >= 0:
                done_batches.add(int(idx))
        net_debt_existing = _safe_json(paths["net_debt_path"])
        for entry in [row for row in (net_debt_existing.get("entries") or []) if isinstance(row, dict)]:
            ticker_norm = str(entry.get("ticker") or "").strip().upper()
            if not ticker_norm:
                continue
            net_debt_resolved_map[ticker_norm] = {
                "ticker": ticker_norm,
                "as_of_date": str(entry.get("as_of_date") or as_of_date),
                "status": str(entry.get("status") or "UNKNOWN"),
                "reason_code": str(entry.get("reason_code") or "UNKNOWN"),
                "total_debt": {
                    "value": entry.get("total_debt_value", UNKNOWN),
                    "tag": (entry.get("tags_used") or {}).get("debt_tag") if isinstance(entry.get("tags_used"), dict) else None,
                    "date": (entry.get("fact_dates_used") or {}).get("debt_date") if isinstance(entry.get("fact_dates_used"), dict) else None,
                    "derived_from": [],
                },
                "cash_equivalents": {
                    "value": entry.get("cash_equivalents_value", UNKNOWN),
                    "tag": (entry.get("tags_used") or {}).get("cash_tag") if isinstance(entry.get("tags_used"), dict) else None,
                    "date": (entry.get("fact_dates_used") or {}).get("cash_date") if isinstance(entry.get("fact_dates_used"), dict) else None,
                    "derived_from": [],
                },
                "net_debt_proxy": entry.get("net_debt_proxy_value", UNKNOWN),
                "derived_from": [str(ref) for ref in (entry.get("derived_from") or []) if str(ref).strip()],
            }

    sec_budget_effective = _as_int_or_none(scout_sec_budget)
    net_budget_effective = _as_int_or_none(scout_net_budget)
    max_seconds_effective = _as_int_or_none(scout_max_seconds)
    started_at = str(existing_state.get("started_at") or utc_now_iso()) if resume_existing else utc_now_iso()
    start_monotonic = time.monotonic()
    if resume_existing:
        prev_remaining = existing_state.get("budget_remaining") if isinstance(existing_state.get("budget_remaining"), dict) else {}
        sec_budget_remaining = (
            sec_budget_effective
            if sec_budget_effective is not None
            else _as_int_or_none(prev_remaining.get("scout_sec_budget"))
        )
        net_budget_remaining = (
            net_budget_effective
            if net_budget_effective is not None
            else _as_int_or_none(prev_remaining.get("scout_net_budget"))
        )
        max_seconds_remaining = (
            max_seconds_effective
            if max_seconds_effective is not None
            else _as_int_or_none(prev_remaining.get("scout_max_seconds"))
        )
        if sec_budget_effective is None:
            sec_budget_effective = _as_int_or_none((existing_state.get("budgets_effective") or {}).get("scout_sec_budget"))
        if net_budget_effective is None:
            net_budget_effective = _as_int_or_none((existing_state.get("budgets_effective") or {}).get("scout_net_budget"))
        if max_seconds_effective is None:
            max_seconds_effective = _as_int_or_none((existing_state.get("budgets_effective") or {}).get("scout_max_seconds"))
    else:
        sec_budget_remaining = sec_budget_effective
        net_budget_remaining = net_budget_effective
        max_seconds_remaining = max_seconds_effective

    batches = _batch_slices(universe_tickers, batch_size=max(1, int(batch_size)))
    total_batches = len(batches)
    stop_reason_code = STOP_NONE
    stop_summary = ""
    run_status = SCOUT_STATE_RUNNING
    invocation_batches = 0
    last_completed_batch = max(done_batches) if done_batches else -1
    progress_state = {
        "current_batch_index": existing_state.get("current_batch_index") if resume_existing else None,
        "current_batch_tickers": list(existing_state.get("current_batch_tickers") or []) if resume_existing else [],
        "current_ticker": str(existing_state.get("current_ticker") or "") if resume_existing else "",
        "hydration_phase": str(existing_state.get("hydration_phase") or PHASE_NOT_STARTED) if resume_existing else PHASE_NOT_STARTED,
        "hydration_status": str(existing_state.get("hydration_status") or HYDRATION_STATUS_IDLE) if resume_existing else HYDRATION_STATUS_IDLE,
        "hydration_started_at": existing_state.get("hydration_started_at") if resume_existing else None,
        "hydration_last_progress_at": existing_state.get("hydration_last_progress_at") if resume_existing else None,
        "last_progress_phase": str(existing_state.get("last_progress_phase") or PHASE_NOT_STARTED) if resume_existing else PHASE_NOT_STARTED,
        "last_completed_ticker": str(existing_state.get("last_completed_ticker") or "") if resume_existing else "",
        "facts_cache_hits": int(existing_state.get("facts_cache_hits") or 0) if resume_existing else 0,
        "facts_cache_misses": int(existing_state.get("facts_cache_misses") or 0) if resume_existing else 0,
        "companyfacts_fetch_attempts": int(existing_state.get("companyfacts_fetch_attempts") or 0) if resume_existing else 0,
        "companyfacts_failures": int(existing_state.get("companyfacts_failures") or 0) if resume_existing else 0,
        "companyfacts_timeouts": int(existing_state.get("companyfacts_timeouts") or 0) if resume_existing else 0,
        "primary_scout_blocker": str(existing_state.get("primary_scout_blocker") or "") if resume_existing else "",
        "stalled_reason_code": str(existing_state.get("stalled_reason_code") or "") if resume_existing else "",
        "price_stage_completed": bool(existing_state.get("price_stage_completed", False)) if resume_existing else False,
        "facts_stage_completed": bool(existing_state.get("facts_stage_completed", False)) if resume_existing else False,
        "scoring_stage_completed": bool(existing_state.get("scoring_stage_completed", False)) if resume_existing else False,
        "finalization_started": bool(existing_state.get("finalization_started", False)) if resume_existing else False,
        "tickers_completed": len(scoreboard_map),
    }

    def _set_hydration_status(status_value: str | None) -> None:
        if not status_value:
            return
        current = str(progress_state.get("hydration_status") or HYDRATION_STATUS_IDLE)
        next_value = str(status_value)
        if current == HYDRATION_STATUS_STALE and next_value != HYDRATION_STATUS_STALE:
            return
        if current == HYDRATION_STATUS_DEGRADED and next_value in {HYDRATION_STATUS_RUNNING, HYDRATION_STATUS_OK}:
            return
        progress_state["hydration_status"] = next_value

    def _mark_progress(
        phase: str,
        *,
        batch_index: int | None = None,
        batch_tickers: list[str] | None = None,
        current_ticker: str | None = None,
        last_completed_ticker: str | None = None,
        hydration_status: str | None = None,
        primary_blocker: str | None = None,
        stalled_reason_code: str | None = None,
        price_stage_completed: bool | None = None,
        facts_stage_completed: bool | None = None,
        scoring_stage_completed: bool | None = None,
        finalization_started: bool | None = None,
    ) -> None:
        now = utc_now_iso()
        current_status = str(progress_state.get("hydration_status") or HYDRATION_STATUS_IDLE)
        if batch_index is not None:
            progress_state["current_batch_index"] = int(batch_index)
        if batch_tickers is not None:
            progress_state["current_batch_tickers"] = list(batch_tickers)
        if current_ticker is not None:
            progress_state["current_ticker"] = str(current_ticker or "")
        if last_completed_ticker is not None:
            progress_state["last_completed_ticker"] = str(last_completed_ticker or "")
        if phase and not (
            current_status == HYDRATION_STATUS_STALE
            and str(phase) in {PHASE_BATCH_FINALIZATION, PHASE_OUTPUT_FINALIZATION, PHASE_DONE}
        ):
            progress_state["hydration_phase"] = str(phase)
            progress_state["last_progress_phase"] = str(phase)
            progress_state["hydration_started_at"] = now
        progress_state["hydration_last_progress_at"] = now
        if primary_blocker is not None and str(primary_blocker).strip():
            progress_state["primary_scout_blocker"] = str(primary_blocker)
        if stalled_reason_code is not None:
            progress_state["stalled_reason_code"] = str(stalled_reason_code)
        if price_stage_completed is not None:
            progress_state["price_stage_completed"] = bool(price_stage_completed)
        if facts_stage_completed is not None:
            progress_state["facts_stage_completed"] = bool(facts_stage_completed)
        if scoring_stage_completed is not None:
            progress_state["scoring_stage_completed"] = bool(scoring_stage_completed)
        if finalization_started is not None:
            progress_state["finalization_started"] = bool(finalization_started)
        progress_state["tickers_completed"] = len(scoreboard_map)
        _set_hydration_status(hydration_status)

    def _record_facts_result(facts_row: dict[str, Any]) -> None:
        fetch_reason = str(facts_row.get("fetch_reason_code") or "").upper()
        if fetch_reason == "CACHE_HIT":
            progress_state["facts_cache_hits"] = int(progress_state.get("facts_cache_hits") or 0) + 1
        else:
            progress_state["facts_cache_misses"] = int(progress_state.get("facts_cache_misses") or 0) + 1
        if bool(facts_row.get("network_attempted")):
            attempts = int(facts_row.get("fetch_attempts") or 0)
            progress_state["companyfacts_fetch_attempts"] = int(progress_state.get("companyfacts_fetch_attempts") or 0) + max(1, attempts)
        if fetch_reason in {"FETCH_4XX", "FETCH_5XX", "EXCEPTION", "OFFLINE_NO_CACHE", "BUDGET_EXHAUSTED"}:
            progress_state["companyfacts_failures"] = int(progress_state.get("companyfacts_failures") or 0) + 1
            _set_hydration_status(HYDRATION_STATUS_DEGRADED)
        if fetch_reason == REASON_SCOUT_FACTS_TIMEOUT:
            progress_state["companyfacts_timeouts"] = int(progress_state.get("companyfacts_timeouts") or 0) + 1
            progress_state["companyfacts_failures"] = int(progress_state.get("companyfacts_failures") or 0) + 1
            _set_hydration_status(HYDRATION_STATUS_DEGRADED)

    def _budget_progress() -> dict[str, Any]:
        elapsed = int(max(0, time.monotonic() - start_monotonic))
        return {
            "effective": {
                "scout_sec_budget": sec_budget_effective,
                "scout_net_budget": net_budget_effective,
                "scout_max_seconds": max_seconds_effective,
            },
            "used": {
                "scout_sec_budget": (
                    max(0, int(sec_budget_effective) - int(sec_budget_remaining))
                    if sec_budget_effective is not None and sec_budget_remaining is not None
                    else None
                ),
                "scout_net_budget": (
                    max(0, int(net_budget_effective) - int(net_budget_remaining))
                    if net_budget_effective is not None and net_budget_remaining is not None
                    else None
                ),
                "scout_max_seconds": (
                    max(0, int(max_seconds_effective) - int(max_seconds_remaining))
                    if max_seconds_effective is not None and max_seconds_remaining is not None
                    else elapsed
                ),
            },
            "remaining": {
                "scout_sec_budget": sec_budget_remaining,
                "scout_net_budget": net_budget_remaining,
                "scout_max_seconds": max_seconds_remaining,
            },
        }

    def _batch_progress() -> dict[str, Any]:
        return {
            "batch_size": max(1, int(batch_size)),
            "total_batches": total_batches,
            "batches_done": len(done_batches),
            "last_completed_batch": max(done_batches) if done_batches else -1,
            "remaining_batches": max(0, total_batches - len(done_batches)),
        }

    def _write_state(status: str, *, stop_code: str, stop_text: str) -> None:
        state_payload = {
            "run_id": run_id,
            "as_of_date": as_of_date,
            "started_at": started_at,
            "updated_at": utc_now_iso(),
            "status": status,
            "top_n": max(1, int(top_n)),
            "with_prices": bool(with_prices),
            "thresholds_effective": thresholds_effective,
            "batch_size": max(1, int(batch_size)),
            "total_tickers": len(universe_tickers),
            "total_batches": total_batches,
            "done_batches": sorted(done_batches),
            "batches_done": len(done_batches),
            "last_completed_batch": max(done_batches) if done_batches else -1,
            "budgets_effective": _budget_progress().get("effective"),
            "budgets_used": _budget_progress().get("used"),
            "budget_remaining": _budget_progress().get("remaining"),
            "tickers_completed": len(scoreboard_map),
            "remaining_tickers": max(0, len(universe_tickers) - len(scoreboard_map)),
            "current_batch_index": progress_state.get("current_batch_index"),
            "current_batch_tickers": list(progress_state.get("current_batch_tickers") or []),
            "current_ticker": str(progress_state.get("current_ticker") or ""),
            "hydration_phase": str(progress_state.get("hydration_phase") or PHASE_NOT_STARTED),
            "hydration_status": str(progress_state.get("hydration_status") or HYDRATION_STATUS_IDLE),
            "hydration_started_at": progress_state.get("hydration_started_at"),
            "hydration_last_progress_at": progress_state.get("hydration_last_progress_at"),
            "last_progress_phase": str(progress_state.get("last_progress_phase") or PHASE_NOT_STARTED),
            "last_completed_ticker": str(progress_state.get("last_completed_ticker") or ""),
            "facts_cache_hits": int(progress_state.get("facts_cache_hits") or 0),
            "facts_cache_misses": int(progress_state.get("facts_cache_misses") or 0),
            "companyfacts_fetch_attempts": int(progress_state.get("companyfacts_fetch_attempts") or 0),
            "companyfacts_failures": int(progress_state.get("companyfacts_failures") or 0),
            "companyfacts_timeouts": int(progress_state.get("companyfacts_timeouts") or 0),
            "price_stage_completed": bool(progress_state.get("price_stage_completed", False)),
            "facts_stage_completed": bool(progress_state.get("facts_stage_completed", False)),
            "scoring_stage_completed": bool(progress_state.get("scoring_stage_completed", False)),
            "finalization_started": bool(progress_state.get("finalization_started", False)),
            "primary_scout_blocker": str(progress_state.get("primary_scout_blocker") or ""),
            "stalled_reason_code": str(progress_state.get("stalled_reason_code") or ""),
            "stop_reason_code": stop_code,
            "stop_summary": stop_text,
            "universe_input_path": str(paths["universe_input_path"]),
        }
        _json_write(paths["state_path"], state_payload)

    def _write_outputs(status: str, *, stop_code: str, stop_text: str) -> dict[str, Any]:
        net_debt_payload = write_net_debt_coverage_for_run(
            run_id=run_id,
            as_of_date=as_of_date,
            tickers=sorted(scoreboard_map.keys()),
            output_path=paths["net_debt_path"],
            resolved_by_ticker=net_debt_resolved_map,
            cfg=cfg,
        )
        owner_earnings_quality_payload = write_owner_earnings_quality_for_run(
            run_id=run_id,
            as_of_date=as_of_date,
            tickers=sorted(scoreboard_map.keys()),
            output_path=paths["owner_earnings_quality_path"],
            scoreboard_rows=list(scoreboard_map.values()),
            facts_rows_by_ticker={ticker: {} for ticker in scoreboard_map.keys()},
            cfg=cfg,
        )
        maintenance_capex_run_payload = write_maintenance_capex_discipline_for_run(
            run_id=run_id,
            as_of_date=as_of_date,
            tickers=sorted(scoreboard_map.keys()),
            output_path=paths["maintenance_capex_discipline_path"],
            scoreboard_rows=list(scoreboard_map.values()),
            cfg=cfg,
        )
        _maintenance_rows_by_ticker = {
            str(r.get("ticker") or "").upper(): r
            for r in (maintenance_capex_run_payload.get("rows") or [])
            if isinstance(r, dict)
        }
        for _ticker_key, _score_row in scoreboard_map.items():
            _maintenance_row = _maintenance_rows_by_ticker.get(str(_ticker_key).upper())
            if isinstance(_maintenance_row, dict):
                _maintenance_refs = [
                    str(ref) for ref in (_maintenance_row.get("derived_from") or []) if str(ref).strip()
                ]
                _asset_class = str(
                    _maintenance_row.get("asset_intensity_class") or "ASSET_INTENSITY_UNKNOWN"
                )
                _asset_reason_codes = [
                    str(code)
                    for code in (_maintenance_row.get("asset_intensity_reason_codes") or [])
                    if str(code).strip()
                ]
                _credibility_class = str(
                    _maintenance_row.get("maintenance_capex_credibility_class")
                    or "MAINTENANCE_CAPEX_CREDIBILITY_UNKNOWN"
                )
                _credibility_reason_codes = [
                    str(code)
                    for code in (_maintenance_row.get("maintenance_capex_credibility_reason_codes") or [])
                    if str(code).strip()
                ]
                _score_row["maintenance_capex_discipline_detail"] = _maintenance_row
                _score_row["asset_intensity_class"] = _asset_class
                _score_row["asset_intensity_reason_codes"] = _asset_reason_codes
                _score_row["maintenance_capex_credibility_class"] = _credibility_class
                _score_row["maintenance_capex_credibility_reason_codes"] = _credibility_reason_codes
                _score_row["maintenance_capex_support_signals"] = [
                    str(code)
                    for code in (_maintenance_row.get("maintenance_capex_support_signals") or [])
                    if str(code).strip()
                ]
                _score_row["maintenance_capex_headwind_signals"] = [
                    str(code)
                    for code in (_maintenance_row.get("maintenance_capex_headwind_signals") or [])
                    if str(code).strip()
                ]
                _score_row["primary_maintenance_capex_caution"] = str(
                    _maintenance_row.get("primary_maintenance_capex_caution") or "OWNER_EARNINGS_UNCLEAR"
                )
                _score_row["maintenance_capex_discipline_summary"] = str(
                    _maintenance_row.get("maintenance_capex_discipline_summary") or ""
                )
                _score_row["metric_values"]["asset_intensity_class"] = _asset_class
                _score_row["metric_values"]["maintenance_capex_credibility_class"] = _credibility_class
                _score_row["inputs_used"]["maintenance_capex_asset_intensity_discipline"] = {
                    "value": {
                        "asset_intensity_class": _asset_class,
                        "maintenance_capex_credibility_class": _credibility_class,
                        "primary_maintenance_capex_caution": _score_row["primary_maintenance_capex_caution"],
                    },
                    "derived_from": _maintenance_refs,
                    "reason_codes": _sorted_unique(_asset_reason_codes + _credibility_reason_codes),
                }
                _score_row["derived_from"] = sorted(
                    set([str(ref) for ref in (_score_row.get("derived_from") or []) if str(ref).strip()] + _maintenance_refs)
                )
                _coverage_row = coverage_map.get(_ticker_key)
                if isinstance(_coverage_row, dict):
                    _coverage_row.update(
                        {
                            "asset_intensity_class": _asset_class,
                            "asset_intensity_reason_codes": _asset_reason_codes,
                            "maintenance_capex_credibility_class": _credibility_class,
                            "maintenance_capex_credibility_reason_codes": _credibility_reason_codes,
                            "maintenance_capex_support_signals": list(
                                _score_row.get("maintenance_capex_support_signals") or []
                            ),
                            "maintenance_capex_headwind_signals": list(
                                _score_row.get("maintenance_capex_headwind_signals") or []
                            ),
                            "primary_maintenance_capex_caution": _score_row["primary_maintenance_capex_caution"],
                            "maintenance_capex_discipline_summary": _score_row["maintenance_capex_discipline_summary"],
                            "derived_from": _score_row["derived_from"],
                        }
                    )
                _yield_row = yield_map.get(_ticker_key)
                if isinstance(_yield_row, dict):
                    _yield_row.update(
                        {
                            "asset_intensity_class": _asset_class,
                            "asset_intensity_reason_codes": _asset_reason_codes,
                            "maintenance_capex_credibility_class": _credibility_class,
                            "maintenance_capex_credibility_reason_codes": _credibility_reason_codes,
                            "maintenance_capex_support_signals": list(
                                _score_row.get("maintenance_capex_support_signals") or []
                            ),
                            "maintenance_capex_headwind_signals": list(
                                _score_row.get("maintenance_capex_headwind_signals") or []
                            ),
                            "primary_maintenance_capex_caution": _score_row["primary_maintenance_capex_caution"],
                            "maintenance_capex_discipline_summary": _score_row["maintenance_capex_discipline_summary"],
                            "derived_from": _score_row["derived_from"],
                        }
                    )
        intangible_economics_payload = write_intangible_economics_for_run(
            run_id=run_id,
            as_of_date=as_of_date,
            tickers=sorted(scoreboard_map.keys()),
            output_path=paths["intangible_economics_path"],
            scoreboard_rows=list(scoreboard_map.values()),
            cfg=cfg,
        )
        intrinsic_discipline_payload = write_intrinsic_discipline_for_run(
            run_id=run_id,
            as_of_date=as_of_date,
            tickers=sorted(scoreboard_map.keys()),
            output_path=paths["intrinsic_discipline_path"],
            scoreboard_rows=list(scoreboard_map.values()),
            cfg=cfg,
        )
        valuation_integrity_payload = write_valuation_integrity_for_run(
            run_id=run_id,
            as_of_date=as_of_date,
            tickers=sorted(scoreboard_map.keys()),
            output_path=paths["valuation_integrity_path"],
            scoreboard_rows=list(scoreboard_map.values()),
            cfg=cfg,
        )
        fundamental_regression_payload = write_fundamental_regression_analytics_for_run(
            run_id=run_id,
            as_of_date=as_of_date,
            tickers=sorted(scoreboard_map.keys()),
            output_path=paths["fundamental_regression_analytics_path"],
            markdown_path=paths["fundamental_regression_analytics_md_path"],
            scoreboard_rows=list(scoreboard_map.values()),
            net_debt_resolved_by_ticker=net_debt_resolved_map,
            cfg=cfg,
        )
        integrity_rows_by_ticker = {
            str(row.get("ticker") or "").upper(): row
            for row in (valuation_integrity_payload.get("rows") or [])
            if isinstance(row, dict)
        }
        for ticker_key, score_row in scoreboard_map.items():
            integrity_row = integrity_rows_by_ticker.get(str(ticker_key).upper())
            if not isinstance(integrity_row, dict):
                continue
            integrity_refs = [
                str(ref)
                for ref in (integrity_row.get("derived_from") or [])
                if str(ref).strip()
            ]
            integrity_class = str(integrity_row.get("valuation_integrity_class") or UNKNOWN)
            integrity_reason_codes = [
                str(code)
                for code in (integrity_row.get("valuation_integrity_reason_codes") or [])
                if str(code).strip()
            ]
            consistency_reason_codes = [
                str(code)
                for code in (integrity_row.get("valuation_consistency_reason_codes") or [])
                if str(code).strip()
            ]
            uniformity_reason_codes = [
                str(code)
                for code in (integrity_row.get("valuation_uniformity_reason_codes") or [])
                if str(code).strip()
            ]
            uniformity_group_id = (
                str(integrity_row.get("valuation_uniformity_group_id"))
                if str(integrity_row.get("valuation_uniformity_group_id") or "").strip()
                else None
            )
            score_row["valuation_integrity_detail"] = integrity_row
            score_row["valuation_integrity_class"] = integrity_class
            score_row["valuation_integrity_reason_codes"] = integrity_reason_codes
            score_row["valuation_consistency_status"] = str(
                integrity_row.get("valuation_consistency_status") or UNKNOWN
            )
            score_row["valuation_consistency_reason_codes"] = consistency_reason_codes
            score_row["valuation_uniformity_group_id"] = uniformity_group_id
            score_row["valuation_uniformity_reason_codes"] = uniformity_reason_codes
            score_row["valuation_input_fingerprint"] = str(
                integrity_row.get("valuation_input_fingerprint") or ""
            )
            score_row["valuation_input_provenance_summary"] = (
                integrity_row.get("valuation_input_provenance_summary")
                if isinstance(integrity_row.get("valuation_input_provenance_summary"), dict)
                else {}
            )
            score_row["metric_values"]["valuation_integrity_class"] = integrity_class
            score_row["inputs_used"]["valuation_integrity_audit"] = {
                "value": {
                    "valuation_integrity_class": integrity_class,
                    "valuation_uniformity_group_id": uniformity_group_id or UNKNOWN,
                },
                "derived_from": integrity_refs,
                "reason_codes": integrity_reason_codes,
            }
            score_row["derived_from"] = sorted(
                set(
                    [str(ref) for ref in (score_row.get("derived_from") or []) if str(ref).strip()]
                    + integrity_refs
                )
            )
            adjusted_confidence = apply_valuation_integrity_headwind(
                score_row.get("valuation_confidence_detail")
                if isinstance(score_row.get("valuation_confidence_detail"), dict)
                else {},
                integrity_class=integrity_class,
                integrity_reason_codes=integrity_reason_codes,
                derived_from=integrity_refs,
            )
            score_row["valuation_confidence_detail"] = adjusted_confidence
            score_row["valuation_confidence_class"] = str(
                adjusted_confidence.get("valuation_confidence_class") or UNKNOWN
            )
            score_row["valuation_confidence_reason_codes"] = [
                str(code)
                for code in (adjusted_confidence.get("valuation_confidence_reason_codes") or [])
                if str(code).strip()
            ]
            score_row["metric_values"]["valuation_confidence_class"] = score_row["valuation_confidence_class"]
            coverage_row = coverage_map.get(ticker_key)
            if isinstance(coverage_row, dict):
                coverage_row.update(
                    {
                        "valuation_integrity_class": integrity_class,
                        "valuation_integrity_reason_codes": integrity_reason_codes,
                        "valuation_consistency_status": score_row["valuation_consistency_status"],
                        "valuation_consistency_reason_codes": consistency_reason_codes,
                        "valuation_uniformity_group_id": uniformity_group_id,
                        "valuation_uniformity_reason_codes": uniformity_reason_codes,
                        "valuation_confidence_class": score_row["valuation_confidence_class"],
                        "valuation_confidence_reason_codes": list(
                            score_row.get("valuation_confidence_reason_codes") or []
                        ),
                        "derived_from": score_row["derived_from"],
                    }
                )
            yield_row = yield_map.get(ticker_key)
            if isinstance(yield_row, dict):
                yield_row.update(
                    {
                        "valuation_integrity_class": integrity_class,
                        "valuation_integrity_reason_codes": integrity_reason_codes,
                        "valuation_consistency_status": score_row["valuation_consistency_status"],
                        "valuation_consistency_reason_codes": consistency_reason_codes,
                        "valuation_uniformity_group_id": uniformity_group_id,
                        "valuation_uniformity_reason_codes": uniformity_reason_codes,
                        "valuation_confidence_class": score_row["valuation_confidence_class"],
                        "valuation_confidence_reason_codes": list(
                            score_row.get("valuation_confidence_reason_codes") or []
                        ),
                        "derived_from": score_row["derived_from"],
                    }
                )
        valuation_confidence_payload = write_valuation_confidence_for_run(
            run_id=run_id,
            as_of_date=as_of_date,
            tickers=sorted(scoreboard_map.keys()),
            output_path=paths["valuation_confidence_path"],
            scoreboard_rows=list(scoreboard_map.values()),
            coverage_rows_by_ticker=dict(coverage_map),
            cfg=cfg,
        )
        evidence_sufficiency_payload = write_evidence_sufficiency_for_run(
            run_id=run_id,
            as_of_date=as_of_date,
            tickers=sorted(scoreboard_map.keys()),
            output_path=paths["evidence_sufficiency_path"],
            scoreboard_rows=list(scoreboard_map.values()),
            cfg=cfg,
        )
        evidence_rows_by_ticker = {
            str(row.get("ticker") or "").upper(): row
            for row in (evidence_sufficiency_payload.get("rows") or [])
            if isinstance(row, dict)
        }
        for ticker_key, score_row in scoreboard_map.items():
            evidence_row = evidence_rows_by_ticker.get(str(ticker_key).upper())
            if not isinstance(evidence_row, dict):
                continue
            evidence_refs = [
                str(ref)
                for ref in (evidence_row.get("derived_from") or [])
                if str(ref).strip()
            ]
            evidence_class = str(evidence_row.get("evidence_sufficiency_class") or UNKNOWN)
            evidence_reason_codes = [
                str(code)
                for code in (evidence_row.get("evidence_sufficiency_reason_codes") or [])
                if str(code).strip()
            ]
            mos_assessment_status = str(evidence_row.get("mos_assessment_status") or UNKNOWN)
            mos_guardrail_reason_codes = [
                str(code)
                for code in (evidence_row.get("mos_guardrail_reason_codes") or [])
                if str(code).strip()
            ]
            score_row["evidence_sufficiency_detail"] = evidence_row
            score_row["evidence_sufficiency_class"] = evidence_class
            score_row["evidence_sufficiency_reason_codes"] = evidence_reason_codes
            score_row["mos_assessment_status"] = mos_assessment_status
            score_row["mos_guardrail_reason_codes"] = mos_guardrail_reason_codes
            score_row["metric_values"]["evidence_sufficiency_class"] = evidence_class
            score_row["inputs_used"]["evidence_sufficiency"] = {
                "value": {
                    "evidence_sufficiency_class": evidence_class,
                    "mos_assessment_status": mos_assessment_status,
                },
                "derived_from": evidence_refs,
                "reason_codes": evidence_reason_codes + mos_guardrail_reason_codes,
            }
            score_row["derived_from"] = sorted(
                set(
                    [str(ref) for ref in (score_row.get("derived_from") or []) if str(ref).strip()]
                    + evidence_refs
                )
            )
            coverage_row = coverage_map.get(ticker_key)
            if isinstance(coverage_row, dict):
                coverage_row.update(
                    {
                        "evidence_sufficiency_class": evidence_class,
                        "evidence_sufficiency_reason_codes": evidence_reason_codes,
                        "mos_assessment_status": mos_assessment_status,
                        "mos_guardrail_reason_codes": mos_guardrail_reason_codes,
                        "derived_from": score_row["derived_from"],
                    }
                )
            yield_row = yield_map.get(ticker_key)
            if isinstance(yield_row, dict):
                yield_row.update(
                    {
                        "evidence_sufficiency_class": evidence_class,
                        "evidence_sufficiency_reason_codes": evidence_reason_codes,
                        "mos_assessment_status": mos_assessment_status,
                        "mos_guardrail_reason_codes": mos_guardrail_reason_codes,
                        "derived_from": score_row["derived_from"],
                    }
                )
        value_type_payload = write_value_type_for_run(
            run_id=run_id,
            as_of_date=as_of_date,
            tickers=sorted(scoreboard_map.keys()),
            output_path=paths["value_type_path"],
            scoreboard_rows=list(scoreboard_map.values()),
            cfg=cfg,
        )
        cyclical_normalization_run_payload = write_cyclical_normalization_for_run(
            run_id=run_id,
            as_of_date=as_of_date,
            tickers=sorted(scoreboard_map.keys()),
            output_path=paths["cyclical_normalization_path"],
            scoreboard_rows=list(scoreboard_map.values()),
            cfg=cfg,
        )
        # Propagate cyclical normalization fields before impairment + readiness
        _cycl_rows_by_ticker = {
            str(r.get("ticker") or "").upper(): r
            for r in (cyclical_normalization_run_payload.get("rows") or [])
            if isinstance(r, dict)
        }
        for _ticker_key, _score_row in scoreboard_map.items():
            _cycl_row = _cycl_rows_by_ticker.get(str(_ticker_key).upper())
            if isinstance(_cycl_row, dict):
                _score_row["cyclical_normalization_detail"] = _cycl_row
                for _field in [
                    "cyclical_profile_class",
                    "cycle_position_class",
                    "cyclical_valuation_risk_class",
                    "conservative_cyclical_denominator",
                    "cycle_aware_value_support_summary",
                    "intrinsic_cycle_awareness_status",
                ]:
                    if _field in _cycl_row:
                        _score_row[_field] = _cycl_row[_field]
        impairment_classification_run_payload = write_impairment_classification_for_run(
            run_id=run_id,
            as_of_date=as_of_date,
            tickers=sorted(scoreboard_map.keys()),
            output_path=paths["impairment_classification_path"],
            scoreboard_rows=list(scoreboard_map.values()),
            cfg=cfg,
        )
        # Propagate impairment classification fields before readiness
        _impairment_rows_by_ticker = {
            str(r.get("ticker") or "").upper(): r
            for r in (impairment_classification_run_payload.get("rows") or [])
            if isinstance(r, dict)
        }
        for _ticker_key, _score_row in scoreboard_map.items():
            _imp_row = _impairment_rows_by_ticker.get(str(_ticker_key).upper())
            if isinstance(_imp_row, dict):
                _score_row["impairment_classification_detail"] = _imp_row
                _score_row["impairment_class_primary"] = str(
                    _imp_row.get("impairment_class_primary") or "IMPAIRMENT_UNKNOWN"
                )
                _score_row["primary_underwriting_caution"] = str(
                    _imp_row.get("primary_underwriting_caution") or "CAUTION_UNKNOWN"
                )
                _score_row["impairment_class_reason_codes"] = [
                    str(code)
                    for code in (_imp_row.get("impairment_class_reason_codes") or [])
                    if str(code).strip()
                ]
        normalization_credibility_run_payload = write_normalization_credibility_for_run(
            run_id=run_id,
            as_of_date=as_of_date,
            tickers=sorted(scoreboard_map.keys()),
            output_path=paths["normalization_credibility_path"],
            scoreboard_rows=list(scoreboard_map.values()),
            cfg=cfg,
        )
        _norm_cred_rows_by_ticker = {
            str(r.get("ticker") or "").upper(): r
            for r in (normalization_credibility_run_payload.get("rows") or [])
            if isinstance(r, dict)
        }
        for _ticker_key, _score_row in scoreboard_map.items():
            _norm_cred_row = _norm_cred_rows_by_ticker.get(str(_ticker_key).upper())
            if isinstance(_norm_cred_row, dict):
                _score_row["normalization_credibility_detail"] = _norm_cred_row
                _score_row["normalization_credibility_class"] = str(
                    _norm_cred_row.get("normalization_credibility_class") or "NORMALIZATION_CREDIBILITY_UNKNOWN"
                )
                _score_row["primary_normalization_caution"] = str(
                    _norm_cred_row.get("primary_normalization_caution") or "NORMALIZATION_UNCLEAR"
                )
                _score_row["normalization_credibility_reason_codes"] = [
                    str(code)
                    for code in (_norm_cred_row.get("normalization_credibility_reason_codes") or [])
                    if str(code).strip()
                ]
        capital_allocation_discipline_run_payload = write_capital_allocation_discipline_for_run(
            run_id=run_id,
            as_of_date=as_of_date,
            tickers=sorted(scoreboard_map.keys()),
            output_path=paths["capital_allocation_discipline_path"],
            scoreboard_rows=list(scoreboard_map.values()),
            cfg=cfg,
        )
        _cap_alloc_rows_by_ticker = {
            str(r.get("ticker") or "").upper(): r
            for r in (capital_allocation_discipline_run_payload.get("rows") or [])
            if isinstance(r, dict)
        }
        for _ticker_key, _score_row in scoreboard_map.items():
            _cap_alloc_row = _cap_alloc_rows_by_ticker.get(str(_ticker_key).upper())
            if isinstance(_cap_alloc_row, dict):
                _score_row["capital_allocation_discipline_detail"] = _cap_alloc_row
                _score_row["capital_allocation_discipline_class"] = str(
                    _cap_alloc_row.get("capital_allocation_discipline_class") or "CAPITAL_ALLOCATION_UNKNOWN"
                )
                _score_row["primary_capital_allocation_caution"] = str(
                    _cap_alloc_row.get("primary_capital_allocation_caution") or "CAPITAL_ALLOCATION_UNCLEAR"
                )
                _score_row["capital_allocation_discipline_reason_codes"] = [
                    str(code)
                    for code in (_cap_alloc_row.get("capital_allocation_discipline_reason_codes") or [])
                    if str(code).strip()
                ]
        accounting_quality_payload = write_accounting_quality_for_run(
            run_id=run_id,
            as_of_date=as_of_date,
            tickers=sorted(scoreboard_map.keys()),
            output_path=paths["accounting_quality_path"],
            scoreboard_rows=list(scoreboard_map.values()),
            cfg=cfg,
        )
        balance_sheet_stress_run_payload = write_balance_sheet_stress_for_run(
            run_id=run_id,
            as_of_date=as_of_date,
            tickers=sorted(scoreboard_map.keys()),
            output_path=paths["balance_sheet_stress_path"],
            scoreboard_rows=list(scoreboard_map.values()),
            cfg=cfg,
        )
        _accounting_rows_by_ticker = {
            str(r.get("ticker") or "").upper(): r
            for r in (accounting_quality_payload.get("rows") or [])
            if isinstance(r, dict)
        }
        _balance_sheet_rows_by_ticker = {
            str(r.get("ticker") or "").upper(): r
            for r in (balance_sheet_stress_run_payload.get("rows") or [])
            if isinstance(r, dict)
        }
        for _ticker_key, _score_row in scoreboard_map.items():
            _accounting_row = _accounting_rows_by_ticker.get(str(_ticker_key).upper())
            if isinstance(_accounting_row, dict):
                _accounting_refs = [
                    str(ref)
                    for ref in (_accounting_row.get("derived_from") or [])
                    if str(ref).strip()
                ]
                _accounting_class = str(
                    _accounting_row.get("accounting_quality_class") or "ACCOUNTING_QUALITY_UNKNOWN"
                )
                _accounting_reason_codes = [
                    str(code)
                    for code in (_accounting_row.get("accounting_quality_reason_codes") or [])
                    if str(code).strip()
                ]
                _score_row["accounting_quality_detail"] = _accounting_row
                _score_row["accounting_quality_class"] = _accounting_class
                _score_row["accounting_quality_reason_codes"] = _accounting_reason_codes
                _score_row["cash_earnings_support_signals"] = [
                    str(code)
                    for code in (_accounting_row.get("cash_earnings_support_signals") or [])
                    if str(code).strip()
                ]
                _score_row["cash_earnings_headwind_signals"] = [
                    str(code)
                    for code in (_accounting_row.get("cash_earnings_headwind_signals") or [])
                    if str(code).strip()
                ]
                _score_row["primary_accounting_caution"] = str(
                    _accounting_row.get("primary_accounting_caution") or "ACCOUNTING_QUALITY_UNCLEAR"
                )
                _score_row["cash_earnings_discipline_summary"] = str(
                    _accounting_row.get("cash_earnings_discipline_summary") or ""
                )
                _score_row["metric_values"]["accounting_quality_class"] = _accounting_class
                _score_row["inputs_used"]["accounting_quality_cash_earnings_discipline"] = {
                    "value": {
                        "accounting_quality_class": _accounting_class,
                        "primary_accounting_caution": _score_row["primary_accounting_caution"],
                    },
                    "derived_from": _accounting_refs,
                    "reason_codes": _accounting_reason_codes,
                }
                _score_row["derived_from"] = sorted(
                    set(
                        [
                            str(ref)
                            for ref in (_score_row.get("derived_from") or [])
                            if str(ref).strip()
                        ]
                        + _accounting_refs
                    )
                )
                _coverage_row = coverage_map.get(_ticker_key)
                if isinstance(_coverage_row, dict):
                    _coverage_row.update(
                        {
                            "accounting_quality_class": _accounting_class,
                            "accounting_quality_reason_codes": _accounting_reason_codes,
                            "cash_earnings_support_signals": list(
                                _score_row.get("cash_earnings_support_signals") or []
                            ),
                            "cash_earnings_headwind_signals": list(
                                _score_row.get("cash_earnings_headwind_signals") or []
                            ),
                            "primary_accounting_caution": _score_row["primary_accounting_caution"],
                            "cash_earnings_discipline_summary": _score_row["cash_earnings_discipline_summary"],
                            "derived_from": _score_row["derived_from"],
                        }
                    )
                _yield_row = yield_map.get(_ticker_key)
                if isinstance(_yield_row, dict):
                    _yield_row.update(
                        {
                            "accounting_quality_class": _accounting_class,
                            "accounting_quality_reason_codes": _accounting_reason_codes,
                            "cash_earnings_support_signals": list(
                                _score_row.get("cash_earnings_support_signals") or []
                            ),
                            "cash_earnings_headwind_signals": list(
                                _score_row.get("cash_earnings_headwind_signals") or []
                            ),
                            "primary_accounting_caution": _score_row["primary_accounting_caution"],
                            "cash_earnings_discipline_summary": _score_row["cash_earnings_discipline_summary"],
                            "derived_from": _score_row["derived_from"],
                        }
                    )
            _balance_sheet_row = _balance_sheet_rows_by_ticker.get(str(_ticker_key).upper())
            if isinstance(_balance_sheet_row, dict):
                _balance_sheet_refs = [
                    str(ref)
                    for ref in (_balance_sheet_row.get("derived_from") or [])
                    if str(ref).strip()
                ]
                _balance_sheet_class = str(
                    _balance_sheet_row.get("balance_sheet_stress_class")
                    or "BALANCE_SHEET_STRESS_UNKNOWN"
                )
                _balance_sheet_reason_codes = [
                    str(code)
                    for code in (_balance_sheet_row.get("balance_sheet_stress_reason_codes") or [])
                    if str(code).strip()
                ]
                _refinancing_class = str(
                    _balance_sheet_row.get("refinancing_risk_class") or "REFINANCING_RISK_UNKNOWN"
                )
                _refinancing_reason_codes = [
                    str(code)
                    for code in (_balance_sheet_row.get("refinancing_risk_reason_codes") or [])
                    if str(code).strip()
                ]
                _score_row["balance_sheet_stress_detail"] = _balance_sheet_row
                _score_row["balance_sheet_stress_class"] = _balance_sheet_class
                _score_row["balance_sheet_stress_reason_codes"] = _balance_sheet_reason_codes
                _score_row["refinancing_risk_class"] = _refinancing_class
                _score_row["refinancing_risk_reason_codes"] = _refinancing_reason_codes
                _score_row["balance_sheet_support_signals"] = [
                    str(code)
                    for code in (_balance_sheet_row.get("balance_sheet_support_signals") or [])
                    if str(code).strip()
                ]
                _score_row["balance_sheet_headwind_signals"] = [
                    str(code)
                    for code in (_balance_sheet_row.get("balance_sheet_headwind_signals") or [])
                    if str(code).strip()
                ]
                _score_row["primary_balance_sheet_caution"] = str(
                    _balance_sheet_row.get("primary_balance_sheet_caution") or "BALANCE_SHEET_UNCLEAR"
                )
                _score_row["balance_sheet_discipline_summary"] = str(
                    _balance_sheet_row.get("balance_sheet_discipline_summary") or ""
                )
                _score_row["metric_values"]["balance_sheet_stress_class"] = _balance_sheet_class
                _score_row["metric_values"]["refinancing_risk_class"] = _refinancing_class
                _score_row["inputs_used"]["balance_sheet_stress_refinancing_risk"] = {
                    "value": {
                        "balance_sheet_stress_class": _balance_sheet_class,
                        "refinancing_risk_class": _refinancing_class,
                        "primary_balance_sheet_caution": _score_row["primary_balance_sheet_caution"],
                    },
                    "derived_from": _balance_sheet_refs,
                    "reason_codes": _sorted_unique(
                        _balance_sheet_reason_codes + _refinancing_reason_codes
                    ),
                }
                _score_row["derived_from"] = sorted(
                    set(
                        [
                            str(ref)
                            for ref in (_score_row.get("derived_from") or [])
                            if str(ref).strip()
                        ]
                        + _balance_sheet_refs
                    )
                )
                _coverage_row = coverage_map.get(_ticker_key)
                if isinstance(_coverage_row, dict):
                    _coverage_row.update(
                        {
                            "balance_sheet_stress_class": _balance_sheet_class,
                            "balance_sheet_stress_reason_codes": _balance_sheet_reason_codes,
                            "refinancing_risk_class": _refinancing_class,
                            "refinancing_risk_reason_codes": _refinancing_reason_codes,
                            "balance_sheet_support_signals": list(
                                _score_row.get("balance_sheet_support_signals") or []
                            ),
                            "balance_sheet_headwind_signals": list(
                                _score_row.get("balance_sheet_headwind_signals") or []
                            ),
                            "primary_balance_sheet_caution": _score_row["primary_balance_sheet_caution"],
                            "balance_sheet_discipline_summary": _score_row["balance_sheet_discipline_summary"],
                            "derived_from": _score_row["derived_from"],
                        }
                    )
                _yield_row = yield_map.get(_ticker_key)
                if isinstance(_yield_row, dict):
                    _yield_row.update(
                        {
                            "balance_sheet_stress_class": _balance_sheet_class,
                            "balance_sheet_stress_reason_codes": _balance_sheet_reason_codes,
                            "refinancing_risk_class": _refinancing_class,
                            "refinancing_risk_reason_codes": _refinancing_reason_codes,
                            "balance_sheet_support_signals": list(
                                _score_row.get("balance_sheet_support_signals") or []
                            ),
                            "balance_sheet_headwind_signals": list(
                                _score_row.get("balance_sheet_headwind_signals") or []
                            ),
                            "primary_balance_sheet_caution": _score_row["primary_balance_sheet_caution"],
                            "balance_sheet_discipline_summary": _score_row["balance_sheet_discipline_summary"],
                            "derived_from": _score_row["derived_from"],
                        }
                    )
        reinvestment_efficiency_payload = write_reinvestment_efficiency_for_run(
            run_id=run_id,
            as_of_date=as_of_date,
            tickers=sorted(scoreboard_map.keys()),
            output_path=paths["reinvestment_efficiency_path"],
            scoreboard_rows=list(scoreboard_map.values()),
            cfg=cfg,
        )
        _reinvestment_rows_by_ticker = {
            str(r.get("ticker") or "").upper(): r
            for r in (reinvestment_efficiency_payload.get("rows") or [])
            if isinstance(r, dict)
        }
        for _ticker_key, _score_row in scoreboard_map.items():
            _reinvestment_row = _reinvestment_rows_by_ticker.get(str(_ticker_key).upper())
            if isinstance(_reinvestment_row, dict):
                _reinvestment_refs = [
                    str(ref)
                    for ref in (_reinvestment_row.get("derived_from") or [])
                    if str(ref).strip()
                ]
                _reinvestment_class = str(
                    _reinvestment_row.get("reinvestment_efficiency_class")
                    or "REINVESTMENT_EFFICIENCY_UNKNOWN"
                )
                _reinvestment_reason_codes = [
                    str(code)
                    for code in (_reinvestment_row.get("reinvestment_efficiency_reason_codes") or [])
                    if str(code).strip()
                ]
                _score_row["reinvestment_efficiency_detail"] = _reinvestment_row
                _score_row["reinvestment_efficiency_class"] = _reinvestment_class
                _score_row["reinvestment_efficiency_reason_codes"] = _reinvestment_reason_codes
                _score_row["reinvestment_support_signals"] = [
                    str(code)
                    for code in (_reinvestment_row.get("reinvestment_support_signals") or [])
                    if str(code).strip()
                ]
                _score_row["reinvestment_headwind_signals"] = [
                    str(code)
                    for code in (_reinvestment_row.get("reinvestment_headwind_signals") or [])
                    if str(code).strip()
                ]
                _score_row["primary_reinvestment_caution"] = str(
                    _reinvestment_row.get("primary_reinvestment_caution") or "REINVESTMENT_UNCLEAR"
                )
                _score_row["reinvestment_efficiency_summary"] = str(
                    _reinvestment_row.get("reinvestment_efficiency_summary") or ""
                )
                _score_row["metric_values"]["reinvestment_efficiency_class"] = _reinvestment_class
                _score_row["inputs_used"]["incremental_reinvestment_efficiency"] = {
                    "value": {
                        "reinvestment_efficiency_class": _reinvestment_class,
                        "primary_reinvestment_caution": _score_row["primary_reinvestment_caution"],
                    },
                    "derived_from": _reinvestment_refs,
                    "reason_codes": _reinvestment_reason_codes,
                }
                _score_row["derived_from"] = sorted(
                    set(
                        [
                            str(ref)
                            for ref in (_score_row.get("derived_from") or [])
                            if str(ref).strip()
                        ]
                        + _reinvestment_refs
                    )
                )
                _coverage_row = coverage_map.get(_ticker_key)
                if isinstance(_coverage_row, dict):
                    _coverage_row.update(
                        {
                            "reinvestment_efficiency_class": _reinvestment_class,
                            "reinvestment_efficiency_reason_codes": _reinvestment_reason_codes,
                            "reinvestment_support_signals": list(
                                _score_row.get("reinvestment_support_signals") or []
                            ),
                            "reinvestment_headwind_signals": list(
                                _score_row.get("reinvestment_headwind_signals") or []
                            ),
                            "primary_reinvestment_caution": _score_row["primary_reinvestment_caution"],
                            "reinvestment_efficiency_summary": _score_row["reinvestment_efficiency_summary"],
                            "derived_from": _score_row["derived_from"],
                        }
                    )
                _yield_row = yield_map.get(_ticker_key)
                if isinstance(_yield_row, dict):
                    _yield_row.update(
                        {
                            "reinvestment_efficiency_class": _reinvestment_class,
                            "reinvestment_efficiency_reason_codes": _reinvestment_reason_codes,
                            "reinvestment_support_signals": list(
                                _score_row.get("reinvestment_support_signals") or []
                            ),
                            "reinvestment_headwind_signals": list(
                                _score_row.get("reinvestment_headwind_signals") or []
                            ),
                            "primary_reinvestment_caution": _score_row["primary_reinvestment_caution"],
                            "reinvestment_efficiency_summary": _score_row["reinvestment_efficiency_summary"],
                            "derived_from": _score_row["derived_from"],
                        }
                    )
        revenue_dependence_run_payload = write_revenue_dependence_for_run(
            run_id=run_id,
            as_of_date=as_of_date,
            tickers=sorted(scoreboard_map.keys()),
            output_path=paths["revenue_dependence_path"],
            scoreboard_rows=list(scoreboard_map.values()),
            cfg=cfg,
        )
        _revenue_rows_by_ticker = {
            str(r.get("ticker") or "").upper(): r
            for r in (revenue_dependence_run_payload.get("rows") or [])
            if isinstance(r, dict)
        }
        for _ticker_key, _score_row in scoreboard_map.items():
            _revenue_row = _revenue_rows_by_ticker.get(str(_ticker_key).upper())
            if isinstance(_revenue_row, dict):
                _revenue_refs = [
                    str(ref)
                    for ref in (_revenue_row.get("derived_from") or [])
                    if str(ref).strip()
                ]
                _revenue_class = str(
                    _revenue_row.get("revenue_dependence_risk_class") or "REVENUE_DEPENDENCE_UNKNOWN"
                )
                _revenue_reason_codes = [
                    str(code)
                    for code in (_revenue_row.get("revenue_dependence_risk_reason_codes") or [])
                    if str(code).strip()
                ]
                _score_row["revenue_dependence_detail"] = _revenue_row
                _score_row["revenue_dependence_risk_class"] = _revenue_class
                _score_row["revenue_dependence_risk_reason_codes"] = _revenue_reason_codes
                _score_row["revenue_dependence_support_signals"] = [
                    str(code)
                    for code in (_revenue_row.get("revenue_dependence_support_signals") or [])
                    if str(code).strip()
                ]
                _score_row["revenue_dependence_headwind_signals"] = [
                    str(code)
                    for code in (_revenue_row.get("revenue_dependence_headwind_signals") or [])
                    if str(code).strip()
                ]
                _score_row["primary_revenue_dependence_caution"] = str(
                    _revenue_row.get("primary_revenue_dependence_caution") or "REVENUE_BASE_UNCLEAR"
                )
                _score_row["revenue_fragility_summary"] = str(
                    _revenue_row.get("revenue_fragility_summary") or ""
                )
                _score_row["metric_values"]["revenue_dependence_risk_class"] = _revenue_class
                _score_row["inputs_used"]["customer_concentration_revenue_dependence_risk"] = {
                    "value": {
                        "revenue_dependence_risk_class": _revenue_class,
                        "primary_revenue_dependence_caution": _score_row["primary_revenue_dependence_caution"],
                    },
                    "derived_from": _revenue_refs,
                    "reason_codes": _revenue_reason_codes,
                }
                _score_row["derived_from"] = sorted(
                    set(
                        [
                            str(ref)
                            for ref in (_score_row.get("derived_from") or [])
                            if str(ref).strip()
                        ]
                        + _revenue_refs
                    )
                )
                _coverage_row = coverage_map.get(_ticker_key)
                if isinstance(_coverage_row, dict):
                    _coverage_row.update(
                        {
                            "revenue_dependence_risk_class": _revenue_class,
                            "revenue_dependence_risk_reason_codes": _revenue_reason_codes,
                            "revenue_dependence_support_signals": list(
                                _score_row.get("revenue_dependence_support_signals") or []
                            ),
                            "revenue_dependence_headwind_signals": list(
                                _score_row.get("revenue_dependence_headwind_signals") or []
                            ),
                            "primary_revenue_dependence_caution": _score_row["primary_revenue_dependence_caution"],
                            "revenue_fragility_summary": _score_row["revenue_fragility_summary"],
                            "derived_from": _score_row["derived_from"],
                        }
                    )
                _yield_row = yield_map.get(_ticker_key)
                if isinstance(_yield_row, dict):
                    _yield_row.update(
                        {
                            "revenue_dependence_risk_class": _revenue_class,
                            "revenue_dependence_risk_reason_codes": _revenue_reason_codes,
                            "revenue_dependence_support_signals": list(
                                _score_row.get("revenue_dependence_support_signals") or []
                            ),
                            "revenue_dependence_headwind_signals": list(
                                _score_row.get("revenue_dependence_headwind_signals") or []
                            ),
                            "primary_revenue_dependence_caution": _score_row["primary_revenue_dependence_caution"],
                            "revenue_fragility_summary": _score_row["revenue_fragility_summary"],
                            "derived_from": _score_row["derived_from"],
                        }
                    )
        returns_persistence_run_payload = write_returns_persistence_for_run(
            run_id=run_id,
            as_of_date=as_of_date,
            tickers=sorted(scoreboard_map.keys()),
            output_path=paths["returns_persistence_path"],
            scoreboard_rows=list(scoreboard_map.values()),
            cfg=cfg,
        )
        _returns_rows_by_ticker = {
            str(r.get("ticker") or "").upper(): r
            for r in (returns_persistence_run_payload.get("rows") or [])
            if isinstance(r, dict)
        }
        for _ticker_key, _score_row in scoreboard_map.items():
            _returns_row = _returns_rows_by_ticker.get(str(_ticker_key).upper())
            if isinstance(_returns_row, dict):
                _returns_refs = [
                    str(ref)
                    for ref in (_returns_row.get("derived_from") or [])
                    if str(ref).strip()
                ]
                _returns_class = str(
                    _returns_row.get("returns_persistence_class") or "RETURNS_PERSISTENCE_UNKNOWN"
                )
                _returns_reason_codes = [
                    str(code)
                    for code in (_returns_row.get("returns_persistence_reason_codes") or [])
                    if str(code).strip()
                ]
                _score_row["returns_persistence_detail"] = _returns_row
                _score_row["returns_persistence_class"] = _returns_class
                _score_row["returns_persistence_reason_codes"] = _returns_reason_codes
                _score_row["returns_support_signals"] = [
                    str(code)
                    for code in (_returns_row.get("returns_support_signals") or [])
                    if str(code).strip()
                ]
                _score_row["returns_headwind_signals"] = [
                    str(code)
                    for code in (_returns_row.get("returns_headwind_signals") or [])
                    if str(code).strip()
                ]
                _score_row["primary_returns_caution"] = str(
                    _returns_row.get("primary_returns_caution") or "RETURNS_DURABILITY_UNCLEAR"
                )
                _score_row["economic_durability_summary"] = str(
                    _returns_row.get("economic_durability_summary") or ""
                )
                _score_row["metric_values"]["returns_persistence_class"] = _returns_class
                _score_row["inputs_used"]["returns_persistence_economic_durability"] = {
                    "value": {
                        "returns_persistence_class": _returns_class,
                        "primary_returns_caution": _score_row["primary_returns_caution"],
                    },
                    "derived_from": _returns_refs,
                    "reason_codes": _returns_reason_codes,
                }
                _score_row["derived_from"] = sorted(
                    set(
                        [
                            str(ref)
                            for ref in (_score_row.get("derived_from") or [])
                            if str(ref).strip()
                        ]
                        + _returns_refs
                    )
                )
                _coverage_row = coverage_map.get(_ticker_key)
                if isinstance(_coverage_row, dict):
                    _coverage_row.update(
                        {
                            "returns_persistence_class": _returns_class,
                            "returns_persistence_reason_codes": _returns_reason_codes,
                            "returns_support_signals": list(
                                _score_row.get("returns_support_signals") or []
                            ),
                            "returns_headwind_signals": list(
                                _score_row.get("returns_headwind_signals") or []
                            ),
                            "primary_returns_caution": _score_row["primary_returns_caution"],
                            "economic_durability_summary": _score_row["economic_durability_summary"],
                            "derived_from": _score_row["derived_from"],
                        }
                    )
                _yield_row = yield_map.get(_ticker_key)
                if isinstance(_yield_row, dict):
                    _yield_row.update(
                        {
                            "returns_persistence_class": _returns_class,
                            "returns_persistence_reason_codes": _returns_reason_codes,
                            "returns_support_signals": list(
                                _score_row.get("returns_support_signals") or []
                            ),
                            "returns_headwind_signals": list(
                                _score_row.get("returns_headwind_signals") or []
                            ),
                            "primary_returns_caution": _score_row["primary_returns_caution"],
                            "economic_durability_summary": _score_row["economic_durability_summary"],
                            "derived_from": _score_row["derived_from"],
                        }
                    )
        investment_readiness_payload = write_investment_readiness_for_run(
            run_id=run_id,
            as_of_date=as_of_date,
            tickers=sorted(scoreboard_map.keys()),
            output_path=paths["investment_readiness_path"],
            scoreboard_rows=list(scoreboard_map.values()),
            cfg=cfg,
        )
        readiness_rows_by_ticker = {
            str(row.get("ticker") or "").upper(): row
            for row in (investment_readiness_payload.get("rows") or [])
            if isinstance(row, dict)
        }
        for ticker_key, score_row in scoreboard_map.items():
            readiness_row = readiness_rows_by_ticker.get(str(ticker_key).upper())
            if not isinstance(readiness_row, dict):
                continue
            readiness_refs = [
                str(ref)
                for ref in (readiness_row.get("derived_from") or [])
                if str(ref).strip()
            ]
            readiness_class = str(readiness_row.get("investment_readiness_class") or UNKNOWN)
            readiness_reason_codes = [
                str(code)
                for code in (readiness_row.get("investment_readiness_reason_codes") or [])
                if str(code).strip()
            ]
            blocker_stack_primary = (
                str(readiness_row.get("blocker_stack_primary"))
                if str(readiness_row.get("blocker_stack_primary") or "").strip()
                else UNKNOWN
            )
            blocker_stack_secondary = (
                str(readiness_row.get("blocker_stack_secondary"))
                if str(readiness_row.get("blocker_stack_secondary") or "").strip()
                else None
            )
            blocker_stack_all = [
                str(code)
                for code in (readiness_row.get("blocker_stack_all") or [])
                if str(code).strip()
            ]
            score_row["investment_readiness_detail"] = readiness_row
            score_row["investment_readiness_class"] = readiness_class
            score_row["investment_readiness_reason_codes"] = readiness_reason_codes
            score_row["blocker_stack_primary"] = blocker_stack_primary
            score_row["blocker_stack_secondary"] = blocker_stack_secondary
            score_row["blocker_stack_all"] = blocker_stack_all
            score_row["blocker_stack_retryable"] = bool(
                readiness_row.get("blocker_stack_retryable", False)
            )
            score_row["blocker_stack_structural"] = bool(
                readiness_row.get("blocker_stack_structural", False)
            )
            score_row["readiness_support_present"] = [
                str(code)
                for code in (readiness_row.get("readiness_support_present") or [])
                if str(code).strip()
            ]
            score_row["readiness_support_missing"] = [
                str(code)
                for code in (readiness_row.get("readiness_support_missing") or [])
                if str(code).strip()
            ]
            score_row["readiness_support_headwinds"] = [
                str(code)
                for code in (readiness_row.get("readiness_support_headwinds") or [])
                if str(code).strip()
            ]
            score_row["primary_next_step"] = str(readiness_row.get("primary_next_step") or UNKNOWN)
            score_row["primary_next_step_reason"] = str(
                readiness_row.get("primary_next_step_reason") or UNKNOWN
            )
            score_row["metric_values"]["investment_readiness_class"] = readiness_class
            score_row["inputs_used"]["investment_readiness"] = {
                "value": {
                    "investment_readiness_class": readiness_class,
                    "blocker_stack_primary": blocker_stack_primary,
                    "primary_next_step": score_row["primary_next_step"],
                },
                "derived_from": readiness_refs,
                "reason_codes": readiness_reason_codes,
            }
            score_row["derived_from"] = sorted(
                set(
                    [str(ref) for ref in (score_row.get("derived_from") or []) if str(ref).strip()]
                    + readiness_refs
                )
            )
            coverage_row = coverage_map.get(ticker_key)
            if isinstance(coverage_row, dict):
                coverage_row.update(
                    {
                        "investment_readiness_class": readiness_class,
                        "investment_readiness_reason_codes": readiness_reason_codes,
                        "blocker_stack_primary": blocker_stack_primary,
                        "blocker_stack_secondary": blocker_stack_secondary,
                        "blocker_stack_all": list(blocker_stack_all),
                        "blocker_stack_retryable": bool(
                            readiness_row.get("blocker_stack_retryable", False)
                        ),
                        "blocker_stack_structural": bool(
                            readiness_row.get("blocker_stack_structural", False)
                        ),
                        "readiness_support_present": list(
                            score_row.get("readiness_support_present") or []
                        ),
                        "readiness_support_missing": list(
                            score_row.get("readiness_support_missing") or []
                        ),
                        "readiness_support_headwinds": list(
                            score_row.get("readiness_support_headwinds") or []
                        ),
                        "primary_next_step": score_row["primary_next_step"],
                        "primary_next_step_reason": score_row["primary_next_step_reason"],
                        "derived_from": score_row["derived_from"],
                    }
                )
            yield_row = yield_map.get(ticker_key)
            if isinstance(yield_row, dict):
                yield_row.update(
                    {
                        "investment_readiness_class": readiness_class,
                        "investment_readiness_reason_codes": readiness_reason_codes,
                        "blocker_stack_primary": blocker_stack_primary,
                        "blocker_stack_secondary": blocker_stack_secondary,
                        "blocker_stack_all": list(blocker_stack_all),
                        "blocker_stack_retryable": bool(
                            readiness_row.get("blocker_stack_retryable", False)
                        ),
                        "blocker_stack_structural": bool(
                            readiness_row.get("blocker_stack_structural", False)
                        ),
                        "readiness_support_present": list(
                            score_row.get("readiness_support_present") or []
                        ),
                        "readiness_support_missing": list(
                            score_row.get("readiness_support_missing") or []
                        ),
                        "readiness_support_headwinds": list(
                            score_row.get("readiness_support_headwinds") or []
                        ),
                        "primary_next_step": score_row["primary_next_step"],
                        "primary_next_step_reason": score_row["primary_next_step_reason"],
                        "derived_from": score_row["derived_from"],
                    }
                )
        payloads = _build_scout_payloads(
            run_id=run_id,
            as_of_date=as_of_date,
            top_n=max(1, int(top_n)),
            with_prices=bool(with_prices),
            thresholds=thresholds,
            universe_meta=universe_meta,
            scoreboard_rows=list(scoreboard_map.values()),
            coverage_rows=list(coverage_map.values()),
            yield_rows=list(yield_map.values()),
            net_debt_payload=net_debt_payload,
            batch_progress=_batch_progress(),
            budget_progress=_budget_progress(),
            progress_state=progress_state,
            run_status=status,
            stop_reason_code=stop_code,
            stop_summary=stop_text,
        )
        gd_summary = _write_graham_dodd_artifacts(
            run_id=run_id,
            as_of_date=as_of_date,
            scoreboard_rows=list(scoreboard_map.values()),
            summary_path=paths["graham_dodd_summary_path"],
            universe_dir=paths["universe_dir"],
        )
        payloads["summary"]["graham_dodd_summary_path"] = str(paths["graham_dodd_summary_path"])
        payloads["summary"]["graham_dodd"] = {
            **(payloads["summary"].get("graham_dodd") if isinstance(payloads["summary"].get("graham_dodd"), dict) else {}),
            "gd_known_count": int(gd_summary.get("gd_known_count", 0)),
            "gd_unknown_count": int(gd_summary.get("gd_unknown_count", 0)),
            "reason_counts": gd_summary.get("reason_counts") if isinstance(gd_summary.get("reason_counts"), dict) else {},
        }
        payloads["summary"]["owner_earnings_quality"] = {
            "known_count": int(owner_earnings_quality_payload.get("known_count") or 0),
            "unknown_count": int(owner_earnings_quality_payload.get("unknown_count") or 0),
            "negative_reason_counts": owner_earnings_quality_payload.get("negative_reason_counts")
            if isinstance(owner_earnings_quality_payload.get("negative_reason_counts"), dict)
            else {},
        }
        payloads["summary"]["owner_earnings_quality_path"] = str(paths["owner_earnings_quality_path"])
        payloads["summary"]["maintenance_capex_discipline"] = {
            "counts_by_maintenance_capex_credibility_class": (
                maintenance_capex_run_payload.get("counts_by_maintenance_capex_credibility_class")
                if isinstance(maintenance_capex_run_payload.get("counts_by_maintenance_capex_credibility_class"), dict)
                else {}
            ),
            "counts_by_asset_intensity_class": (
                maintenance_capex_run_payload.get("counts_by_asset_intensity_class")
                if isinstance(maintenance_capex_run_payload.get("counts_by_asset_intensity_class"), dict)
                else {}
            ),
            "counts_by_primary_maintenance_capex_caution": (
                maintenance_capex_run_payload.get("counts_by_primary_maintenance_capex_caution")
                if isinstance(maintenance_capex_run_payload.get("counts_by_primary_maintenance_capex_caution"), dict)
                else {}
            ),
            "most_common_maintenance_capex_reason_codes": (
                maintenance_capex_run_payload.get("most_common_maintenance_capex_reason_codes")
                if isinstance(maintenance_capex_run_payload.get("most_common_maintenance_capex_reason_codes"), list)
                else []
            ),
        }
        payloads["summary"]["maintenance_capex_discipline_path"] = str(
            paths["maintenance_capex_discipline_path"]
        )
        payloads["summary"]["accounting_quality"] = {
            "counts_by_accounting_quality_class": (
                accounting_quality_payload.get("counts_by_accounting_quality_class")
                if isinstance(accounting_quality_payload.get("counts_by_accounting_quality_class"), dict)
                else {}
            ),
            "counts_by_primary_accounting_caution": (
                accounting_quality_payload.get("counts_by_primary_accounting_caution")
                if isinstance(accounting_quality_payload.get("counts_by_primary_accounting_caution"), dict)
                else {}
            ),
            "most_common_accounting_quality_reason_codes": (
                accounting_quality_payload.get("most_common_accounting_quality_reason_codes")
                if isinstance(accounting_quality_payload.get("most_common_accounting_quality_reason_codes"), list)
                else []
            ),
        }
        payloads["summary"]["accounting_quality_path"] = str(paths["accounting_quality_path"])
        payloads["summary"]["balance_sheet_stress"] = {
            "counts_by_balance_sheet_stress_class": (
                balance_sheet_stress_run_payload.get("counts_by_balance_sheet_stress_class")
                if isinstance(balance_sheet_stress_run_payload.get("counts_by_balance_sheet_stress_class"), dict)
                else {}
            ),
            "counts_by_refinancing_risk_class": (
                balance_sheet_stress_run_payload.get("counts_by_refinancing_risk_class")
                if isinstance(balance_sheet_stress_run_payload.get("counts_by_refinancing_risk_class"), dict)
                else {}
            ),
            "counts_by_primary_balance_sheet_caution": (
                balance_sheet_stress_run_payload.get("counts_by_primary_balance_sheet_caution")
                if isinstance(balance_sheet_stress_run_payload.get("counts_by_primary_balance_sheet_caution"), dict)
                else {}
            ),
            "most_common_balance_sheet_stress_reason_codes": (
                balance_sheet_stress_run_payload.get("most_common_balance_sheet_stress_reason_codes")
                if isinstance(balance_sheet_stress_run_payload.get("most_common_balance_sheet_stress_reason_codes"), list)
                else []
            ),
        }
        payloads["summary"]["balance_sheet_stress_path"] = str(paths["balance_sheet_stress_path"])
        payloads["summary"]["returns_persistence"] = {
            "counts_by_returns_persistence_class": (
                returns_persistence_run_payload.get("counts_by_returns_persistence_class")
                if isinstance(returns_persistence_run_payload.get("counts_by_returns_persistence_class"), dict)
                else {}
            ),
            "counts_by_primary_returns_caution": (
                returns_persistence_run_payload.get("counts_by_primary_returns_caution")
                if isinstance(returns_persistence_run_payload.get("counts_by_primary_returns_caution"), dict)
                else {}
            ),
            "most_common_returns_persistence_reason_codes": (
                returns_persistence_run_payload.get("most_common_returns_persistence_reason_codes")
                if isinstance(returns_persistence_run_payload.get("most_common_returns_persistence_reason_codes"), list)
                else []
            ),
        }
        payloads["summary"]["returns_persistence_path"] = str(paths["returns_persistence_path"])
        payloads["summary"]["revenue_dependence"] = {
            "counts_by_revenue_dependence_risk_class": (
                revenue_dependence_run_payload.get("counts_by_revenue_dependence_risk_class")
                if isinstance(revenue_dependence_run_payload.get("counts_by_revenue_dependence_risk_class"), dict)
                else {}
            ),
            "counts_by_primary_revenue_dependence_caution": (
                revenue_dependence_run_payload.get("counts_by_primary_revenue_dependence_caution")
                if isinstance(revenue_dependence_run_payload.get("counts_by_primary_revenue_dependence_caution"), dict)
                else {}
            ),
            "most_common_revenue_dependence_reason_codes": (
                revenue_dependence_run_payload.get("most_common_revenue_dependence_reason_codes")
                if isinstance(revenue_dependence_run_payload.get("most_common_revenue_dependence_reason_codes"), list)
                else []
            ),
        }
        payloads["summary"]["revenue_dependence_path"] = str(paths["revenue_dependence_path"])
        payloads["summary"]["intangible_economics"] = {
            "known_count": int(intangible_economics_payload.get("known_count") or 0),
            "unknown_count": int(intangible_economics_payload.get("unknown_count") or 0),
            "negative_reason_counts": intangible_economics_payload.get("negative_reason_counts")
            if isinstance(intangible_economics_payload.get("negative_reason_counts"), dict)
            else {},
        }
        payloads["summary"]["intangible_economics_path"] = str(paths["intangible_economics_path"])
        payloads["summary"]["reinvestment_efficiency"] = {
            "counts_by_reinvestment_efficiency_class": (
                reinvestment_efficiency_payload.get("counts_by_reinvestment_efficiency_class")
                if isinstance(reinvestment_efficiency_payload.get("counts_by_reinvestment_efficiency_class"), dict)
                else {}
            ),
            "counts_by_primary_reinvestment_caution": (
                reinvestment_efficiency_payload.get("counts_by_primary_reinvestment_caution")
                if isinstance(reinvestment_efficiency_payload.get("counts_by_primary_reinvestment_caution"), dict)
                else {}
            ),
            "most_common_reinvestment_reason_codes": (
                reinvestment_efficiency_payload.get("most_common_reinvestment_reason_codes")
                if isinstance(reinvestment_efficiency_payload.get("most_common_reinvestment_reason_codes"), list)
                else []
            ),
        }
        payloads["summary"]["reinvestment_efficiency_path"] = str(paths["reinvestment_efficiency_path"])
        payloads["summary"]["intrinsic_discipline"] = {
            "known_count": int(intrinsic_discipline_payload.get("known_count") or 0),
            "unknown_count": int(intrinsic_discipline_payload.get("unknown_count") or 0),
            "negative_reason_counts": intrinsic_discipline_payload.get("negative_reason_counts")
            if isinstance(intrinsic_discipline_payload.get("negative_reason_counts"), dict)
            else {},
            "limited_support_count": int(intrinsic_discipline_payload.get("limited_support_count") or 0),
            "unknown_support_count": int(intrinsic_discipline_payload.get("unknown_support_count") or 0),
        }
        payloads["summary"]["intrinsic_discipline_path"] = str(paths["intrinsic_discipline_path"])
        payloads["summary"]["evidence_sufficiency"] = {
            "counts_by_evidence_sufficiency_class": evidence_sufficiency_payload.get("counts_by_evidence_sufficiency_class")
            if isinstance(evidence_sufficiency_payload.get("counts_by_evidence_sufficiency_class"), dict)
            else {},
            "counts_by_mos_assessment_status": evidence_sufficiency_payload.get("counts_by_mos_assessment_status")
            if isinstance(evidence_sufficiency_payload.get("counts_by_mos_assessment_status"), dict)
            else {},
            "sufficiency_reason_counts": evidence_sufficiency_payload.get("sufficiency_reason_counts")
            if isinstance(evidence_sufficiency_payload.get("sufficiency_reason_counts"), dict)
            else {},
        }
        payloads["summary"]["evidence_sufficiency_path"] = str(paths["evidence_sufficiency_path"])
        payloads["summary"]["valuation_confidence"] = {
            "known_count": int(valuation_confidence_payload.get("known_count") or 0),
            "unknown_count": int(valuation_confidence_payload.get("unknown_count") or 0),
            "high_confidence_count": int(valuation_confidence_payload.get("high_confidence_count") or 0),
            "high_fragility_count": int(valuation_confidence_payload.get("high_fragility_count") or 0),
            "fragility_reason_counts": valuation_confidence_payload.get("fragility_reason_counts")
            if isinstance(valuation_confidence_payload.get("fragility_reason_counts"), dict)
            else {},
            "single_support_only_count": int(valuation_confidence_payload.get("single_support_only_count") or 0),
        }
        payloads["summary"]["valuation_confidence_path"] = str(paths["valuation_confidence_path"])
        payloads["summary"]["valuation_integrity"] = {
            "counts_by_integrity_class": valuation_integrity_payload.get("counts_by_integrity_class")
            if isinstance(valuation_integrity_payload.get("counts_by_integrity_class"), dict)
            else {},
            "integrity_reason_counts": valuation_integrity_payload.get("integrity_reason_counts")
            if isinstance(valuation_integrity_payload.get("integrity_reason_counts"), dict)
            else {},
            "exact_uniformity_cluster_count": int(
                valuation_integrity_payload.get("exact_uniformity_cluster_count") or 0
            ),
        }
        payloads["summary"]["valuation_integrity_path"] = str(paths["valuation_integrity_path"])
        payloads["summary"]["fundamental_regression_analytics"] = {
            "counts_by_shares_source_class": fundamental_regression_payload.get("counts_by_shares_source_class")
            if isinstance(fundamental_regression_payload.get("counts_by_shares_source_class"), dict)
            else {},
            "counts_by_fcf_source_class": fundamental_regression_payload.get("counts_by_fcf_source_class")
            if isinstance(fundamental_regression_payload.get("counts_by_fcf_source_class"), dict)
            else {},
            "counts_by_total_debt_source_class": fundamental_regression_payload.get("counts_by_total_debt_source_class")
            if isinstance(fundamental_regression_payload.get("counts_by_total_debt_source_class"), dict)
            else {},
            "unknown_input_counts": fundamental_regression_payload.get("unknown_input_counts")
            if isinstance(fundamental_regression_payload.get("unknown_input_counts"), dict)
            else {},
            "formula_mismatch_counts": fundamental_regression_payload.get("formula_mismatch_counts")
            if isinstance(fundamental_regression_payload.get("formula_mismatch_counts"), dict)
            else {},
            "threshold_breaches": [
                row for row in (fundamental_regression_payload.get("threshold_breaches") or []) if isinstance(row, dict)
            ],
        }
        payloads["summary"]["fundamental_regression_analytics_path"] = str(paths["fundamental_regression_analytics_path"])
        payloads["summary"]["value_type"] = {
            "counts_by_primary_value_type": value_type_payload.get("counts_by_primary_value_type")
            if isinstance(value_type_payload.get("counts_by_primary_value_type"), dict)
            else {},
            "fragile_value_count": int(value_type_payload.get("fragile_value_count") or 0),
            "unknown_value_type_count": int(value_type_payload.get("unknown_value_type_count") or 0),
            "value_type_reason_counts": value_type_payload.get("value_type_reason_counts")
            if isinstance(value_type_payload.get("value_type_reason_counts"), dict)
            else {},
        }
        payloads["summary"]["value_type_path"] = str(paths["value_type_path"])
        payloads["summary"]["investment_readiness"] = {
            "counts_by_readiness_class": investment_readiness_payload.get("counts_by_readiness_class")
            if isinstance(investment_readiness_payload.get("counts_by_readiness_class"), dict)
            else {},
            "primary_blocker_counts": investment_readiness_payload.get("primary_blocker_counts")
            if isinstance(investment_readiness_payload.get("primary_blocker_counts"), dict)
            else {},
            "retryable_blocker_count": int(
                investment_readiness_payload.get("retryable_blocker_count") or 0
            ),
            "structural_blocker_count": int(
                investment_readiness_payload.get("structural_blocker_count") or 0
            ),
            "primary_next_step_counts": investment_readiness_payload.get("primary_next_step_counts")
            if isinstance(investment_readiness_payload.get("primary_next_step_counts"), dict)
            else {},
        }
        payloads["summary"]["investment_readiness_path"] = str(paths["investment_readiness_path"])
        payloads["summary"]["cyclical_normalization"] = {
            "counts_by_cyclical_profile_class": cyclical_normalization_run_payload.get("counts_by_cyclical_profile_class")
            if isinstance(cyclical_normalization_run_payload.get("counts_by_cyclical_profile_class"), dict)
            else {},
            "counts_by_cyclical_valuation_risk_class": cyclical_normalization_run_payload.get("counts_by_cyclical_valuation_risk_class")
            if isinstance(cyclical_normalization_run_payload.get("counts_by_cyclical_valuation_risk_class"), dict)
            else {},
        }
        payloads["summary"]["cyclical_normalization_path"] = str(paths["cyclical_normalization_path"])
        payloads["summary"]["impairment_classification"] = {
            "counts_by_impairment_class": impairment_classification_run_payload.get("counts_by_impairment_class")
            if isinstance(impairment_classification_run_payload.get("counts_by_impairment_class"), dict)
            else {},
        }
        payloads["summary"]["impairment_classification_path"] = str(paths["impairment_classification_path"])
        payloads["summary"]["normalization_credibility"] = {
            "counts_by_normalization_credibility_class": normalization_credibility_run_payload.get("counts_by_normalization_credibility_class")
            if isinstance(normalization_credibility_run_payload.get("counts_by_normalization_credibility_class"), dict)
            else {},
        }
        payloads["summary"]["normalization_credibility_path"] = str(paths["normalization_credibility_path"])
        payloads["summary"]["capital_allocation_discipline"] = {
            "counts_by_capital_allocation_discipline_class": capital_allocation_discipline_run_payload.get("counts_by_capital_allocation_discipline_class")
            if isinstance(capital_allocation_discipline_run_payload.get("counts_by_capital_allocation_discipline_class"), dict)
            else {},
        }
        payloads["summary"]["capital_allocation_discipline_path"] = str(paths["capital_allocation_discipline_path"])
        payloads["summary"]["universe_rankings_path"] = str(paths["rankings_path"])
        payloads["summary"]["depth_queue_path"] = str(paths["depth_queue_path"])
        _json_write(paths["summary_path"], payloads["summary"])
        _json_write(paths["scoreboard_path"], payloads["scoreboard"])
        _json_write(paths["shortlist_path"], payloads["shortlist"])
        _json_write(paths["coverage_path"], payloads["coverage"])
        _json_write(paths["calibration_path"], payloads["calibration"])
        _json_write(paths["yield_path"], payloads["yield"])
        _json_write(paths["rankings_path"], payloads["rankings"])
        _json_write(paths["depth_queue_path"], payloads["depth_queue"])
        _json_write(paths["maintenance_capex_discipline_path"], maintenance_capex_run_payload)
        _json_write(paths["accounting_quality_path"], accounting_quality_payload)
        _json_write(paths["balance_sheet_stress_path"], balance_sheet_stress_run_payload)
        _json_write(paths["returns_persistence_path"], returns_persistence_run_payload)
        _json_write(paths["revenue_dependence_path"], revenue_dependence_run_payload)
        _json_write(paths["reinvestment_efficiency_path"], reinvestment_efficiency_payload)
        _json_write(paths["intrinsic_discipline_path"], intrinsic_discipline_payload)
        _json_write(paths["evidence_sufficiency_path"], evidence_sufficiency_payload)
        _json_write(paths["valuation_confidence_path"], valuation_confidence_payload)
        _json_write(paths["valuation_integrity_path"], valuation_integrity_payload)
        _json_write(paths["value_type_path"], value_type_payload)
        _json_write(paths["investment_readiness_path"], investment_readiness_payload)
        _json_write(paths["cyclical_normalization_path"], cyclical_normalization_run_payload)
        _json_write(paths["impairment_classification_path"], impairment_classification_run_payload)
        _json_write(paths["normalization_credibility_path"], normalization_credibility_run_payload)
        _json_write(paths["capital_allocation_discipline_path"], capital_allocation_discipline_run_payload)
        return payloads["summary"]

    _mark_progress(
        PHASE_BATCH_PREP,
        hydration_status=HYDRATION_STATUS_RUNNING,
        price_stage_completed=False,
        facts_stage_completed=False,
        scoring_stage_completed=False,
        finalization_started=False,
    )
    _write_state(SCOUT_STATE_RUNNING, stop_code=STOP_NONE, stop_text="")

    for batch_index, batch_tickers in enumerate(batches):
        if batch_index in done_batches:
            continue
        if max_batches is not None and int(max_batches) > 0 and invocation_batches >= int(max_batches):
            run_status = SCOUT_STATE_PARTIAL
            stop_reason_code = STOP_MAX_BATCHES_REACHED
            stop_summary = f"Stopped after max_batches={int(max_batches)} in this invocation."
            break

        latest_state = _safe_json(paths["state_path"])
        if str(latest_state.get("status") or "").upper() == SCOUT_STATE_CANCELLED:
            run_status = SCOUT_STATE_CANCELLED
            stop_reason_code = STOP_CANCELLED
            stop_summary = str(latest_state.get("stop_summary") or "Cancelled by user.")
            break

        if max_seconds_remaining is not None and int(max_seconds_remaining) <= 0:
            run_status = SCOUT_STATE_PARTIAL
            stop_reason_code = STOP_BUDGET_EXHAUSTED
            stop_summary = "Scout max-seconds budget exhausted before next batch."
            break

        batch_started = utc_now_iso()
        _mark_progress(
            PHASE_BATCH_PREP,
            batch_index=int(batch_index),
            batch_tickers=list(batch_tickers),
            current_ticker="",
            hydration_status=HYDRATION_STATUS_RUNNING,
            price_stage_completed=False,
            facts_stage_completed=False,
            scoring_stage_completed=False,
            finalization_started=False,
            stalled_reason_code="",
        )
        _write_state(SCOUT_STATE_RUNNING, stop_code=stop_reason_code, stop_text=stop_summary)

        price_rows_by_ticker: dict[str, dict[str, Any]] = {}
        _mark_progress(
            PHASE_PRICE_COLLECTION,
            batch_index=int(batch_index),
            batch_tickers=list(batch_tickers),
            current_ticker="",
            price_stage_completed=False,
            facts_stage_completed=False,
            scoring_stage_completed=False,
        )
        _write_state(SCOUT_STATE_RUNNING, stop_code=stop_reason_code, stop_text=stop_summary)
        if bool(with_prices):
            if net_budget_remaining is not None and int(net_budget_remaining) <= 0:
                run_status = SCOUT_STATE_PARTIAL
                stop_reason_code = STOP_BUDGET_EXHAUSTED
                stop_summary = "Scout network budget exhausted before batch."
                _mark_progress(
                    PHASE_PRICE_COLLECTION,
                    batch_index=int(batch_index),
                    batch_tickers=list(batch_tickers),
                    hydration_status=HYDRATION_STATUS_DEGRADED,
                    primary_blocker=STOP_BUDGET_EXHAUSTED,
                    stalled_reason_code=STOP_BUDGET_EXHAUSTED,
                )
                remaining = list(batch_tickers)
                for ticker in remaining:
                    score_row, cov_row, yld_row, nd_row = _build_budget_skipped_rows(
                        ticker=ticker,
                        reason_code=STOP_BUDGET_EXHAUSTED,
                    )
                    score_row, cov_row, yld_row = _apply_facts_blocker_policy(score_row, cov_row, yld_row)
                    scoreboard_map[ticker] = score_row
                    coverage_map[ticker] = cov_row
                    yield_map[ticker] = yld_row
                    nd_row["as_of_date"] = as_of_date
                    net_debt_resolved_map[ticker] = nd_row
                batch_payload = {
                    "run_id": run_id,
                    "batch_index": int(batch_index),
                    "tickers": list(batch_tickers),
                    "started_at": batch_started,
                    "finished_at": utc_now_iso(),
                    "status": "SKIPPED_BUDGET",
                    "stop_reason_code": STOP_BUDGET_EXHAUSTED,
                    "counts": _batch_counts(
                        scoreboard_rows=[scoreboard_map[ticker] for ticker in batch_tickers if ticker in scoreboard_map],
                        coverage_rows=[coverage_map[ticker] for ticker in batch_tickers if ticker in coverage_map],
                        yield_rows=[yield_map[ticker] for ticker in batch_tickers if ticker in yield_map],
                    ),
                }
                _json_write(paths["batch_dir"] / f"batch_{int(batch_index):04d}.json", batch_payload)
                _write_outputs(SCOUT_STATE_PARTIAL, stop_code=stop_reason_code, stop_text=stop_summary)
                _write_state(SCOUT_STATE_PARTIAL, stop_code=stop_reason_code, stop_text=stop_summary)
                break
            allowed_network = len(batch_tickers)
            if net_budget_remaining is not None:
                allowed_network = min(allowed_network, int(net_budget_remaining))
            network_tickers = batch_tickers[:allowed_network]
            local_only_tickers = batch_tickers[allowed_network:]
            if network_tickers:
                price_payload = write_prices_for_run(
                    tickers=network_tickers,
                    as_of_date=as_of_date,
                    run_id=run_id,
                    fallback_days=max(0, int(cfg.price_fallback_days)),
                    local_only=False,
                    cfg=cfg,
                )
                for row in (price_payload.get("rows") or []):
                    if isinstance(row, dict):
                        ticker_norm = str(row.get("ticker") or "").upper()
                        if ticker_norm:
                            price_rows_by_ticker[ticker_norm] = row
                if net_budget_remaining is not None:
                    net_budget_remaining = max(0, int(net_budget_remaining) - len(network_tickers))
            if local_only_tickers:
                price_payload_local = write_prices_for_run(
                    tickers=local_only_tickers,
                    as_of_date=as_of_date,
                    run_id=run_id,
                    fallback_days=max(0, int(cfg.price_fallback_days)),
                    local_only=True,
                    cfg=cfg,
                )
                for row in (price_payload_local.get("rows") or []):
                    if isinstance(row, dict):
                        ticker_norm = str(row.get("ticker") or "").upper()
                        if ticker_norm:
                            price_rows_by_ticker[ticker_norm] = row
        else:
            price_payload = write_prices_for_run(
                tickers=batch_tickers,
                as_of_date=as_of_date,
                run_id=run_id,
                fallback_days=max(0, int(cfg.price_fallback_days)),
                local_only=True,
                cfg=cfg,
            )
            for row in (price_payload.get("rows") or []):
                if isinstance(row, dict):
                    ticker_norm = str(row.get("ticker") or "").upper()
                    if ticker_norm:
                        price_rows_by_ticker[ticker_norm] = row
        _mark_progress(
            PHASE_PRICE_COLLECTION,
            batch_index=int(batch_index),
            batch_tickers=list(batch_tickers),
            current_ticker="",
            price_stage_completed=True,
        )
        _write_state(SCOUT_STATE_RUNNING, stop_code=stop_reason_code, stop_text=stop_summary)

        batch_status = "DONE"
        processed_count = 0
        for idx, ticker in enumerate(batch_tickers):
            elapsed = int(max(0, time.monotonic() - start_monotonic))
            if max_seconds_remaining is not None:
                max_seconds_remaining = max(0, int(max_seconds_effective or 0) - elapsed)
            if max_seconds_remaining is not None and int(max_seconds_remaining) <= 0:
                batch_status = "PARTIAL" if processed_count > 0 else "SKIPPED_BUDGET"
                run_status = SCOUT_STATE_PARTIAL
                stop_reason_code = STOP_BUDGET_EXHAUSTED
                stop_summary = "Scout max-seconds budget exhausted during batch."
                for rem in batch_tickers[idx:]:
                    score_row, cov_row, yld_row, nd_row = _build_budget_skipped_rows(
                        ticker=rem,
                        reason_code=STOP_BUDGET_EXHAUSTED,
                    )
                    score_row, cov_row, yld_row = _apply_facts_blocker_policy(score_row, cov_row, yld_row)
                    scoreboard_map[rem] = score_row
                    coverage_map[rem] = cov_row
                    yield_map[rem] = yld_row
                    nd_row["as_of_date"] = as_of_date
                    net_debt_resolved_map[rem] = nd_row
                break
            if sec_budget_remaining is not None and int(sec_budget_remaining) <= 0:
                batch_status = "PARTIAL" if processed_count > 0 else "SKIPPED_BUDGET"
                run_status = SCOUT_STATE_PARTIAL
                stop_reason_code = STOP_BUDGET_EXHAUSTED
                stop_summary = "Scout SEC budget exhausted during batch."
                for rem in batch_tickers[idx:]:
                    score_row, cov_row, yld_row, nd_row = _build_budget_skipped_rows(
                        ticker=rem,
                        reason_code=STOP_BUDGET_EXHAUSTED,
                    )
                    score_row, cov_row, yld_row = _apply_facts_blocker_policy(score_row, cov_row, yld_row)
                    scoreboard_map[rem] = score_row
                    coverage_map[rem] = cov_row
                    yield_map[rem] = yld_row
                    nd_row["as_of_date"] = as_of_date
                    net_debt_resolved_map[rem] = nd_row
                break

            sec_budget_for_call = sec_budget_remaining if sec_budget_remaining is not None else None
            phase_timeout_seconds = _scout_phase_timeout_seconds(cfg=cfg, max_seconds_remaining=max_seconds_remaining)
            _mark_progress(
                PHASE_COMPANYFACTS_ACQUISITION,
                batch_index=int(batch_index),
                batch_tickers=list(batch_tickers),
                current_ticker=ticker,
                facts_stage_completed=False,
                scoring_stage_completed=False,
            )
            _write_state(SCOUT_STATE_RUNNING, stop_code=stop_reason_code, stop_text=stop_summary)

            def _facts_progress(phase_name: str, _payload: dict[str, Any]) -> None:
                mapped_phase = PHASE_COMPANYFACTS_ACQUISITION
                if str(phase_name).upper() in {"FACTS_NORMALIZATION", "FACTS_READY"}:
                    mapped_phase = PHASE_FACTS_NORMALIZATION
                _mark_progress(
                    mapped_phase,
                    batch_index=int(batch_index),
                    batch_tickers=list(batch_tickers),
                    current_ticker=ticker,
                    facts_stage_completed=str(phase_name).upper() == "FACTS_READY",
                )
                _write_state(SCOUT_STATE_RUNNING, stop_code=stop_reason_code, stop_text=stop_summary)

            try:
                facts_row = _call_with_timeout(
                    phase=PHASE_COMPANYFACTS_ACQUISITION,
                    ticker_label=str(ticker),
                    timeout_seconds=phase_timeout_seconds,
                    fn=resolve_financial_facts_asof,
                    ticker=ticker,
                    as_of_date=as_of_date,
                    run_id=run_id,
                    refresh=False,
                    sec_budget=sec_budget_for_call,
                    progress_hook=_facts_progress,
                    cfg=cfg,
                )
            except ScoutPhaseTimeoutError:
                facts_row = _facts_timeout_row(
                    ticker=ticker,
                    as_of_date=as_of_date,
                    run_id=run_id,
                    reason_code=REASON_SCOUT_FACTS_TIMEOUT,
                    phase=PHASE_COMPANYFACTS_ACQUISITION,
                    timeout_seconds=float(phase_timeout_seconds or 0.0),
                )
                _mark_progress(
                    PHASE_COMPANYFACTS_ACQUISITION,
                    batch_index=int(batch_index),
                    batch_tickers=list(batch_tickers),
                    current_ticker=ticker,
                    facts_stage_completed=True,
                    hydration_status=HYDRATION_STATUS_DEGRADED,
                    primary_blocker=REASON_SCOUT_FACTS_TIMEOUT,
                    stalled_reason_code=REASON_SCOUT_FACTS_TIMEOUT,
                )
                _write_state(SCOUT_STATE_RUNNING, stop_code=stop_reason_code, stop_text=stop_summary)
            _record_facts_result(facts_row)
            if sec_budget_remaining is not None and bool(facts_row.get("network_attempted")) and int(sec_budget_remaining) > 0:
                sec_budget_remaining = max(0, int(sec_budget_remaining) - 1)
            _mark_progress(
                PHASE_NET_DEBT_RESOLUTION,
                batch_index=int(batch_index),
                batch_tickers=list(batch_tickers),
                current_ticker=ticker,
                facts_stage_completed=True,
                scoring_stage_completed=False,
            )
            _write_state(SCOUT_STATE_RUNNING, stop_code=stop_reason_code, stop_text=stop_summary)
            try:
                net_debt_resolved = _call_with_timeout(
                    phase=PHASE_NET_DEBT_RESOLUTION,
                    ticker_label=str(ticker),
                    timeout_seconds=phase_timeout_seconds,
                    fn=resolve_net_debt_proxy,
                    ticker=ticker,
                    as_of_date=as_of_date,
                    run_id=run_id,
                    facts_row=facts_row,
                    cfg=cfg,
                )
            except ScoutPhaseTimeoutError:
                net_debt_resolved = _net_debt_timeout_row(
                    ticker=ticker,
                    as_of_date=as_of_date,
                    reason_code=REASON_SCOUT_NET_DEBT_TIMEOUT,
                    phase=PHASE_NET_DEBT_RESOLUTION,
                    timeout_seconds=float(phase_timeout_seconds or 0.0),
                )
                _mark_progress(
                    PHASE_NET_DEBT_RESOLUTION,
                    batch_index=int(batch_index),
                    batch_tickers=list(batch_tickers),
                    current_ticker=ticker,
                    hydration_status=HYDRATION_STATUS_DEGRADED,
                    primary_blocker=REASON_SCOUT_NET_DEBT_TIMEOUT,
                    stalled_reason_code=REASON_SCOUT_NET_DEBT_TIMEOUT,
                )
                _write_state(SCOUT_STATE_RUNNING, stop_code=stop_reason_code, stop_text=stop_summary)

            _mark_progress(
                PHASE_SCOUT_SCORING,
                batch_index=int(batch_index),
                batch_tickers=list(batch_tickers),
                current_ticker=ticker,
                facts_stage_completed=True,
                scoring_stage_completed=False,
            )
            _write_state(SCOUT_STATE_RUNNING, stop_code=stop_reason_code, stop_text=stop_summary)
            try:
                score_row, cov_row, yld_row = _call_with_timeout(
                    phase=PHASE_SCOUT_SCORING,
                    ticker_label=str(ticker),
                    timeout_seconds=phase_timeout_seconds,
                    fn=_build_scout_record,
                    ticker=ticker,
                    as_of_date=as_of_date,
                    price_row=price_rows_by_ticker.get(ticker, {}),
                    facts_row=facts_row,
                    net_debt_resolved=net_debt_resolved,
                    thresholds=thresholds,
                    require_ev_yield=require_ev_yield,
                )
                score_row, cov_row, yld_row = _apply_facts_blocker_policy(score_row, cov_row, yld_row)
            except ScoutPhaseTimeoutError:
                batch_status = "PARTIAL" if processed_count > 0 else "FAILED"
                run_status = SCOUT_STATE_PARTIAL
                stop_reason_code = STOP_STALE_SCOUT
                stop_summary = (
                    f"Scout stalled during {PHASE_SCOUT_SCORING} for ticker={ticker} "
                    f"after {float(phase_timeout_seconds or 0.0):.1f}s."
                )
                _mark_progress(
                    PHASE_SCOUT_SCORING,
                    batch_index=int(batch_index),
                    batch_tickers=list(batch_tickers),
                    current_ticker=ticker,
                    hydration_status=HYDRATION_STATUS_STALE,
                    primary_blocker=REASON_SCOUT_SCORING_TIMEOUT,
                    stalled_reason_code=REASON_SCOUT_SCORING_TIMEOUT,
                )
                _write_state(SCOUT_STATE_PARTIAL, stop_code=stop_reason_code, stop_text=stop_summary)
                break
            scoreboard_map[ticker] = score_row
            coverage_map[ticker] = cov_row
            yield_map[ticker] = yld_row
            net_debt_resolved_map[ticker] = net_debt_resolved
            processed_count += 1
            _mark_progress(
                PHASE_SCOUT_SCORING,
                batch_index=int(batch_index),
                batch_tickers=list(batch_tickers),
                current_ticker=ticker,
                last_completed_ticker=ticker,
                facts_stage_completed=True,
                scoring_stage_completed=True,
            )
            _write_state(SCOUT_STATE_RUNNING, stop_code=stop_reason_code, stop_text=stop_summary)

        _mark_progress(
            PHASE_BATCH_FINALIZATION,
            batch_index=int(batch_index),
            batch_tickers=list(batch_tickers),
            current_ticker="",
            finalization_started=True,
        )
        _write_state(run_status if run_status in {SCOUT_STATE_PARTIAL, SCOUT_STATE_CANCELLED} else SCOUT_STATE_RUNNING, stop_code=stop_reason_code, stop_text=stop_summary)

        batch_payload = {
            "run_id": run_id,
            "batch_index": int(batch_index),
            "tickers": list(batch_tickers),
            "started_at": batch_started,
            "finished_at": utc_now_iso(),
            "status": batch_status,
            "stop_reason_code": stop_reason_code if batch_status != "DONE" else STOP_NONE,
            "counts": _batch_counts(
                scoreboard_rows=[scoreboard_map[ticker] for ticker in batch_tickers if ticker in scoreboard_map],
                coverage_rows=[coverage_map[ticker] for ticker in batch_tickers if ticker in coverage_map],
                yield_rows=[yield_map[ticker] for ticker in batch_tickers if ticker in yield_map],
            ),
        }
        _json_write(paths["batch_dir"] / f"batch_{int(batch_index):04d}.json", batch_payload)

        if batch_status == "DONE":
            done_batches.add(int(batch_index))
            last_completed_batch = max(done_batches) if done_batches else -1
        invocation_batches += 1

        interim_status = run_status if run_status in {SCOUT_STATE_PARTIAL, SCOUT_STATE_CANCELLED} else SCOUT_STATE_RUNNING
        _mark_progress(
            PHASE_OUTPUT_FINALIZATION,
            batch_index=int(batch_index),
            batch_tickers=list(batch_tickers),
            current_ticker="",
            finalization_started=True,
        )
        _write_outputs(interim_status, stop_code=stop_reason_code, stop_text=stop_summary)
        _write_state(interim_status, stop_code=stop_reason_code, stop_text=stop_summary)

        if run_status in {SCOUT_STATE_PARTIAL, SCOUT_STATE_CANCELLED}:
            break

    if run_status == SCOUT_STATE_RUNNING:
        if len(done_batches) >= total_batches:
            run_status = SCOUT_STATE_DONE
            stop_reason_code = STOP_NONE
            stop_summary = ""
        else:
            run_status = SCOUT_STATE_PARTIAL
            if stop_reason_code == STOP_NONE:
                stop_reason_code = STOP_MAX_BATCHES_REACHED if max_batches is not None else STOP_NONE
                stop_summary = (
                    f"Stopped after max_batches={int(max_batches)} in this invocation."
                    if max_batches is not None
                    else "Run incomplete; resume to continue pending batches."
                )

    _mark_progress(
        PHASE_DONE if run_status == SCOUT_STATE_DONE else str(progress_state.get("hydration_phase") or PHASE_NOT_STARTED),
        current_ticker="" if run_status == SCOUT_STATE_DONE else progress_state.get("current_ticker"),
        hydration_status=HYDRATION_STATUS_OK if run_status == SCOUT_STATE_DONE else None,
        finalization_started=True,
    )
    final_summary = _write_outputs(run_status, stop_code=stop_reason_code, stop_text=stop_summary)
    _write_state(run_status, stop_code=stop_reason_code, stop_text=stop_summary)

    return {
        **final_summary,
        "status": "OK",
        "universe_summary_path": str(paths["summary_path"]),
        "universe_scoreboard_path": str(paths["scoreboard_path"]),
        "universe_shortlist_path": str(paths["shortlist_path"]),
        "universe_coverage_path": str(paths["coverage_path"]),
        "universe_scout_calibration_path": str(paths["calibration_path"]),
        "yield_coverage_path": str(paths["yield_path"]),
        "net_debt_coverage_path": str(paths["net_debt_path"]),
        "owner_earnings_quality_path": str(paths["owner_earnings_quality_path"]),
        "maintenance_capex_discipline_path": str(paths["maintenance_capex_discipline_path"]),
        "accounting_quality_path": str(paths["accounting_quality_path"]),
        "balance_sheet_stress_path": str(paths["balance_sheet_stress_path"]),
        "returns_persistence_path": str(paths["returns_persistence_path"]),
        "intangible_economics_path": str(paths["intangible_economics_path"]),
        "reinvestment_efficiency_path": str(paths["reinvestment_efficiency_path"]),
        "intrinsic_discipline_path": str(paths["intrinsic_discipline_path"]),
        "valuation_confidence_path": str(paths["valuation_confidence_path"]),
        "valuation_integrity_path": str(paths["valuation_integrity_path"]),
        "value_type_path": str(paths["value_type_path"]),
        "investment_readiness_path": str(paths["investment_readiness_path"]),
        "graham_dodd_summary_path": str(paths["graham_dodd_summary_path"]),
        "universe_rankings_path": str(paths["rankings_path"]),
        "depth_queue_path": str(paths["depth_queue_path"]),
        "scout_state_path": str(paths["state_path"]),
        "universe_input_path": str(paths["universe_input_path"]),
    }


def open_universe_scout(*, run_id: str, top_n: int = 10) -> dict[str, Any]:
    cfg = get_config()
    paths = _run_paths(cfg=cfg, run_id=run_id)
    summary_path = paths["summary_path"]
    scoreboard_path = paths["scoreboard_path"]
    shortlist_path = paths["shortlist_path"]
    coverage_path = paths["coverage_path"]
    calibration_path = paths["calibration_path"]
    yield_path = paths["yield_path"]
    net_debt_path = paths["net_debt_path"]
    owner_earnings_quality_path = paths["owner_earnings_quality_path"]
    maintenance_capex_discipline_path = paths["maintenance_capex_discipline_path"]
    accounting_quality_path = paths["accounting_quality_path"]
    balance_sheet_stress_path = paths["balance_sheet_stress_path"]
    returns_persistence_path = paths["returns_persistence_path"]
    revenue_dependence_path = paths["revenue_dependence_path"]
    intangible_economics_path = paths["intangible_economics_path"]
    reinvestment_efficiency_path = paths["reinvestment_efficiency_path"]
    intrinsic_discipline_path = paths["intrinsic_discipline_path"]
    evidence_sufficiency_path = paths["evidence_sufficiency_path"]
    valuation_confidence_path = paths["valuation_confidence_path"]
    valuation_integrity_path = paths["valuation_integrity_path"]
    value_type_path = paths["value_type_path"]
    investment_readiness_path = paths["investment_readiness_path"]
    graham_dodd_summary_path = paths["graham_dodd_summary_path"]
    rankings_path = paths["rankings_path"]
    depth_queue_path = paths["depth_queue_path"]
    state_path = paths["state_path"]
    if not (summary_path.exists() and scoreboard_path.exists() and shortlist_path.exists() and coverage_path.exists()):
        state_payload = _safe_json(state_path)
        return {
            "run_id": run_id,
            "status": "MISSING",
            "run_status": str(state_payload.get("status") or SCOUT_STATE_RUNNING) if state_payload else SCOUT_STATE_RUNNING,
            "hydration_status": str(state_payload.get("hydration_status") or HYDRATION_STATUS_IDLE) if state_payload else HYDRATION_STATUS_IDLE,
            "last_progress_phase": str(state_payload.get("last_progress_phase") or PHASE_NOT_STARTED) if state_payload else PHASE_NOT_STARTED,
            "current_ticker": str(state_payload.get("current_ticker") or "") if state_payload else "",
            "primary_scout_blocker": str(state_payload.get("primary_scout_blocker") or "") if state_payload else "",
            "stalled_reason_code": str(state_payload.get("stalled_reason_code") or "") if state_payload else "",
            "facts_blockers": {},
            "universe_summary_path": str(summary_path),
            "universe_scoreboard_path": str(scoreboard_path),
            "universe_shortlist_path": str(shortlist_path),
            "universe_coverage_path": str(coverage_path),
            "universe_scout_calibration_path": str(calibration_path),
            "yield_coverage_path": str(yield_path),
            "net_debt_coverage_path": str(net_debt_path),
            "owner_earnings_quality_path": str(owner_earnings_quality_path),
            "maintenance_capex_discipline_path": str(maintenance_capex_discipline_path),
            "accounting_quality_path": str(accounting_quality_path),
            "balance_sheet_stress_path": str(balance_sheet_stress_path),
            "returns_persistence_path": str(returns_persistence_path),
            "revenue_dependence_path": str(revenue_dependence_path),
            "intangible_economics_path": str(intangible_economics_path),
            "reinvestment_efficiency_path": str(reinvestment_efficiency_path),
            "intrinsic_discipline_path": str(intrinsic_discipline_path),
            "evidence_sufficiency_path": str(evidence_sufficiency_path),
            "valuation_confidence_path": str(valuation_confidence_path),
            "valuation_integrity_path": str(valuation_integrity_path),
            "value_type_path": str(value_type_path),
            "investment_readiness_path": str(investment_readiness_path),
            "graham_dodd_summary_path": str(graham_dodd_summary_path),
            "universe_rankings_path": str(rankings_path),
            "depth_queue_path": str(depth_queue_path),
            "scout_state_path": str(state_path),
        }

    summary_payload = _safe_json(summary_path)
    state_payload = _safe_json(state_path)
    shortlist_payload = _safe_json(shortlist_path)
    coverage_payload = _safe_json(coverage_path)
    calibration_payload = _safe_json(calibration_path)
    candidates = [row for row in (shortlist_payload.get("top_candidates") or []) if isinstance(row, dict)]
    suggestions: list[str] = []
    unknown_reasons = coverage_payload.get("unknown_reason_counts") if isinstance(coverage_payload.get("unknown_reason_counts"), dict) else {}
    if int(unknown_reasons.get("OFFLINE_NO_CACHE", 0)) > 0:
        suggestions.append("Seed price/companyfacts caches in online mode, then rerun universe-scout.")
    if int(unknown_reasons.get("CIK_MISSING", 0)) > 0:
        suggestions.append("Fix CIK mapping for missing tickers before depth handoff.")
    if not suggestions and int((summary_payload.get("counts") or {}).get(PASS, 0)) == 0:
        suggestions.append("CALIBRATION_REQUIRED: no PASS names; run universe-scout-calibration-open and review near misses.")

    blocker_counts = (
        calibration_payload.get("blocker_counts")
        if isinstance(calibration_payload.get("blocker_counts"), dict)
        else (summary_payload.get("coverage") or {}).get("blocker_counts")
    )
    top_blockers = [
        {"primary_blocker_category": str(name), "count": int(count)}
        for name, count in sorted((blocker_counts or {}).items(), key=lambda kv: (-int(kv[1]), str(kv[0])))[:10]
    ]

    return {
        "run_id": run_id,
        "status": "OK",
        "as_of_date": summary_payload.get("as_of_date"),
        "counts": summary_payload.get("counts") if isinstance(summary_payload.get("counts"), dict) else {},
        "thresholds_effective": summary_payload.get("thresholds_effective") if isinstance(summary_payload.get("thresholds_effective"), dict) else {},
        "calibration_required": summary_payload.get("calibration_required") if isinstance(summary_payload.get("calibration_required"), dict) else {},
        "coverage_breakdown": summary_payload.get("coverage") if isinstance(summary_payload.get("coverage"), dict) else {},
        "facts_blockers": summary_payload.get("facts_blockers")
        if isinstance(summary_payload.get("facts_blockers"), dict)
        else {},
        "owner_earnings_quality": summary_payload.get("owner_earnings_quality")
        if isinstance(summary_payload.get("owner_earnings_quality"), dict)
        else {},
        "maintenance_capex_discipline": summary_payload.get("maintenance_capex_discipline")
        if isinstance(summary_payload.get("maintenance_capex_discipline"), dict)
        else {},
        "intangible_economics": summary_payload.get("intangible_economics")
        if isinstance(summary_payload.get("intangible_economics"), dict)
        else {},
        "reinvestment_efficiency": summary_payload.get("reinvestment_efficiency")
        if isinstance(summary_payload.get("reinvestment_efficiency"), dict)
        else {},
        "intrinsic_discipline": summary_payload.get("intrinsic_discipline")
        if isinstance(summary_payload.get("intrinsic_discipline"), dict)
        else {},
        "evidence_sufficiency": summary_payload.get("evidence_sufficiency")
        if isinstance(summary_payload.get("evidence_sufficiency"), dict)
        else {},
        "valuation_confidence": summary_payload.get("valuation_confidence")
        if isinstance(summary_payload.get("valuation_confidence"), dict)
        else {},
        "valuation_integrity": summary_payload.get("valuation_integrity")
        if isinstance(summary_payload.get("valuation_integrity"), dict)
        else {},
        "value_type": summary_payload.get("value_type")
        if isinstance(summary_payload.get("value_type"), dict)
        else {},
        "investment_readiness": summary_payload.get("investment_readiness")
        if isinstance(summary_payload.get("investment_readiness"), dict)
        else {},
        "returns_persistence": summary_payload.get("returns_persistence")
        if isinstance(summary_payload.get("returns_persistence"), dict)
        else {},
        "top_blockers": top_blockers,
        "top_candidates": candidates[: max(1, int(top_n))],
        "batch_progress": summary_payload.get("batch_progress") if isinstance(summary_payload.get("batch_progress"), dict) else {},
        "budget_progress": summary_payload.get("budget_progress") if isinstance(summary_payload.get("budget_progress"), dict) else {},
        "hydration_status": str(
            summary_payload.get("hydration_status")
            or state_payload.get("hydration_status")
            or HYDRATION_STATUS_IDLE
        ),
        "last_progress_phase": str(
            summary_payload.get("last_progress_phase")
            or state_payload.get("last_progress_phase")
            or PHASE_NOT_STARTED
        ),
        "hydration_progress": summary_payload.get("hydration_progress")
        if isinstance(summary_payload.get("hydration_progress"), dict)
        else {
            "current_batch_index": state_payload.get("current_batch_index"),
            "current_batch_tickers": list(state_payload.get("current_batch_tickers") or []),
            "current_ticker": str(state_payload.get("current_ticker") or ""),
            "hydration_phase": str(state_payload.get("hydration_phase") or PHASE_NOT_STARTED),
            "hydration_started_at": state_payload.get("hydration_started_at"),
            "hydration_last_progress_at": state_payload.get("hydration_last_progress_at"),
            "last_completed_ticker": str(state_payload.get("last_completed_ticker") or ""),
            "facts_cache_hits": int(state_payload.get("facts_cache_hits") or 0),
            "facts_cache_misses": int(state_payload.get("facts_cache_misses") or 0),
            "companyfacts_fetch_attempts": int(state_payload.get("companyfacts_fetch_attempts") or 0),
            "companyfacts_failures": int(state_payload.get("companyfacts_failures") or 0),
            "companyfacts_timeouts": int(state_payload.get("companyfacts_timeouts") or 0),
            "price_stage_completed": bool(state_payload.get("price_stage_completed", False)),
            "facts_stage_completed": bool(state_payload.get("facts_stage_completed", False)),
            "scoring_stage_completed": bool(state_payload.get("scoring_stage_completed", False)),
            "finalization_started": bool(state_payload.get("finalization_started", False)),
            "tickers_completed": int(state_payload.get("tickers_completed") or 0),
        },
        "primary_scout_blocker": str(
            summary_payload.get("primary_scout_blocker")
            or state_payload.get("primary_scout_blocker")
            or ""
        ),
        "stalled_reason_code": str(
            summary_payload.get("stalled_reason_code")
            or state_payload.get("stalled_reason_code")
            or ""
        ),
        "run_status": str(summary_payload.get("run_status") or state_payload.get("status") or SCOUT_STATE_DONE),
        "stop_reason_code": str(summary_payload.get("stop_reason_code") or state_payload.get("stop_reason_code") or STOP_NONE),
        "suggestions": suggestions,
        "universe_summary_path": str(summary_path),
        "universe_scoreboard_path": str(scoreboard_path),
        "universe_shortlist_path": str(shortlist_path),
        "universe_coverage_path": str(coverage_path),
        "universe_scout_calibration_path": str(calibration_path),
        "yield_coverage_path": str(yield_path),
        "net_debt_coverage_path": str(net_debt_path),
        "owner_earnings_quality_path": str(owner_earnings_quality_path),
        "maintenance_capex_discipline_path": str(maintenance_capex_discipline_path),
        "accounting_quality_path": str(accounting_quality_path),
        "balance_sheet_stress_path": str(balance_sheet_stress_path),
        "returns_persistence_path": str(returns_persistence_path),
        "revenue_dependence_path": str(revenue_dependence_path),
        "intangible_economics_path": str(intangible_economics_path),
        "reinvestment_efficiency_path": str(reinvestment_efficiency_path),
        "intrinsic_discipline_path": str(intrinsic_discipline_path),
        "evidence_sufficiency_path": str(evidence_sufficiency_path),
        "valuation_confidence_path": str(valuation_confidence_path),
        "valuation_integrity_path": str(valuation_integrity_path),
        "value_type_path": str(value_type_path),
        "investment_readiness_path": str(investment_readiness_path),
        "graham_dodd_summary_path": str(graham_dodd_summary_path),
        "universe_rankings_path": str(rankings_path),
        "depth_queue_path": str(depth_queue_path),
        "scout_state_path": str(state_path),
    }


def open_graham_dodd(*, run_id: str, top_n: int = 10) -> dict[str, Any]:
    cfg = get_config()
    path = cfg.outputs_dir / "universe" / run_id / "graham_dodd_summary.json"
    if not path.exists():
        return {
            "run_id": run_id,
            "status": "MISSING",
            "graham_dodd_summary_path": str(path),
        }
    payload = _safe_json(path)
    rows = [row for row in (payload.get("rows") or []) if isinstance(row, dict)]
    reason_counts = payload.get("reason_counts") if isinstance(payload.get("reason_counts"), dict) else {}
    top_mos_epv = [row for row in (payload.get("top_mos_epv") or []) if isinstance(row, dict)][: max(1, int(top_n))]
    top_mos_netnet = [row for row in (payload.get("top_mos_netnet") or []) if isinstance(row, dict)][: max(1, int(top_n))]
    top_netnet = [row for row in (payload.get("top_netnet_situations") or []) if isinstance(row, dict)][: max(1, int(top_n))]
    return {
        "run_id": run_id,
        "status": "OK",
        "graham_dodd_summary_path": str(path),
        "ticker_count": int(payload.get("ticker_count") or len(rows)),
        "gd_known_count": int(payload.get("gd_known_count") or 0),
        "gd_unknown_count": int(payload.get("gd_unknown_count") or 0),
        "reason_counts": reason_counts,
        "top_mos_epv": top_mos_epv,
        "top_mos_netnet": top_mos_netnet,
        "top_netnet_situations": top_netnet,
    }


def open_universe_depth_queue(*, run_id: str, top_n: int = 10) -> dict[str, Any]:
    cfg = get_config()
    path = cfg.outputs_dir / "universe" / run_id / "depth_queue.json"
    if not path.exists():
        return {
            "run_id": run_id,
            "status": "MISSING",
            "depth_queue_path": str(path),
        }
    payload = _safe_json(path)
    entries = [row for row in (payload.get("entries") or []) if isinstance(row, dict)]
    return {
        "run_id": run_id,
        "status": "OK",
        "depth_queue_path": str(path),
        "queue_count": int(payload.get("queue_count") or len(entries)),
        "entries": entries[: max(1, int(top_n))],
    }


def open_universe_rankings(*, run_id: str, top_n: int = 10) -> dict[str, Any]:
    cfg = get_config()
    path = cfg.outputs_dir / "universe" / run_id / "universe_rankings.json"
    if not path.exists():
        return {
            "run_id": run_id,
            "status": "MISSING",
            "universe_rankings_path": str(path),
        }
    payload = _safe_json(path)
    top_overall = [row for row in (payload.get("top_overall") or []) if isinstance(row, dict)]
    top_pass = [row for row in (payload.get("top_pass") or []) if isinstance(row, dict)]
    top_watch = [row for row in (payload.get("top_watch") or []) if isinstance(row, dict)]
    return {
        "run_id": run_id,
        "status": "OK",
        "universe_rankings_path": str(path),
        "ticker_count": int(payload.get("ticker_count") or 0),
        "denominator_counts": payload.get("denominator_counts") if isinstance(payload.get("denominator_counts"), dict) else {},
        "coverage_stats": payload.get("coverage_stats") if isinstance(payload.get("coverage_stats"), dict) else {},
        "top_overall": top_overall[: max(1, int(top_n))],
        "top_pass": top_pass[: max(1, int(top_n))],
        "top_watch": top_watch[: max(1, int(top_n))],
    }


def universe_depth_queue_to_runs(*, run_id: str, max_runs: int | None = None) -> dict[str, Any]:
    cfg = get_config()
    path = cfg.outputs_dir / "universe" / run_id / "depth_queue.json"
    if not path.exists():
        return {
            "run_id": run_id,
            "status": "MISSING",
            "depth_queue_path": str(path),
            "commands": [],
        }
    payload = _safe_json(path)
    entries = [row for row in (payload.get("entries") or []) if isinstance(row, dict)]
    if max_runs is not None and int(max_runs) > 0:
        entries = entries[: int(max_runs)]
    commands = [str(row.get("command") or "").strip() for row in entries if str(row.get("command") or "").strip()]
    return {
        "run_id": run_id,
        "status": "OK",
        "depth_queue_path": str(path),
        "max_runs": int(max_runs) if max_runs is not None else None,
        "queue_count": int(payload.get("queue_count") or len([row for row in (payload.get("entries") or []) if isinstance(row, dict)])),
        "selected_count": len(commands),
        "commands": commands,
    }


def open_universe_scout_calibration(*, run_id: str, top_n: int = 10) -> dict[str, Any]:
    cfg = get_config()
    path = cfg.sectors_dir / run_id / "universe_scout_calibration.json"
    if not path.exists():
        return {
            "run_id": run_id,
            "status": "MISSING",
            "universe_scout_calibration_path": str(path),
        }
    payload = _safe_json(path)
    counts = payload.get("counts") if isinstance(payload.get("counts"), dict) else {}
    blocker_counts = payload.get("blocker_counts") if isinstance(payload.get("blocker_counts"), dict) else {}
    yield_subreason_counts = (
        payload.get("yield_blocker_subreason_counts")
        if isinstance(payload.get("yield_blocker_subreason_counts"), dict)
        else {}
    )
    near_misses = [row for row in (payload.get("top_near_misses") or []) if isinstance(row, dict)]
    adjustments = [str(item) for item in (payload.get("suggested_threshold_adjustments") or []) if str(item).strip()]
    top_blockers = [
        {"primary_blocker_category": str(name), "count": int(count)}
        for name, count in sorted(blocker_counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))[: max(1, int(top_n))]
    ]
    top_near_misses = [
        {
            "ticker": str(row.get("ticker") or ""),
            "primary_blocker_category": str(row.get("primary_blocker_category") or BLOCKER_OTHER_UNKNOWN),
            "delta_to_pass": row.get("delta_to_pass", UNKNOWN),
            "yield_metric_used": str(row.get("yield_metric_used") or UNKNOWN),
            "yield_denominator_used": str(row.get("yield_denominator_used") or UNKNOWN),
            "ev_status": str(row.get("ev_status") or UNKNOWN),
            "mos_epv": row.get("mos_epv", UNKNOWN),
            "mos_netnet": row.get("mos_netnet", UNKNOWN),
            "gd_primary_reason_code": str(row.get("gd_primary_reason_code") or UNKNOWN),
            "yield_delta_to_pass": row.get("yield_delta_to_pass", UNKNOWN),
        }
        for row in near_misses[: max(1, int(top_n))]
    ]
    return {
        "run_id": run_id,
        "status": "OK",
        "universe_scout_calibration_path": str(path),
        "counts": {
            PASS: int(counts.get(PASS, 0)),
            WATCH: int(counts.get(WATCH, 0)),
            FAIL: int(counts.get(FAIL, 0)),
        },
        "thresholds_effective": payload.get("thresholds_effective") if isinstance(payload.get("thresholds_effective"), dict) else {},
        "blocker_counts": blocker_counts,
        "yield_blocker_subreason_counts": yield_subreason_counts,
        "top_blockers": top_blockers,
        "missing_input_breakdown": payload.get("missing_input_breakdown")
        if isinstance(payload.get("missing_input_breakdown"), dict)
        else {},
        "top_near_misses": top_near_misses,
        "suggested_threshold_adjustments": adjustments,
        "calibration_required": payload.get("calibration_required") if isinstance(payload.get("calibration_required"), dict) else {},
    }


def open_universe_yield_coverage(*, run_id: str, top_n: int = 10) -> dict[str, Any]:
    cfg = get_config()
    path = cfg.outputs_dir / "universe" / run_id / "yield_coverage.json"
    calibration_path = cfg.sectors_dir / run_id / "universe_scout_calibration.json"
    if not path.exists():
        return {
            "run_id": run_id,
            "status": "MISSING",
            "yield_coverage_path": str(path),
        }
    payload = _safe_json(path)
    rows = [row for row in (payload.get("rows") or []) if isinstance(row, dict)]
    known = len([row for row in rows if str(row.get("yield_status") or "").upper() in {"OK", "LOW"}])
    unknown = len([row for row in rows if str(row.get("yield_status") or "").upper() == "UNKNOWN"])
    blocker_breakdown = payload.get("yield_blocker_breakdown") if isinstance(payload.get("yield_blocker_breakdown"), dict) else {}
    ev_known = len([row for row in rows if str(row.get("ev_status") or "").upper() == "OK"])
    ev_unknown = len(rows) - ev_known
    denominator_breakdown: dict[str, int] = {"EV": 0, "MARKET_CAP": 0, "UNKNOWN": 0}
    blocker_by_denominator: dict[str, int] = {}
    for row in rows:
        denominator = str(row.get("yield_denominator_used") or "UNKNOWN").upper()
        if denominator not in denominator_breakdown:
            denominator = "UNKNOWN"
        denominator_breakdown[denominator] = denominator_breakdown.get(denominator, 0) + 1
        reason = str(row.get("yield_reason_code") or "UNKNOWN").upper()
        if reason in {
            BLOCKER_LOW_YIELD_OWNER_EARNINGS_EV,
            BLOCKER_LOW_YIELD_FCF_EV,
            BLOCKER_LOW_YIELD_OWNER_EARNINGS,
            BLOCKER_LOW_YIELD_FCF,
            BLOCKER_MISSING_EV,
            BLOCKER_NEGATIVE_CFO,
            BLOCKER_NEGATIVE_FCF,
        }:
            key = f"{denominator}:{reason}"
            blocker_by_denominator[key] = blocker_by_denominator.get(key, 0) + 1
    blocker_by_denominator = dict(sorted(blocker_by_denominator.items(), key=lambda kv: (-kv[1], kv[0])))

    near_miss_rows = [
        row
        for row in rows
        if str(row.get("scout_status") or "").upper() == WATCH
        and str(row.get("yield_status") or "").upper() == "LOW"
        and _is_num(row.get("yield_delta_to_pass"))
    ]
    near_miss_rows.sort(
        key=lambda row: (
            float(row.get("yield_delta_to_pass") or 0.0),
            str(row.get("ticker") or ""),
        )
    )
    top_near_misses = [
        {
            "ticker": str(row.get("ticker") or ""),
            "yield_metric_used": str(row.get("yield_metric_used") or UNKNOWN),
            "yield_denominator_used": str(row.get("yield_denominator_used") or UNKNOWN),
            "yield_reason_code": str(row.get("yield_reason_code") or "UNKNOWN"),
            "yield_delta_to_pass": row.get("yield_delta_to_pass", UNKNOWN),
        }
        for row in near_miss_rows[: max(1, int(top_n))]
    ]

    calibration_payload = _safe_json(calibration_path)
    yield_subreason_counts = (
        calibration_payload.get("yield_blocker_subreason_counts")
        if isinstance(calibration_payload.get("yield_blocker_subreason_counts"), dict)
        else {}
    )
    return {
        "run_id": run_id,
        "status": "OK",
        "yield_coverage_path": str(path),
        "known_count": int(known),
        "unknown_count": int(unknown),
        "ev_known_count": int(payload.get("ev_known_count", ev_known)),
        "ev_unknown_count": int(payload.get("ev_unknown_count", ev_unknown)),
        "yield_denominator_counts": denominator_breakdown,
        "status_counts": payload.get("status_counts") if isinstance(payload.get("status_counts"), dict) else {},
        "yield_reason_counts": payload.get("yield_reason_counts") if isinstance(payload.get("yield_reason_counts"), dict) else {},
        "yield_blocker_breakdown": blocker_breakdown,
        "yield_blocker_by_denominator": blocker_by_denominator,
        "yield_blocker_subreason_counts": yield_subreason_counts,
        "top_near_miss_watch_by_yield_delta": top_near_misses,
    }


def open_net_debt_coverage(*, run_id: str) -> dict[str, Any]:
    cfg = get_config()
    path = cfg.outputs_dir / "universe" / run_id / "net_debt_coverage.json"
    if not path.exists():
        return {
            "run_id": run_id,
            "status": "MISSING",
            "net_debt_coverage_path": str(path),
        }
    payload = _safe_json(path)
    entries = [row for row in (payload.get("entries") or []) if isinstance(row, dict)]
    status_counts = payload.get("status_counts") if isinstance(payload.get("status_counts"), dict) else {}
    reason_counts = payload.get("reason_counts") if isinstance(payload.get("reason_counts"), dict) else {}
    missing_components: list[dict[str, Any]] = []
    for row in entries:
        reason = str(row.get("reason_code") or "UNKNOWN").upper()
        if reason not in {"MISSING_DEBT", "MISSING_CASH", "MISSING_BOTH", "NO_FACTS"}:
            continue
        if reason == "MISSING_DEBT":
            suggestion = "expand debt tag mapping or hydrate facts"
        elif reason == "MISSING_CASH":
            suggestion = "expand cash tag mapping or hydrate facts"
        else:
            suggestion = "hydrate facts"
        missing_components.append(
            {
                "ticker": str(row.get("ticker") or ""),
                "reason_code": reason,
                "debt_tag": (row.get("tags_used") or {}).get("debt_tag") if isinstance(row.get("tags_used"), dict) else None,
                "cash_tag": (row.get("tags_used") or {}).get("cash_tag") if isinstance(row.get("tags_used"), dict) else None,
                "suggested_action": suggestion,
            }
        )
    missing_components = sorted(missing_components, key=lambda row: (str(row.get("reason_code") or ""), str(row.get("ticker") or "")))
    return {
        "run_id": run_id,
        "status": "OK",
        "net_debt_coverage_path": str(path),
        "ticker_count": len(entries),
        "status_counts": status_counts,
        "reason_counts": reason_counts,
        "missing_components": missing_components,
    }


def open_universe_scout_status(*, run_id: str) -> dict[str, Any]:
    cfg = get_config()
    paths = _run_paths(cfg=cfg, run_id=run_id)
    state_payload = _safe_json(paths["state_path"])
    if not state_payload:
        return {
            "run_id": run_id,
            "status": "MISSING",
            "scout_state_path": str(paths["state_path"]),
        }
    summary_payload = _safe_json(paths["summary_path"])
    return {
        "run_id": run_id,
        "status": "OK",
        "scout_state_path": str(paths["state_path"]),
        "run_status": str(state_payload.get("status") or SCOUT_STATE_RUNNING),
        "as_of_date": state_payload.get("as_of_date"),
        "batches_done": int(state_payload.get("batches_done") or 0),
        "total_batches": int(state_payload.get("total_batches") or 0),
        "last_completed_batch": int(state_payload.get("last_completed_batch") or -1),
        "tickers_completed": int(state_payload.get("tickers_completed") or 0),
        "remaining_tickers": int(state_payload.get("remaining_tickers") or 0),
        "current_batch_index": state_payload.get("current_batch_index"),
        "current_batch_tickers": list(state_payload.get("current_batch_tickers") or []),
        "current_ticker": str(state_payload.get("current_ticker") or ""),
        "hydration_phase": str(state_payload.get("hydration_phase") or PHASE_NOT_STARTED),
        "hydration_status": str(state_payload.get("hydration_status") or HYDRATION_STATUS_IDLE),
        "hydration_started_at": state_payload.get("hydration_started_at"),
        "hydration_last_progress_at": state_payload.get("hydration_last_progress_at"),
        "last_progress_phase": str(state_payload.get("last_progress_phase") or PHASE_NOT_STARTED),
        "last_completed_ticker": str(state_payload.get("last_completed_ticker") or ""),
        "facts_cache_hits": int(state_payload.get("facts_cache_hits") or 0),
        "facts_cache_misses": int(state_payload.get("facts_cache_misses") or 0),
        "companyfacts_fetch_attempts": int(state_payload.get("companyfacts_fetch_attempts") or 0),
        "companyfacts_failures": int(state_payload.get("companyfacts_failures") or 0),
        "companyfacts_timeouts": int(state_payload.get("companyfacts_timeouts") or 0),
        "price_stage_completed": bool(state_payload.get("price_stage_completed", False)),
        "facts_stage_completed": bool(state_payload.get("facts_stage_completed", False)),
        "scoring_stage_completed": bool(state_payload.get("scoring_stage_completed", False)),
        "finalization_started": bool(state_payload.get("finalization_started", False)),
        "primary_scout_blocker": str(state_payload.get("primary_scout_blocker") or ""),
        "stalled_reason_code": str(state_payload.get("stalled_reason_code") or ""),
        "retryable_facts_blocker_count": int(summary_payload.get("facts_blockers", {}).get("retryable_facts_blocker_count") or 0)
        if isinstance(summary_payload.get("facts_blockers"), dict)
        else 0,
        "terminal_facts_blocker_count": int(summary_payload.get("facts_blockers", {}).get("terminal_facts_blocker_count") or 0)
        if isinstance(summary_payload.get("facts_blockers"), dict)
        else 0,
        "partial_usable_facts_count": int(summary_payload.get("facts_blockers", {}).get("partial_usable_facts_count") or 0)
        if isinstance(summary_payload.get("facts_blockers"), dict)
        else 0,
        "economic_fail_count_vs_evidence_fail_count": summary_payload.get("facts_blockers", {}).get("economic_fail_count_vs_evidence_fail_count")
        if isinstance(summary_payload.get("facts_blockers"), dict)
        else {},
        "budget_remaining": state_payload.get("budget_remaining") if isinstance(state_payload.get("budget_remaining"), dict) else {},
        "stop_reason_code": str(state_payload.get("stop_reason_code") or STOP_NONE),
        "stop_summary": str(state_payload.get("stop_summary") or ""),
        "counts": summary_payload.get("counts") if isinstance(summary_payload.get("counts"), dict) else {},
        "top_blocker_categories": summary_payload.get("top_blocker_categories")
        if isinstance(summary_payload.get("top_blocker_categories"), list)
        else [],
    }


def run_universe_scout_resume(
    *,
    run_id: str,
    max_batches: int | None = None,
    scout_sec_budget: int | None = None,
    scout_net_budget: int | None = None,
    scout_max_seconds: int | None = None,
) -> dict[str, Any]:
    cfg = get_config()
    paths = _run_paths(cfg=cfg, run_id=run_id)
    state_payload = _safe_json(paths["state_path"])
    if not state_payload:
        raise ValueError(f"Missing scout_state.json for run_id={run_id}")
    if str(state_payload.get("status") or "").upper() == SCOUT_STATE_DONE:
        return {
            "status": "OK",
            "run_id": run_id,
            "summary": "Run already completed.",
            "scout_state_path": str(paths["state_path"]),
        }
    if str(state_payload.get("status") or "").upper() == SCOUT_STATE_CANCELLED:
        raise ValueError("Run is CANCELLED. Use --force-restart on universe-scout to restart.")

    input_payload = _safe_json(paths["universe_input_path"])
    canonical_tickers = [str(t).strip().upper() for t in (input_payload.get("canonical_tickers") or []) if str(t).strip()]
    source = input_payload.get("source") if isinstance(input_payload.get("source"), dict) else {}
    universe_file = Path(str(source.get("universe_file"))) if str(source.get("universe_file") or "").strip() else None
    sector_run_id = str(source.get("sector_run_id") or "").strip() or None
    top_n = int(state_payload.get("top_n") or 50)
    with_prices = bool(state_payload.get("with_prices", True))
    return run_universe_scout(
        run_id=run_id,
        as_of_date=str(state_payload.get("as_of_date") or ""),
        top_n=max(1, int(top_n)),
        tickers=canonical_tickers or None,
        sector_run_id=sector_run_id,
        universe_csv=universe_file,
        with_prices=with_prices,
        batch_size=max(1, int(state_payload.get("batch_size") or 200)),
        max_batches=max_batches,
        scout_sec_budget=scout_sec_budget,
        scout_net_budget=scout_net_budget,
        scout_max_seconds=scout_max_seconds,
        threshold_overrides=state_payload.get("thresholds_effective") if isinstance(state_payload.get("thresholds_effective"), dict) else None,
    )


def cancel_universe_scout_run(*, run_id: str, reason: str) -> dict[str, Any]:
    cfg = get_config()
    paths = _run_paths(cfg=cfg, run_id=run_id)
    state_payload = _safe_json(paths["state_path"])
    if not state_payload:
        return {
            "run_id": run_id,
            "status": "MISSING",
            "scout_state_path": str(paths["state_path"]),
        }
    state_payload["status"] = SCOUT_STATE_CANCELLED
    state_payload["stop_reason_code"] = STOP_CANCELLED
    state_payload["stop_summary"] = str(reason or "Cancelled by user.")
    state_payload["updated_at"] = utc_now_iso()
    _json_write(paths["state_path"], state_payload)
    return {
        "run_id": run_id,
        "status": "OK",
        "scout_state_path": str(paths["state_path"]),
        "run_status": SCOUT_STATE_CANCELLED,
        "stop_reason_code": STOP_CANCELLED,
        "stop_summary": state_payload["stop_summary"],
    }


def run_universe_scout_to_depth(
    *,
    scout_run_id: str,
    depth_run_id: str,
    sector: str,
    iterations: int,
    top_k: int,
    shortlist_limit: int | None = None,
    force_restart: bool = False,
    with_research: bool = True,
    with_synthesis: bool = True,
) -> dict[str, Any]:
    from app.rlm.loop import run_sector_rlm_loop

    cfg = get_config()
    scout_dir = cfg.sectors_dir / scout_run_id
    shortlist_path = scout_dir / "universe_shortlist.json"
    calibration_path = scout_dir / "universe_scout_calibration.json"
    shortlist_payload = _safe_json(shortlist_path)
    calibration_payload = _safe_json(calibration_path)
    if not shortlist_payload:
        raise ValueError(f"Missing or invalid universe shortlist for run_id={scout_run_id}")
    as_of_date = str(shortlist_payload.get("as_of_date") or "").strip()
    if not as_of_date:
        raise ValueError(f"Universe shortlist missing as_of_date: {shortlist_path}")

    top_candidates = [row for row in (shortlist_payload.get("top_candidates") or []) if isinstance(row, dict)]
    pass_candidates = [row for row in top_candidates if str(row.get("scout_status") or "").upper() == PASS]
    watch_candidates = [row for row in top_candidates if str(row.get("scout_status") or "").upper() == WATCH]
    selected_pool = top_candidates
    selection_rationale = "DEFAULT_SHORTLIST_ORDER"
    if not pass_candidates and watch_candidates:
        near_miss_rows = [row for row in (calibration_payload.get("top_near_misses") or []) if isinstance(row, dict)]
        near_miss_watch_order = _dedupe_tickers_keep_order(
            [
                str(row.get("ticker") or "")
                for row in near_miss_rows
                if str(row.get("ticker") or "").strip().upper() in {
                    str(candidate.get("ticker") or "").strip().upper() for candidate in watch_candidates
                }
            ]
        )
        watch_by_ticker = {
            str(row.get("ticker") or "").strip().upper(): row
            for row in watch_candidates
            if str(row.get("ticker") or "").strip()
        }
        near_miss_watch = [watch_by_ticker[ticker] for ticker in near_miss_watch_order if ticker in watch_by_ticker]
        remaining_watch = [
            row for row in watch_candidates if str(row.get("ticker") or "").strip().upper() not in set(near_miss_watch_order)
        ]
        selected_pool = near_miss_watch + remaining_watch
        selection_rationale = "PASS=0_NEAR_MISS_WATCH_PRIORITY"
    selected = selected_pool[: int(shortlist_limit)] if isinstance(shortlist_limit, int) and shortlist_limit > 0 else selected_pool
    selected_tickers = _dedupe_tickers_keep_order([str(row.get("ticker") or "") for row in selected])
    if not selected_tickers:
        raise ValueError("Universe shortlist has no PASS/WATCH tickers to pass into depth.")

    depth_payload = run_sector_rlm_loop(
        sector=sector,
        as_of_date=as_of_date,
        run_id=depth_run_id,
        peer_limit=max(1, len(selected_tickers)),
        min_peers=max(1, min(len(selected_tickers), int(top_k))),
        limit_dossiers=max(1, len(selected_tickers)),
        years_back=10,
        workers=4,
        iterations=max(1, int(iterations)),
        top_k=max(1, int(top_k)),
        with_research=bool(with_research),
        with_synthesis=bool(with_synthesis),
        with_prices=True,
        budget_usd=None,
        resume=False,
        mode="depth",
        force_restart=bool(force_restart),
        seed_tickers=selected_tickers,
        parent_run_id=scout_run_id,
        shortlist_source=str(shortlist_path),
    )

    depth_dir = cfg.sectors_dir / depth_run_id
    depth_dir.mkdir(parents=True, exist_ok=True)
    linkage_path = depth_dir / "universe_scout_linkage.json"
    linkage_payload = _safe_json(linkage_path)
    linkage_payload.update(
        {
            "scout_run_id": scout_run_id,
            "depth_run_id": depth_run_id,
            "as_of_date": as_of_date,
            "sector": sector,
            "shortlist_source": str(shortlist_path),
            "calibration_source": str(calibration_path),
            "selection_rationale": selection_rationale,
            "selected_tickers": selected_tickers,
            "selected_candidates": [
                {
                    "ticker": str(row.get("ticker") or ""),
                    "scout_status": str(row.get("scout_status") or WATCH),
                    "score_total": float(row.get("score_total") or 0.0),
                    "primary_blocker_category": str(row.get("primary_blocker_category") or BLOCKER_OTHER_UNKNOWN),
                    "delta_to_pass": row.get("delta_to_pass", UNKNOWN),
                    "derived_from": list(row.get("derived_from") or []),
                }
                for row in selected
            ],
            "generated_at": utc_now_iso(),
            "derived_from": [
                str(shortlist_path),
                "universe_shortlist.top_candidates[*].derived_from",
                str(calibration_path),
                "universe_scout_calibration.top_near_misses[*]",
            ],
        }
    )
    linkage_path.write_text(json.dumps(linkage_payload, indent=2), encoding="utf-8")

    return {
        "status": "OK",
        "scout_run_id": scout_run_id,
        "depth_run_id": depth_run_id,
        "shortlist_source": str(shortlist_path),
        "selection_rationale": selection_rationale,
        "selected_tickers": selected_tickers,
        "linkage_path": str(linkage_path),
        "depth_result": depth_payload,
    }
