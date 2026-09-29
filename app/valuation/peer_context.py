"""Peer-relative context.

Computes sector peer distributions for quality and valuation metrics, then
compares a ticker against its detected sector peers.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

from app.autonomous.financial_integrity import MARKET_CAP_UNIT_USD_MILLIONS
from app.db import get_db
from app.util.financial_data_access import (
    ANNUAL_COMPANYFACTS_PERIOD_TYPES,
    companyfacts_rows,
    latest_companyfacts_value,
)
from app.valuation.lineage import latest_decision_eligible_valuation_rows

logger = logging.getLogger(__name__)

_MEDIANS_TTL_SECONDS = 24 * 3600  # 24 hours
_CACHE_SCHEMA_VERSION = 3
_MIN_PEERS = 5

SUPPORTED_COMPARISON_METRICS = (
    "roic",
    "operating_margin",
    "revenue_growth_5y",
    "ev_ebitda",
    "ev_ebit",
    "ev_sales",
    "p_e",
    "p_b",
    "fcf_yield",
    "dividend_yield",
)

PEER_METRIC_DISPLAY_NAMES = {
    "roic": "ROIC",
    "operating_margin": "Operating margin",
    "revenue_growth_5y": "Revenue growth 5Y",
    "ev_ebitda": "EV/EBITDA",
    "ev_ebit": "EV/EBIT",
    "ev_sales": "EV/Sales",
    "p_e": "P/E",
    "p_b": "P/B",
    "fcf_yield": "FCF yield",
    "dividend_yield": "Dividend yield",
}

_PEER_METRIC_ALIASES = {
    "roic": "roic",
    "operating_margin": "operating_margin",
    "operating margin": "operating_margin",
    "revenue_growth_5y": "revenue_growth_5y",
    "revenue growth 5y": "revenue_growth_5y",
    "revenue growth": "revenue_growth_5y",
    "ev/ebitda": "ev_ebitda",
    "ev_ebitda": "ev_ebitda",
    "evebitda": "ev_ebitda",
    "ev to ebitda": "ev_ebitda",
    "ev/ebit": "ev_ebit",
    "ev_ebit": "ev_ebit",
    "evebit": "ev_ebit",
    "ev to ebit": "ev_ebit",
    "ev/sales": "ev_sales",
    "ev_sales": "ev_sales",
    "evsales": "ev_sales",
    "ev to sales": "ev_sales",
    "p/e": "p_e",
    "p_e": "p_e",
    "pe": "p_e",
    "price/earnings": "p_e",
    "price to earnings": "p_e",
    "p/b": "p_b",
    "p_b": "p_b",
    "pb": "p_b",
    "price/book": "p_b",
    "price to book": "p_b",
    "fcf yield": "fcf_yield",
    "fcf_yield": "fcf_yield",
    "free cash flow yield": "fcf_yield",
    "dividend yield": "dividend_yield",
    "dividend_yield": "dividend_yield",
}

_RELATIVE_RATIO_KEYS = {
    "roic": "roic_vs_median",
    "operating_margin": "operating_margin_vs_median",
    "revenue_growth_5y": "revenue_growth_vs_median",
    "ev_ebitda": "ev_ebitda_vs_median",
    "ev_ebit": "ev_ebit_vs_median",
    "ev_sales": "ev_sales_vs_median",
    "p_e": "p_e_vs_median",
    "p_b": "p_b_vs_median",
    "fcf_yield": "fcf_yield_vs_median",
    "dividend_yield": "dividend_yield_vs_median",
}


def normalize_peer_metric(metric: str) -> str | None:
    normalized = " ".join(str(metric or "").strip().lower().replace("-", " ").split())
    normalized = normalized.replace(" / ", "/").replace(" /", "/").replace("/ ", "/")
    if normalized in _PEER_METRIC_ALIASES:
        return _PEER_METRIC_ALIASES[normalized]
    compact = normalized.replace(" ", "").replace("_", "")
    return _PEER_METRIC_ALIASES.get(compact)


def _cache_dir() -> Path:
    return Path("data/cache/sector_medians")


def _cache_path(sector: str, as_of_date: str) -> Path:
    asof_token = str(as_of_date or "").strip() or "unknown_asof"
    return _cache_dir() / f"{sector}__{asof_token}.json"


def _cache_is_fresh(path: Path, ttl_seconds: int = _MEDIANS_TTL_SECONDS) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema_version") != _CACHE_SCHEMA_VERSION:
            return False
        computed_at = data.get("computed_at", "")
        ts = datetime.fromisoformat(computed_at.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - ts).total_seconds() < ttl_seconds
    except Exception:
        return False


def _load_sector_tickers(sector: str) -> list[str]:
    """Load tickers for a sector from the sector universe cache.

    Falls back to discover_sector_universe if cache is missing.
    """
    cache = Path("data/cache/sector_universe") / f"{sector}.json"
    if cache.exists():
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            rows = data.get("rows", [])
            return [str(r.get("ticker")) for r in rows if isinstance(r, dict) and r.get("ticker")]
        except Exception:
            pass
    # Fallback: call discover_sector_universe (makes SEC API call, cached 7 days)
    try:
        from app.universe.sector_universe import discover_sector_universe

        results = discover_sector_universe(sector)
        return [str(r.get("ticker")) for r in results if isinstance(r, dict) and r.get("ticker")]
    except Exception as exc:
        logger.warning("peer_context: failed to load sector tickers for %s: %s", sector, exc)
        return []


def _latest_market_price(conn: Any, ticker: str, *, as_of_date: str) -> float | None:
    try:
        rows = latest_decision_eligible_valuation_rows(
            conn,
            ticker=ticker.upper(),
            as_of_date=as_of_date,
        )
    except Exception:
        return None
    rows.sort(
        key=lambda row: (
            str(row["as_of_date"] or ""),
            str(row["created_at"] or ""),
            int(row["id"]),
        ),
        reverse=True,
    )
    for row in rows:
        raw = row["inputs_json"]
        if not raw:
            continue
        try:
            payload = json.loads(str(raw))
        except Exception:
            continue
        for key in ("market_price", "price"):
            value = payload.get(key)
            if isinstance(value, (int, float)) and value > 0:
                return float(value)
    return None


def _latest_shares_outstanding(conn: Any, ticker: str, *, as_of_date: str) -> float | None:
    return latest_companyfacts_value(
        conn,
        ticker,
        line_item="shares_outstanding",
        period_types=ANNUAL_COMPANYFACTS_PERIOD_TYPES,
        as_of_date=as_of_date,
        value_not_null=True,
        require_filed_asof=True,
    )


def _latest_market_cap(conn: Any, ticker: str, *, as_of_date: str) -> float | None:
    try:
        columns = {
            str(row["name"]) for row in conn.execute("PRAGMA table_info(market_caps)").fetchall()
        }
        if "market_cap_unit" not in columns:
            return None
        row = conn.execute(
            """
            SELECT market_cap, market_cap_unit
            FROM market_caps
            WHERE ticker = ?
              AND market_cap_status = 'OK'
              AND market_cap IS NOT NULL
              AND market_cap_unit = ?
              AND effective_as_of_date <= ?
            ORDER BY effective_as_of_date DESC, id DESC
            LIMIT 1
            """,
            (ticker.upper(), MARKET_CAP_UNIT_USD_MILLIONS, as_of_date),
        ).fetchone()
    except Exception:
        row = None
    value = row["market_cap"] if row and "market_cap" in row.keys() else None
    if isinstance(value, (int, float)) and value > 0:
        return float(value)
    return None


def _market_cap_to_statement_units(
    market_cap: float | None, reference_value: float | None
) -> float | None:
    """Identity conversion: both canonical inputs are already USD millions."""

    _ = reference_value
    if not isinstance(market_cap, (int, float)):
        return None
    return float(market_cap)


def _safe_div(numerator: float | None, denominator: float | None) -> float | None:
    if not isinstance(numerator, (int, float)) or not isinstance(denominator, (int, float)):
        return None
    if float(denominator) == 0.0:
        return None
    return float(numerator) / float(denominator)


def _positive_div(numerator: float | None, denominator: float | None) -> float | None:
    if not isinstance(denominator, (int, float)) or float(denominator) <= 0:
        return None
    return _safe_div(numerator, denominator)


def _compute_ticker_metrics(ticker: str, *, as_of_date: str) -> dict[str, float | None]:
    """Compute peer-comparable metrics for a single ticker from companyfacts."""
    with get_db() as conn:

        def _latest(item: str) -> float | None:
            return latest_companyfacts_value(
                conn,
                ticker,
                line_item=item,
                period_types=ANNUAL_COMPANYFACTS_PERIOD_TYPES,
                as_of_date=as_of_date,
                require_filed_asof=True,
            )

        def _revenue_cagr() -> float | None:
            rows = companyfacts_rows(
                conn,
                ticker,
                columns=("fiscal_year", "value"),
                period_types=ANNUAL_COMPANYFACTS_PERIOD_TYPES,
                line_items=("revenue",),
                as_of_date=as_of_date,
                require_filed_asof=True,
                order_by="fiscal_year ASC",
            )
            if len(rows) < 2:
                return None
            oldest_yr, oldest_val = int(rows[0][0]), float(rows[0][1])
            latest_yr, latest_val = int(rows[-1][0]), float(rows[-1][1])
            years = latest_yr - oldest_yr
            if years <= 0 or oldest_val <= 0:
                return None
            return (latest_val / oldest_val) ** (1.0 / years) - 1.0

        oi = _latest("operating_income")
        ta = _latest("total_assets")
        rev = _latest("revenue")
        debt = _latest("total_debt")
        cash = _latest("cash")
        depreciation_amortization = _latest("depreciation_amortization")
        depreciation = _latest("depreciation")
        intangible_amortization = _latest("intangible_amortization")
        net_income = _latest("net_income")
        equity = _latest("equity")
        cfo = _latest("cfo")
        capex = _latest("capex")
        dividends = _latest("dividends_paid_amount")
        market_cap = _latest_market_cap(conn, ticker, as_of_date=as_of_date)
        market_cap_statement_units = _market_cap_to_statement_units(
            market_cap, rev or oi or net_income
        )

        roic = (
            oi / ta
            if isinstance(oi, (int, float)) and isinstance(ta, (int, float)) and ta > 0
            else None
        )
        op_margin = (
            oi / rev
            if isinstance(oi, (int, float)) and isinstance(rev, (int, float)) and rev > 0
            else None
        )
        rev_growth = _revenue_cagr()

        da = depreciation_amortization
        if da is None:
            da_parts = [
                value
                for value in (depreciation, intangible_amortization)
                if isinstance(value, (int, float))
            ]
            da = sum(da_parts) if da_parts else None
        ebitda = (
            float(oi) + float(da)
            if isinstance(oi, (int, float)) and isinstance(da, (int, float))
            else None
        )
        enterprise_value = (
            market_cap_statement_units + float(debt) - float(cash)
            if isinstance(market_cap_statement_units, (int, float))
            and isinstance(debt, (int, float))
            and isinstance(cash, (int, float))
            else None
        )
        fcf = (
            float(cfo) - abs(float(capex))
            if isinstance(cfo, (int, float)) and isinstance(capex, (int, float))
            else None
        )
        dividend_amount = abs(float(dividends)) if isinstance(dividends, (int, float)) else None

    return {
        "roic": round(roic, 4) if roic is not None else None,
        "operating_margin": round(op_margin, 4) if op_margin is not None else None,
        "revenue_growth_5y": round(rev_growth, 4) if rev_growth is not None else None,
        "ev_ebitda": round(_positive_div(enterprise_value, ebitda), 4)
        if _positive_div(enterprise_value, ebitda) is not None
        else None,
        "ev_ebit": round(_positive_div(enterprise_value, oi), 4)
        if _positive_div(enterprise_value, oi) is not None
        else None,
        "ev_sales": round(_positive_div(enterprise_value, rev), 4)
        if _positive_div(enterprise_value, rev) is not None
        else None,
        "p_e": round(_positive_div(market_cap_statement_units, net_income), 4)
        if _positive_div(market_cap_statement_units, net_income) is not None
        else None,
        "p_b": round(_positive_div(market_cap_statement_units, equity), 4)
        if _positive_div(market_cap_statement_units, equity) is not None
        else None,
        "fcf_yield": round(_safe_div(fcf, market_cap_statement_units), 4)
        if _safe_div(fcf, market_cap_statement_units) is not None
        else None,
        "dividend_yield": round(_safe_div(dividend_amount, market_cap_statement_units), 4)
        if _safe_div(dividend_amount, market_cap_statement_units) is not None
        else None,
    }


def compute_sector_medians(
    sector: str,
    as_of_date: str,
    *,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Compute median ROIC, operating margin, and revenue growth for a sector.

    Returns cached result if within TTL, otherwise recomputes.
    """
    cp = _cache_path(sector, as_of_date)
    if use_cache and _cache_is_fresh(cp):
        try:
            return json.loads(cp.read_text(encoding="utf-8"))
        except Exception:
            pass

    tickers = _load_sector_tickers(sector)

    # Compute metrics for each peer
    peer_metrics: list[dict[str, Any]] = []
    for t in tickers:
        metrics = _compute_ticker_metrics(t, as_of_date=as_of_date)
        if any(v is not None for v in metrics.values()):
            peer_metrics.append({"ticker": t, **metrics})

    peer_count = len(peer_metrics)

    if peer_count < _MIN_PEERS:
        return {
            "schema_version": _CACHE_SCHEMA_VERSION,
            "sector": sector,
            "as_of_date": as_of_date,
            "peer_count": peer_count,
            "peer_tickers": [p["ticker"] for p in peer_metrics],
            "medians": {metric: None for metric in SUPPORTED_COMPARISON_METRICS},
            "q1": {metric: None for metric in SUPPORTED_COMPARISON_METRICS},
            "q3": {metric: None for metric in SUPPORTED_COMPARISON_METRICS},
            "metric_values": {metric: [] for metric in SUPPORTED_COMPARISON_METRICS},
            "computed_at": datetime.now(timezone.utc).isoformat(),
        }

    def _values_for(key: str) -> list[float]:
        return sorted(float(p[key]) for p in peer_metrics if isinstance(p.get(key), (int, float)))

    def _median_of(key: str) -> float | None:
        vals = _values_for(key)
        return round(median(vals), 4) if len(vals) >= _MIN_PEERS else None

    def _quartiles_of(key: str) -> tuple[float | None, float | None]:
        vals = _values_for(key)
        if len(vals) < _MIN_PEERS:
            return None, None
        midpoint = len(vals) // 2
        if len(vals) % 2:
            lower = vals[:midpoint]
            upper = vals[midpoint + 1 :]
        else:
            lower = vals[:midpoint]
            upper = vals[midpoint:]
        if not lower or not upper:
            return None, None
        return round(median(lower), 4), round(median(upper), 4)

    q1: dict[str, float | None] = {}
    q3: dict[str, float | None] = {}
    metric_values: dict[str, list[dict[str, Any]]] = {}
    for metric in SUPPORTED_COMPARISON_METRICS:
        q1_value, q3_value = _quartiles_of(metric)
        q1[metric] = q1_value
        q3[metric] = q3_value
        metric_values[metric] = [
            {"ticker": str(peer["ticker"]), "value": float(peer[metric])}
            for peer in peer_metrics
            if isinstance(peer.get(metric), (int, float))
        ]

    result = {
        "schema_version": _CACHE_SCHEMA_VERSION,
        "sector": sector,
        "as_of_date": as_of_date,
        "peer_count": peer_count,
        "peer_tickers": [p["ticker"] for p in peer_metrics],
        "medians": {metric: _median_of(metric) for metric in SUPPORTED_COMPARISON_METRICS},
        "q1": q1,
        "q3": q3,
        "metric_values": metric_values,
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }

    # Write cache
    try:
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(json.dumps(result, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("peer_context: failed to write cache for %s: %s", sector, exc)

    return result


def _detect_sector(ticker: str) -> str | None:
    """Auto-detect which sector a ticker belongs to by checking cached peer lists."""
    cache_dir = Path("data/cache/sector_universe")
    if not cache_dir.exists():
        return None
    for cache_file in cache_dir.glob("*.json"):
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            rows = data.get("rows", [])
            tickers_in_sector = {
                str(r.get("ticker", "")).upper() for r in rows if isinstance(r, dict)
            }
            if ticker.upper() in tickers_in_sector:
                return cache_file.stem  # filename without .json = sector name
        except Exception:
            continue
    return None


def _percentile_rank(value: float | None, peer_values: list[dict[str, Any]]) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    vals = [
        float(row["value"]) for row in peer_values if isinstance(row.get("value"), (int, float))
    ]
    if not vals:
        return None
    below_or_equal = sum(1 for item in vals if item <= float(value))
    return round(below_or_equal / len(vals) * 100.0, 1)


def compute_peer_relative_metrics(
    ticker: str,
    as_of_date: str,
    *,
    cfg: Any | None = None,
    sector: str | None = None,
) -> dict[str, Any]:
    """Compare a ticker's metrics against its sector peer medians.

    Auto-detects sector from cached peer lists. Returns status, metrics,
    ratios, and relative position classification.
    """
    ticker_upper = ticker.upper()

    # Auto-detect sector
    sector = str(sector).strip() if sector else _detect_sector(ticker_upper)
    if sector is None:
        return {"status": "NO_SECTOR_MATCH", "sector": None, "peer_count": 0}

    # Get sector medians
    medians_result = compute_sector_medians(sector, as_of_date)
    peer_count = medians_result.get("peer_count", 0)
    if peer_count < _MIN_PEERS:
        return {"status": "INSUFFICIENT_PEERS", "sector": sector, "peer_count": peer_count}

    medians = medians_result.get("medians", {})
    q1 = medians_result.get("q1", {})
    q3 = medians_result.get("q3", {})
    metric_values = medians_result.get("metric_values", {})

    # Compute ticker's own metrics
    ticker_metrics = _compute_ticker_metrics(ticker_upper, as_of_date=as_of_date)

    # Compare: ratio = ticker / median
    relative_ratios: dict[str, float | None] = {}
    percentile_ranks: dict[str, float | None] = {}
    valid_ratios: list[float] = []

    for metric_key, ratio_key in _RELATIVE_RATIO_KEYS.items():
        tv = ticker_metrics.get(metric_key)
        mv = medians.get(metric_key)
        if isinstance(tv, (int, float)) and isinstance(mv, (int, float)) and mv != 0:
            ratio = float(tv) / float(mv)
            relative_ratios[ratio_key] = round(ratio, 2)
            if metric_key in {"roic", "operating_margin", "revenue_growth_5y"}:
                valid_ratios.append(ratio)
        else:
            relative_ratios[ratio_key] = None
        peer_values_for_metric = (
            metric_values.get(metric_key) if isinstance(metric_values, dict) else []
        )
        percentile_ranks[metric_key] = _percentile_rank(
            tv,
            peer_values_for_metric if isinstance(peer_values_for_metric, list) else [],
        )

    # Classify overall position from average ratio
    if valid_ratios:
        avg_ratio = sum(valid_ratios) / len(valid_ratios)
        if avg_ratio > 1.5:
            position = "LEADER"
        elif avg_ratio > 1.1:
            position = "ABOVE_AVERAGE"
        elif avg_ratio >= 0.9:
            position = "AVERAGE"
        elif avg_ratio >= 0.6:
            position = "BELOW_AVERAGE"
        else:
            position = "LAGGARD"
    else:
        position = "UNKNOWN"

    return {
        "status": "OK",
        "sector": sector,
        "peer_count": peer_count,
        "ticker_metrics": ticker_metrics,
        "sector_medians": medians,
        "sector_q1": q1 if isinstance(q1, dict) else {},
        "sector_q3": q3 if isinstance(q3, dict) else {},
        "relative_ratios": relative_ratios,
        "percentile_ranks": percentile_ranks,
        "peer_set_used": list(medians_result.get("peer_tickers") or []),
        "peer_metric_values": metric_values if isinstance(metric_values, dict) else {},
        "relative_position": position,
    }
