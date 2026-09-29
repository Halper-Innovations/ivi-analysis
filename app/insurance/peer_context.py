"""Insurance subtype-aware peer comparisons.

The generic peer context compares a company against the broad inferred sector.
For insurers that can be misleading: mortgage insurers, life writers, P&C
underwriters, and reinsurers have different economics. This module first tries
to build an exact insurance-subtype cohort and only falls back to broad-sector
peers when the subtype set is too thin.
"""
from __future__ import annotations

from functools import lru_cache
from statistics import median
from typing import Any

from app.insurance.packet import load_latest_insurance_packet
from app.insurance.routing import ISSUER_INSURANCE_UNDERWRITER, route_security
from app.valuation.peer_context import _compute_ticker_metrics, _load_sector_tickers, compute_peer_relative_metrics

_MIN_SUBTYPE_PEERS = 3


def _position_from_ratios(ratios: list[float]) -> str:
    if not ratios:
        return "UNKNOWN"
    avg_ratio = sum(ratios) / len(ratios)
    if avg_ratio > 1.5:
        return "LEADER"
    if avg_ratio > 1.1:
        return "ABOVE_AVERAGE"
    if avg_ratio >= 0.9:
        return "AVERAGE"
    if avg_ratio >= 0.6:
        return "BELOW_AVERAGE"
    return "LAGGARD"


@lru_cache(maxsize=1024)
def _insurance_subtype_for(ticker: str, as_of_date: str) -> str | None:
    packet = load_latest_insurance_packet(ticker)
    routing = packet.get("routing") if isinstance(packet.get("routing"), dict) else {}
    if routing.get("issuer_type") == ISSUER_INSURANCE_UNDERWRITER and routing.get("insurance_subtype"):
        return str(routing["insurance_subtype"])
    try:
        routed = route_security(ticker, as_of_date=as_of_date)
    except Exception:
        return None
    if routed.issuer_type != ISSUER_INSURANCE_UNDERWRITER:
        return None
    return routed.insurance_subtype


@lru_cache(maxsize=1024)
def _ticker_metrics_for(ticker: str, as_of_date: str) -> tuple[float | None, float | None, float | None]:
    metrics = _compute_ticker_metrics(ticker, as_of_date=as_of_date)
    return (
        metrics.get("roic"),
        metrics.get("operating_margin"),
        metrics.get("revenue_growth_5y"),
    )


def _metrics_dict(ticker: str, as_of_date: str) -> dict[str, float | None]:
    roic, operating_margin, revenue_growth = _ticker_metrics_for(ticker.upper(), as_of_date)
    return {
        "roic": roic,
        "operating_margin": operating_margin,
        "revenue_growth_5y": revenue_growth,
    }


def _median_of(peer_metrics: list[dict[str, Any]], key: str) -> float | None:
    vals = [item[key] for item in peer_metrics if isinstance(item.get(key), (int, float))]
    return round(median(vals), 4) if len(vals) >= _MIN_SUBTYPE_PEERS else None


def _broad_sector_fallback(ticker: str, as_of_date: str, *, target_subtype: str | None, reason: str) -> dict[str, Any]:
    broad = compute_peer_relative_metrics(ticker, as_of_date)
    broad["peer_scope"] = "broad_sector_fallback"
    broad["peer_group"] = "insurance"
    broad["insurance_subtype"] = target_subtype
    broad["fallback_reason"] = reason
    return broad


def compute_insurance_subtype_peer_relative_metrics(
    ticker: str,
    as_of_date: str,
    *,
    min_peers: int = _MIN_SUBTYPE_PEERS,
    fallback_to_sector: bool = True,
) -> dict[str, Any]:
    """Compare an insurer against exact-subtype peers when enough data exists."""
    ticker_upper = ticker.upper()
    target_subtype = _insurance_subtype_for(ticker_upper, as_of_date)
    if not target_subtype:
        if fallback_to_sector:
            return _broad_sector_fallback(
                ticker_upper,
                as_of_date,
                target_subtype=None,
                reason="insurance_subtype_unavailable",
            )
        return {
            "status": "NO_INSURANCE_SUBTYPE",
            "sector": "insurance",
            "peer_scope": "insurance_subtype",
            "peer_group": None,
            "insurance_subtype": None,
            "peer_count": 0,
        }

    peer_metrics: list[dict[str, Any]] = []
    for peer in _load_sector_tickers("insurance"):
        peer_upper = str(peer or "").upper()
        if not peer_upper or peer_upper == ticker_upper:
            continue
        if _insurance_subtype_for(peer_upper, as_of_date) != target_subtype:
            continue
        metrics = _metrics_dict(peer_upper, as_of_date)
        if any(value is not None for value in metrics.values()):
            peer_metrics.append({"ticker": peer_upper, **metrics})

    peer_count = len(peer_metrics)
    if peer_count < min_peers:
        if fallback_to_sector:
            return _broad_sector_fallback(
                ticker_upper,
                as_of_date,
                target_subtype=target_subtype,
                reason="insufficient_subtype_peers",
            )
        return {
            "status": "INSUFFICIENT_SUBTYPE_PEERS",
            "sector": "insurance",
            "peer_scope": "insurance_subtype",
            "peer_group": f"insurance:{target_subtype}",
            "insurance_subtype": target_subtype,
            "peer_count": peer_count,
            "peer_tickers": [item["ticker"] for item in peer_metrics],
            "sector_medians": {"roic": None, "operating_margin": None, "revenue_growth_5y": None},
        }

    medians = {
        "roic": _median_of(peer_metrics, "roic"),
        "operating_margin": _median_of(peer_metrics, "operating_margin"),
        "revenue_growth_5y": _median_of(peer_metrics, "revenue_growth_5y"),
    }
    ticker_metrics = _metrics_dict(ticker_upper, as_of_date)
    relative_ratios: dict[str, float | None] = {}
    valid_ratios: list[float] = []
    for metric_key, ratio_key in [
        ("roic", "roic_vs_median"),
        ("operating_margin", "operating_margin_vs_median"),
        ("revenue_growth_5y", "revenue_growth_vs_median"),
    ]:
        tv = ticker_metrics.get(metric_key)
        mv = medians.get(metric_key)
        if isinstance(tv, (int, float)) and isinstance(mv, (int, float)) and mv != 0:
            ratio = float(tv) / float(mv)
            relative_ratios[ratio_key] = round(ratio, 2)
            valid_ratios.append(ratio)
        else:
            relative_ratios[ratio_key] = None

    return {
        "status": "OK",
        "sector": "insurance",
        "peer_scope": "insurance_subtype",
        "peer_group": f"insurance:{target_subtype}",
        "insurance_subtype": target_subtype,
        "peer_count": peer_count,
        "peer_tickers": [item["ticker"] for item in peer_metrics],
        "ticker_metrics": ticker_metrics,
        "sector_medians": medians,
        "relative_ratios": relative_ratios,
        "relative_position": _position_from_ratios(valid_ratios),
    }
