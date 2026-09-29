from __future__ import annotations

import logging
from statistics import mean
from typing import Any

from app.config import AppConfig, get_config
from app.util.issuer_classification import (
    BANK_LIKE_LINE_ITEMS,
    ISSUER_CLASS_FINANCIAL,
    resolve_issuer_classification,
)
from app.valuation.facts import resolve_financial_facts_asof
from app.valuation.intangible_economics import (
    UNKNOWN,
    _CAPEX_TAG_PRIORITY,
    _GROSS_PROFIT_TAG_PRIORITY,
    _RND_TAG_PRIORITY,
    _REVENUE_TAG_PRIORITY,
    _dedupe_refs,
    _is_num,
    _series_from_companyfacts,
)
from app.valuation.owner_earnings import _load_companyfacts_payload


ENTERPRISE_SOFTWARE = "ENTERPRISE_SOFTWARE"
SEMICONDUCTOR = "SEMICONDUCTOR"
CONSUMER_HARDWARE = "CONSUMER_HARDWARE"
INDUSTRIAL_TECH = "INDUSTRIAL_TECH"
NETWORK_INFRA = "NETWORK_INFRA"
PLATFORM_HYBRID = "PLATFORM_HYBRID"
TRADITIONAL_OPERATING = "TRADITIONAL_OPERATING"

logger = logging.getLogger(__name__)

_GENERIC_REVENUE_TAGS = {
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueNet",
    "SalesRevenueGoodsNet",
    "Revenues",
}
_REVENUE_TAG_HINTS = (
    "revenue",
    "sales",
    "subscription",
    "service",
    "product",
    "license",
    "advertising",
    "transaction",
    "marketplace",
)
_REVENUE_STREAM_KEYWORDS: dict[str, tuple[str, ...]] = {
    "subscription": ("subscription", "saas", "hosting", "cloud"),
    "services": ("service", "support", "maintenance", "consulting"),
    "licenses": ("license", "licensing", "royalty"),
    "products": ("product", "hardware", "device", "equipment", "appliance"),
    "advertising": ("advertising", "advertis", "marketingservices"),
    "payments": ("transaction", "marketplace", "merchant"),
}


def _resolve_companyfacts(
    *,
    ticker: str,
    as_of_date: str,
    facts_row: dict[str, Any] | None,
    companyfacts: dict[str, Any] | None,
    cfg: AppConfig | None,
) -> tuple[dict[str, Any], list[str]]:
    refs = [str(ref) for ref in (facts_row or {}).get("derived_from", []) if str(ref).strip()]
    if isinstance(companyfacts, dict) and isinstance(companyfacts.get("companyfacts"), dict):
        return companyfacts["companyfacts"], _dedupe_refs(refs)
    if isinstance(companyfacts, dict) and isinstance(companyfacts.get("facts"), dict):
        return companyfacts, _dedupe_refs(refs)

    cfg = cfg or get_config()
    resolved_row = facts_row if isinstance(facts_row, dict) else resolve_financial_facts_asof(
        ticker=str(ticker or "").strip().upper(),
        as_of_date=as_of_date,
        refresh=False,
        cfg=cfg,
    )
    loaded = _load_companyfacts_payload(resolved_row)
    refs = refs + [str(ref) for ref in resolved_row.get("derived_from", []) if str(ref).strip()]
    return loaded, _dedupe_refs(refs)


def _infer_companyfacts_issuer_classification(
    companyfacts: dict[str, Any],
    *,
    ticker: str | None = None,
    cik: object = None,
    cfg: AppConfig | None = None,
) -> str:
    """SIC-first (VOE_ISSUER_CLASSIFICATION_BY_SIC); the tag-name substring rule below
    answers only when no SIC code is on file for this registrant."""
    entity_name = str(companyfacts.get("entityName") or "").strip()
    facts = companyfacts.get("facts") if isinstance(companyfacts.get("facts"), dict) else {}
    line_items: list[str] = []
    raw_tag_texts: list[str] = [entity_name]
    for taxonomy in ("us-gaap", "dei"):
        node = facts.get(taxonomy) if isinstance(facts, dict) else {}
        if not isinstance(node, dict):
            continue
        tags = [str(tag) for tag in node.keys() if str(tag).strip()]
        line_items.extend(tags)
        raw_tag_texts.extend(tags)
    return resolve_issuer_classification(
        cik=cik or companyfacts.get("cik"),
        ticker=ticker,
        texts=raw_tag_texts,
        line_items=line_items,
        cfg=cfg,
    )[0]


def _companyfacts_line_item_set(companyfacts: dict[str, Any]) -> set[str]:
    facts = companyfacts.get("facts") if isinstance(companyfacts.get("facts"), dict) else {}
    out: set[str] = set()
    for taxonomy in ("us-gaap", "dei"):
        node = facts.get(taxonomy) if isinstance(facts, dict) else {}
        if not isinstance(node, dict):
            continue
        for tag in node.keys():
            normalized = str(tag or "").strip().lower()
            if normalized:
                out.add(normalized)
    return out


def _metric_series(
    *,
    companyfacts: dict[str, Any],
    as_of_date: str,
    priority: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    return _series_from_companyfacts(
        companyfacts=companyfacts,
        as_of_date=as_of_date,
        priority=priority,
        expected_unit_exact=("usd",),
    )


def _series_map(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        year = int(row.get("year") or 0)
        if year <= 0 or not _is_num(row.get("value")):
            continue
        out[year] = {
            "value": float(row["value"]),
            "derived_from": [str(ref) for ref in (row.get("derived_from") or []) if str(ref).strip()],
        }
    return out


def _recent_ratio_average(
    numerator: list[dict[str, Any]],
    denominator: list[dict[str, Any]],
    *,
    window: int = 3,
) -> tuple[float | str, int, list[str]]:
    denom_by_year = _series_map(denominator)
    ratios: list[tuple[int, float, list[str]]] = []
    for row in numerator:
        year = int(row.get("year") or 0)
        denom = denom_by_year.get(year)
        if year <= 0 or not isinstance(denom, dict):
            continue
        if not _is_num(row.get("value")) or not _is_num(denom.get("value")):
            continue
        denom_value = float(denom["value"])
        if denom_value <= 0.0:
            continue
        ratios.append(
            (
                year,
                float(row["value"]) / denom_value,
                _dedupe_refs(list(row.get("derived_from") or []) + list(denom.get("derived_from") or [])),
            )
        )
    ratios = sorted(ratios, key=lambda item: item[0])[-max(1, int(window)) :]
    if not ratios:
        return UNKNOWN, 0, []
    return round(mean(value for _year, value, _refs in ratios), 6), len(ratios), _dedupe_refs(
        [ref for _year, _value, refs in ratios for ref in refs]
    )


def _recent_gross_margin_average(
    gross_profit: list[dict[str, Any]],
    revenue: list[dict[str, Any]],
    *,
    window: int = 3,
) -> tuple[float | str, int, list[str]]:
    return _recent_ratio_average(gross_profit, revenue, window=window)


def _revenue_stream_count(companyfacts: dict[str, Any], as_of_date: str) -> tuple[int, list[str]]:
    facts = companyfacts.get("facts") if isinstance(companyfacts.get("facts"), dict) else {}
    us_gaap = facts.get("us-gaap") if isinstance(facts, dict) else {}
    if not isinstance(us_gaap, dict):
        return 0, []

    stream_hits: dict[str, list[str]] = {}
    asof_norm = str(as_of_date or "").strip()
    for tag, payload in us_gaap.items():
        tag_text = str(tag or "").strip()
        if not tag_text or tag_text in _GENERIC_REVENUE_TAGS:
            continue
        lowered = tag_text.lower()
        if not any(hint in lowered for hint in _REVENUE_TAG_HINTS):
            continue
        matched_streams = [
            stream_name
            for stream_name, keywords in _REVENUE_STREAM_KEYWORDS.items()
            if any(keyword in lowered for keyword in keywords)
        ]
        if not matched_streams:
            continue
        units = payload.get("units") if isinstance(payload, dict) else {}
        usd_rows = units.get("USD") if isinstance(units, dict) else None
        if not isinstance(usd_rows, list):
            continue
        valid_rows = [
            row for row in usd_rows
            if isinstance(row, dict)
            and _is_num(row.get("val"))
            and str(row.get("end") or "").strip()
            and str(row.get("end") or "") <= asof_norm
        ]
        if not valid_rows:
            continue
        ref = f"companyfacts.us-gaap.{tag_text}"
        for stream_name in matched_streams:
            stream_hits.setdefault(stream_name, []).append(ref)

    refs = _dedupe_refs([ref for hit_refs in stream_hits.values() for ref in hit_refs])
    return len(stream_hits), refs


def _classify_from_metrics(
    *,
    avg_rnd_to_revenue: float | str,
    avg_gross_margin: float | str,
    avg_capex_to_revenue: float | str,
    revenue_stream_count: int,
) -> str:
    rnd = float(avg_rnd_to_revenue) if _is_num(avg_rnd_to_revenue) else None
    gross_margin = float(avg_gross_margin) if _is_num(avg_gross_margin) else None
    capex = float(avg_capex_to_revenue) if _is_num(avg_capex_to_revenue) else None

    if rnd is None or gross_margin is None:
        return TRADITIONAL_OPERATING
    if rnd > 0.10 and gross_margin > 0.50 and revenue_stream_count >= 2:
        return PLATFORM_HYBRID
    if rnd > 0.15 and gross_margin > 0.65 and capex is not None and capex < 0.08:
        return ENTERPRISE_SOFTWARE
    if rnd > 0.12 and gross_margin > 0.45 and capex is not None and capex > 0.08:
        return SEMICONDUCTOR
    if rnd > 0.10 and gross_margin > 0.55:
        return NETWORK_INFRA
    if rnd > 0.05 and 0.30 <= gross_margin <= 0.55 and capex is not None and capex >= 0.08:
        return CONSUMER_HARDWARE
    if rnd > 0.03 and 0.35 <= gross_margin <= 0.55:
        return INDUSTRIAL_TECH
    return TRADITIONAL_OPERATING


def _decision_path_for_metrics(
    *,
    avg_rnd_to_revenue: float | str,
    avg_gross_margin: float | str,
    avg_capex_to_revenue: float | str,
    revenue_stream_count: int,
) -> str:
    rnd = float(avg_rnd_to_revenue) if _is_num(avg_rnd_to_revenue) else None
    gross_margin = float(avg_gross_margin) if _is_num(avg_gross_margin) else None
    capex = float(avg_capex_to_revenue) if _is_num(avg_capex_to_revenue) else None
    if rnd is None or gross_margin is None:
        return "missing_core_metrics"
    if rnd > 0.10 and gross_margin > 0.50 and revenue_stream_count >= 2:
        return "platform_hybrid_multi_stream"
    if rnd > 0.15 and gross_margin > 0.65 and capex is not None and capex < 0.08:
        return "enterprise_software_high_margin_low_capex"
    if rnd > 0.12 and gross_margin > 0.45 and capex is not None and capex > 0.08:
        return "semiconductor_high_capex"
    if rnd > 0.10 and gross_margin > 0.55:
        return "network_infra_high_margin"
    if rnd > 0.05 and 0.30 <= gross_margin <= 0.55 and capex is not None and capex >= 0.08:
        return "consumer_hardware_capex_heavy"
    if rnd > 0.03 and 0.35 <= gross_margin <= 0.55:
        return "industrial_tech_mid_margin"
    return "traditional_operating_fallback"


def _confidence_label(*, ratio_counts: list[int], revenue_stream_count: int, category: str) -> str:
    core_coverage = sum(1 for count in ratio_counts if count >= 3)
    if category == TRADITIONAL_OPERATING and core_coverage == 0:
        return "LOW"
    if core_coverage == len(ratio_counts) and (category != PLATFORM_HYBRID or revenue_stream_count >= 2):
        return "HIGH"
    if core_coverage >= 2:
        return "MEDIUM"
    return "LOW"


def classify_company_category(
    ticker: str,
    as_of_date: str,
    *,
    facts_row: dict[str, Any] | None = None,
    companyfacts: dict[str, Any] | None = None,
    cfg: AppConfig | None = None,
) -> dict[str, Any]:
    ticker_norm = str(ticker or "").strip().upper()
    resolved_facts_row = facts_row if isinstance(facts_row, dict) else None
    resolved_companyfacts, base_refs = _resolve_companyfacts(
        ticker=ticker_norm,
        as_of_date=as_of_date,
        facts_row=resolved_facts_row,
        companyfacts=companyfacts,
        cfg=cfg,
    )
    if resolved_facts_row is None:
        resolved_facts_row = resolve_financial_facts_asof(
            ticker=ticker_norm,
            as_of_date=as_of_date,
            refresh=False,
            cfg=cfg or get_config(),
        )
    if not resolved_companyfacts:
        return {
            "ticker": ticker_norm,
            "as_of_date": as_of_date,
            "category": TRADITIONAL_OPERATING,
            "confidence": "LOW",
            "metrics_used": {
                "avg_rnd_to_revenue_3y": UNKNOWN,
                "avg_gross_margin_3y": UNKNOWN,
                "avg_capex_to_revenue_3y": UNKNOWN,
                "revenue_stream_count": 0,
            },
            "derived_from": base_refs,
        }

    issuer_classification = _infer_companyfacts_issuer_classification(
        resolved_companyfacts,
        ticker=ticker_norm,
        cik=(resolved_facts_row or {}).get("cik"),
        cfg=cfg,
    )
    line_item_set = _companyfacts_line_item_set(resolved_companyfacts)
    rnd_rows = _metric_series(companyfacts=resolved_companyfacts, as_of_date=as_of_date, priority=_RND_TAG_PRIORITY)
    revenue_rows = _metric_series(companyfacts=resolved_companyfacts, as_of_date=as_of_date, priority=_REVENUE_TAG_PRIORITY)
    gross_profit_rows = _metric_series(companyfacts=resolved_companyfacts, as_of_date=as_of_date, priority=_GROSS_PROFIT_TAG_PRIORITY)
    capex_rows = _metric_series(companyfacts=resolved_companyfacts, as_of_date=as_of_date, priority=_CAPEX_TAG_PRIORITY)

    avg_rnd_to_revenue, rnd_count, rnd_refs = _recent_ratio_average(rnd_rows, revenue_rows, window=3)
    avg_gross_margin, gm_count, gm_refs = _recent_gross_margin_average(gross_profit_rows, revenue_rows, window=3)
    avg_capex_to_revenue, capex_count, capex_refs = _recent_ratio_average(capex_rows, revenue_rows, window=3)
    revenue_stream_count, stream_refs = _revenue_stream_count(resolved_companyfacts, as_of_date)

    cik = str((resolved_facts_row or {}).get("cik") or "").strip()
    cfg = cfg or get_config()
    override_map = {
        str(key or "").strip().lstrip("0"): str(value or "").strip().upper()
        for key, value in (cfg.tech_category_cik_overrides or {}).items()
        if str(key or "").strip()
    }
    override_category = str(override_map.get(cik.lstrip("0")) or "").strip().upper()
    decision_path = _decision_path_for_metrics(
        avg_rnd_to_revenue=avg_rnd_to_revenue,
        avg_gross_margin=avg_gross_margin,
        avg_capex_to_revenue=avg_capex_to_revenue,
        revenue_stream_count=revenue_stream_count,
    )

    software_escape_hatch = (
        issuer_classification == ISSUER_CLASS_FINANCIAL
        and not bool(line_item_set & BANK_LIKE_LINE_ITEMS)
        and _is_num(avg_rnd_to_revenue)
        and float(avg_rnd_to_revenue) >= 0.05
        and _is_num(avg_gross_margin)
        and float(avg_gross_margin) >= 0.50
    )

    if override_category:
        category = override_category
        decision_path = f"cik_override:{cik}"
    elif issuer_classification == ISSUER_CLASS_FINANCIAL and not software_escape_hatch:
        category = TRADITIONAL_OPERATING
        decision_path = "issuer_classification_financial_gate"
    else:
        category = _classify_from_metrics(
            avg_rnd_to_revenue=avg_rnd_to_revenue,
            avg_gross_margin=avg_gross_margin,
            avg_capex_to_revenue=avg_capex_to_revenue,
            revenue_stream_count=revenue_stream_count,
        )
        if software_escape_hatch:
            decision_path = f"financial_escape_hatch->{decision_path}"
    confidence = _confidence_label(
        ratio_counts=[rnd_count, gm_count, capex_count],
        revenue_stream_count=revenue_stream_count,
        category=category,
    )

    if ticker_norm == "ORCL":
        logger.info(
            "tech_category debug %s cik=%s issuer=%s rnd=%s gm=%s capex=%s streams=%s path=%s category=%s override=%s",
            ticker_norm,
            cik,
            issuer_classification,
            avg_rnd_to_revenue,
            avg_gross_margin,
            avg_capex_to_revenue,
            revenue_stream_count,
            decision_path,
            category,
            bool(override_category),
        )

    return {
        "ticker": ticker_norm,
        "as_of_date": as_of_date,
        "category": category,
        "confidence": confidence,
        "metrics_used": {
            "avg_rnd_to_revenue_3y": avg_rnd_to_revenue,
            "avg_gross_margin_3y": avg_gross_margin,
            "avg_capex_to_revenue_3y": avg_capex_to_revenue,
            "revenue_stream_count": revenue_stream_count,
            "years_used": {
                "rnd_to_revenue": rnd_count,
                "gross_margin": gm_count,
                "capex_to_revenue": capex_count,
            },
            "issuer_classification": issuer_classification,
            "decision_path": decision_path,
            "cik_override_applied": bool(override_category),
        },
        "derived_from": _dedupe_refs(
            base_refs
            + rnd_refs
            + gm_refs
            + capex_refs
            + stream_refs
            + ([f"tech_category_override:{cik}"] if override_category else [])
            + (["issuer_classification:financial"] if issuer_classification == ISSUER_CLASS_FINANCIAL else [])
        ),
    }
