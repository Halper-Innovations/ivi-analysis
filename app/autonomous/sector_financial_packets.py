"""Deterministic builders for sector-level autonomous financial packets."""

from __future__ import annotations

import json
import math
import re
from contextvars import ContextVar
from dataclasses import replace
from datetime import date
from functools import lru_cache
from typing import Any

from app.alpha.llm_tools import _shares_cagr, _split_adjusted_share_series
from app.alpha.schemas import TickerSignalPacket
from app.alpha.signal_assembler import assemble_sector_packets
from app.alpha.solvency_scanner import going_concern_asserted
from app.autonomous.financial_integrity import (
    MARKET_CAP_UNIT_USD_MILLIONS,
    PRICE_UNIT_USD_PER_SHARE,
    SHARES_BASIS_UNADJUSTED,
    canonical_metric_trace,
    stable_quote_hash,
)
from app.autonomous.sector_contract import SectorCompanyFinancialPacket
from app.util.financial_data_access import (
    ANNUAL_COMPANYFACTS_PERIOD_TYPES,
    companyfacts_rows,
    issuer_companyfacts_rows,
    latest_companyfacts_value,
)
from app.util.issuer_classification import (
    ISSUER_CLASS_FINANCIAL,
    infer_issuer_classification,
    resolve_issuer_classification,
)
from app.valuation.anchor_policy import is_decline_class, select_anchor
from app.valuation.peer_context import (
    PEER_METRIC_DISPLAY_NAMES,
    compute_peer_relative_metrics,
    normalize_peer_metric,
)
from app.valuation.provenance import validate_v2_valuation_method_provenance
from app.valuation.lineage import latest_decision_eligible_valuation_row


GENERIC_VALUATION_METHODS = ("dcf", "epv", "graham", "ncav")
TECHNOLOGY_ADJUSTED_DCF_METHOD = "technology_adjusted_dcf"
ROIC_WACC_DEFAULT = 0.10
ROIC_WACC_DEFAULT_REASON = "CONSERVATIVE_SECTOR_AGNOSTIC_DEFAULT_TUNE_LATER"
ROIC_INVESTED_CAPITAL_BASIS = "with_goodwill"
ROIC_TAX_LINE_ITEMS = ("income_tax_expense",)
ROIC_PRETAX_LINE_ITEMS = ("pretax_income",)
GROSS_MARGIN_COST_LINE_ITEMS = (
    "cost_of_revenue",
    "cost_of_goods_sold",
    "cost_of_goods_and_services_sold",
)
RAW_COMPANYFACTS_FALLBACK_LINE_ITEMS = (
    "income_continuing",
    "retained_earnings",
    "sga",
    "income_tax_expense",
    "pretax_income",
    "cost_of_revenue",
)
HISTORICAL_MULTIPLE_KEYS = ("pe", "ev_to_ebitda", "price_to_book", "fcf_yield")
_V2_ISSUER_FACT_CONTEXT: ContextVar[dict[str, Any] | None] = ContextVar(
    "v2_sector_packet_issuer_fact_context",
    default=None,
)
HISTORICAL_MULTIPLE_LABELS = {
    "pe": "P/E",
    "ev_to_ebitda": "EV/EBITDA",
    "price_to_book": "P/B",
    "fcf_yield": "FCF yield",
}
HISTORICAL_MULTIPLE_PRICE_FALLBACK_DAYS = 7
HISTORICAL_MULTIPLE_PRICE_TIMEOUT_SECONDS = 10.0
HISTORICAL_MULTIPLE_PRICE_MAX_RETRIES = 1


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_finite_num(value: Any) -> bool:
    if not _is_num(value):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _optional_float(value: Any) -> float | None:
    return float(value) if _is_num(value) else None


def _raw_scorecard(packet: TickerSignalPacket) -> dict[str, Any]:
    return packet.raw_valuation if isinstance(packet.raw_valuation, dict) else {}


def _quality_context(packet: TickerSignalPacket) -> dict[str, Any]:
    if isinstance(packet.raw_quality_ctx, dict) and packet.raw_quality_ctx:
        return packet.raw_quality_ctx
    scorecard = _raw_scorecard(packet)
    qctx = scorecard.get("quality_context")
    return qctx if isinstance(qctx, dict) else {}


def _pricing_zone_detail(packet: TickerSignalPacket) -> dict[str, Any]:
    scorecard = _raw_scorecard(packet)
    pzd = scorecard.get("pricing_zone_detail")
    return pzd if isinstance(pzd, dict) else {}


def _impairment_context(packet: TickerSignalPacket) -> dict[str, Any]:
    scorecard = _raw_scorecard(packet)
    detail = scorecard.get("impairment_classification_detail")
    if not isinstance(detail, dict):
        detail = scorecard.get("business_impairment_classification")
    if not isinstance(detail, dict) and scorecard.get("impairment_class_primary"):
        detail = scorecard
    if not isinstance(detail, dict):
        return {}

    payload: dict[str, Any] = {}
    for key in (
        "impairment_class_primary",
        "primary_underwriting_caution",
        "impairment_class_reason_codes",
        "impairment_support_signals",
        "impairment_rebuttal_signals",
    ):
        value = detail.get(key)
        if value is not None:
            payload[key] = value
    return payload


def _solvency_payload(packet: TickerSignalPacket) -> dict[str, Any]:
    research = packet.research_report if isinstance(packet.research_report, dict) else {}
    solvency = research.get("solvency")
    return solvency if isinstance(solvency, dict) else {}


def _generic_valuation_anchor(packet: TickerSignalPacket) -> tuple[str | None, float | None]:
    # Canonical rule via the shared policy (audit: anchor-rule-backtest-vs-
    # live-divergence): max of POSITIVE (dcf, epv), graham->ncav fallback —
    # NOT first-numeric (which let a negative DCF block a positive EPV and
    # under-anchored every name where EPV > DCF).
    selection = select_anchor(
        dcf=packet.dcf_value if _is_num(packet.dcf_value) else None,
        epv=packet.epv_value if _is_num(packet.epv_value) else None,
        graham=packet.graham_value if _is_num(packet.graham_value) else None,
        ncav=packet.ncav_value if _is_num(packet.ncav_value) else None,
    )
    return selection.method, selection.value


def _technology_adjusted_anchor(packet: TickerSignalPacket) -> tuple[str | None, float | None]:
    # ONE trigger predicate with the backtest (review ANCHOR-1): the leg fires
    # whenever the writer persisted OK/positive divergence diagnostics — the
    # fundamentals-based category classification already decided whether the
    # diagnostics exist. The old sweep-sector-token gate made the deployed
    # rule diverge from the measured rule on every diag-OK name whose sweep
    # sector lacked a tech token.
    if str(packet.pricing_zone or "") == "VALUATION_ANOMALY":
        # Anomaly names never anchor: signal_assembler nulls the generic
        # methods and the backtest excludes the rows; the diagnostics bypass
        # must not re-anchor them (review ANCHOR-3).
        return None, None
    diagnostics = _raw_scorecard(packet).get("tech_valuation_divergence_diagnostics")
    if not isinstance(diagnostics, dict):
        return None, None
    if diagnostics.get("status") not in {None, "OK"}:
        return None, None
    adjusted_anchor = _optional_float(diagnostics.get("adjusted_anchor"))
    if adjusted_anchor is None or adjusted_anchor <= 0:
        return None, None
    return TECHNOLOGY_ADJUSTED_DCF_METHOD, adjusted_anchor


def _sector_specific_valuation_anchor(
    packet: TickerSignalPacket, *, sector: str | None
) -> tuple[str | None, float | None]:
    # Positivity required: a negative sector-specific value must not anchor
    # AND must not report VALID_SECTOR_SPECIFIC while select_anchor skips it
    #.
    if _is_num(packet.insurance_value) and float(packet.insurance_value) > 0:
        return packet.insurance_method or "insurance", float(packet.insurance_value)
    return _technology_adjusted_anchor(packet)


def _valuation_anchor(
    packet: TickerSignalPacket, *, sector: str | None
) -> tuple[str | None, float | None]:
    sector_method, sector_anchor = _sector_specific_valuation_anchor(packet, sector=sector)
    selection = select_anchor(
        dcf=packet.dcf_value if _is_num(packet.dcf_value) else None,
        epv=packet.epv_value if _is_num(packet.epv_value) else None,
        graham=packet.graham_value if _is_num(packet.graham_value) else None,
        ncav=packet.ncav_value if _is_num(packet.ncav_value) else None,
        sector_specific=(
            (sector_method, float(sector_anchor))
            if sector_method is not None and _is_num(sector_anchor)
            else None
        ),
        # Decline-cap policy: shrinking businesses anchor on the no-growth
        # basis (ONE predicate across packets/backtest/render via
        # is_decline_class).
        decline_class=is_decline_class(_quality_context(packet).get("revenue_trend_class")),
    )
    return selection.method, selection.value


def _available_valuation_methods(
    packet: TickerSignalPacket, *, sector: str | None = None
) -> list[str]:
    methods: list[str] = []
    # Positive values only: select_anchor can never choose a negative-valued
    # method, so listing one as "available" misstates the selection rule
    # (review ANCHOR-8).
    if _is_num(packet.insurance_value) and float(packet.insurance_value) > 0:
        methods.append(packet.insurance_method or "insurance")
    tech_method, tech_anchor = _technology_adjusted_anchor(packet)
    if tech_method and tech_anchor is not None:
        methods.append(tech_method)
    for method, value in [
        ("dcf", packet.dcf_value),
        ("epv", packet.epv_value),
        ("graham", packet.graham_value),
        ("ncav", packet.ncav_value),
    ]:
        if _is_num(value) and float(value) > 0:
            methods.append(method)
    return methods


def _discount_to_anchor(current_price: float | None, anchor: float | None) -> float | None:
    if current_price is None or anchor is None or anchor <= 0:
        return None
    return (anchor - current_price) / anchor


def _model_fit_status(packet: TickerSignalPacket, *, sector: str | None) -> str:
    if packet.model_status == "MODEL_BLOCKED" or packet.model_blockers:
        return "BLOCKED"
    _sector_method, sector_anchor = _sector_specific_valuation_anchor(packet, sector=sector)
    # VALID_SECTOR_SPECIFIC iff the sector leg would actually anchor, or the
    # name is insurance-routed (packet present). A bare negative
    # insurance_value is skipped by the selection rule and must not carry the
    # label.
    if packet.insurance_packet or sector_anchor is not None:
        return "VALID_SECTOR_SPECIFIC"
    if _available_valuation_methods(packet, sector=sector):
        return "VALID_GENERIC"
    return "UNKNOWN"


def _data_quality_status(packet: TickerSignalPacket, anchor: float | None) -> str:
    if packet.current_price is None:
        return "MISSING_PRICE"
    if anchor is None:
        return "MISSING_VALUATION"
    if packet.model_status == "MODEL_BLOCKED":
        return "MODEL_BLOCKED"
    filing_evidence_status = (
        packet.filing_risk_metadata.get("evidence_status")
        if isinstance(packet.filing_risk_metadata, dict)
        else None
    )
    if (
        packet.filing_risk_status == "NO_FILING"
        and filing_evidence_status == "RISK_SECTION_NOT_FOUND"
    ):
        return "RISK_SECTION_NOT_FOUND"
    if (
        packet.filing_risk_status in {"OK", "KEYWORD_FALLBACK"}
        and filing_evidence_status == "STALE_READABLE_RISK_SECTION"
    ):
        return "STALE_FILING_RISK"
    if packet.filing_risk_status in {"ERROR", "NO_FILING"}:
        return str(packet.filing_risk_status)
    return "OK"


def _blockers(packet: TickerSignalPacket, anchor: float | None) -> list[str]:
    blockers: list[str] = []
    if packet.current_price is None:
        blockers.append("MISSING_PRICE")
    if anchor is None:
        blockers.append("MISSING_VALUATION")
    if packet.gate_verdict == "BLOCK":
        blockers.append("GATE_BLOCK")
    if packet.solvency_risk == "CRITICAL":
        blockers.append("SOLVENCY_CRITICAL")
    blockers.extend(str(item) for item in (packet.model_blockers or []))
    blockers.extend(str(item) for item in (packet.valuation_provenance_blockers or []))
    return list(dict.fromkeys(blockers))


def _confidence_caps(packet: TickerSignalPacket) -> list[str]:
    caps: list[str] = []
    filing_evidence_status = (
        packet.filing_risk_metadata.get("evidence_status")
        if isinstance(packet.filing_risk_metadata, dict)
        else None
    )
    if (
        packet.filing_risk_status == "NO_FILING"
        and filing_evidence_status == "RISK_SECTION_NOT_FOUND"
    ):
        caps.append("FILING_RISK_SECTION_NOT_FOUND")
    elif (
        packet.filing_risk_status in {"OK", "KEYWORD_FALLBACK"}
        and filing_evidence_status == "STALE_READABLE_RISK_SECTION"
    ):
        caps.append("FILING_RISK_STALE_ANNUAL_FILING")
    elif packet.filing_risk_status in {"ERROR", "NO_FILING", "KEYWORD_FALLBACK"}:
        caps.append(f"FILING_RISK_{packet.filing_risk_status}")
    if packet.anomaly_count > 0:
        caps.append("FINANCIAL_ANOMALIES_PRESENT")
    if packet.method_tension_type and packet.method_tension_type not in {"NONE", "UNKNOWN"}:
        caps.append(f"METHOD_TENSION_{packet.method_tension_type}")
    if _is_num(packet.growth_dependency_ratio) and float(packet.growth_dependency_ratio) >= 0.70:
        caps.append("HIGH_GROWTH_DEPENDENCY")
    if not packet.quarterly_revenue_trend or packet.quarterly_revenue_trend == "UNKNOWN":
        caps.append("QUARTERLY_REVENUE_TREND_UNKNOWN")
    caps.extend(f"MODEL_WARNING_{item}" for item in (packet.model_fit_warnings or []))
    return list(dict.fromkeys(str(item) for item in caps))


def _financial_status(blockers: list[str], caps: list[str]) -> str:
    blocker_set = set(blockers)
    if {"MISSING_PRICE", "MISSING_VALUATION"} & blocker_set:
        return "Data Insufficient"
    if {"GATE_BLOCK"} & blocker_set:
        return "Blocked"
    if {"SOLVENCY_CRITICAL"} & blocker_set:
        return "Balance-Sheet Constrained"
    if blocker_set:
        return "Model-Fit Constrained"
    if any(str(cap).startswith("FILING_RISK") for cap in caps):
        return "Financially Viable With Evidence Caps"
    return "Financially Viable"


def _business_quality(
    packet: TickerSignalPacket,
    *,
    as_of_date: str | None = None,
    v2_data_plane: bool = False,
) -> dict[str, Any]:
    qctx = _quality_context(packet)
    impairment = _impairment_context(packet)
    gross_margin = _gross_margin_metrics(
        packet.ticker,
        as_of_date=as_of_date,
        v2_data_plane=v2_data_plane,
    )
    normalized_operating_margin = _normalized_operating_margin_metrics(
        packet.ticker,
        as_of_date=as_of_date,
        v2_data_plane=v2_data_plane,
    )
    forensic_scores = _forensic_quality_scores(
        packet.ticker,
        as_of_date=as_of_date,
        current_price=_optional_float(packet.current_price),
        v2_data_plane=v2_data_plane,
    )
    payload = {
        "gate_verdict": packet.gate_verdict,
        "confidence_class": packet.confidence_class,
        "moat_score": packet.moat_score,
        "moat_classification": packet.moat_classification,
        "downside_risk_class": packet.downside_risk_class,
        "revenue_cagr_5y": qctx.get("revenue_cagr_5y"),
        "revenue_cagr_3y": qctx.get("revenue_cagr_3y"),
        "earnings_quality": qctx.get("earnings_quality"),
        "valuation_supports": list(packet.valuation_supports or []),
        "valuation_headwinds": list(packet.valuation_headwinds or []),
        "quarterly_revenue_trend": packet.quarterly_revenue_trend,
        "peer_position": packet.peer_position,
        **gross_margin,
        **normalized_operating_margin,
        **forensic_scores,
    }
    if impairment:
        payload["impairment_classification"] = impairment
        payload["impairment_class_primary"] = impairment.get("impairment_class_primary")
        payload["primary_underwriting_caution"] = impairment.get("primary_underwriting_caution")
    return payload


def _reinvestment(packet: TickerSignalPacket) -> dict[str, Any]:
    qctx = _quality_context(packet)
    return {
        "revenue_cagr_5y": qctx.get("revenue_cagr_5y"),
        "revenue_cagr_3y": qctx.get("revenue_cagr_3y"),
        "growth_dependency_ratio": packet.growth_dependency_ratio,
        "growth_dependency_status": (
            "HIGH"
            if _is_num(packet.growth_dependency_ratio)
            and float(packet.growth_dependency_ratio) >= 0.70
            else "MODERATE_OR_LOW"
            if _is_num(packet.growth_dependency_ratio)
            else "UNKNOWN"
        ),
    }


@lru_cache(maxsize=512)
def _cached_companyfacts_cik(ticker: str, as_of_date: str | None) -> str | None:
    try:
        from app.db import get_db

        params: list[str] = [ticker.upper()]
        clauses = [
            "ticker = ?",
            "source_url IS NOT NULL",
            "source_url LIKE '%CIK%.json'",
        ]
        if as_of_date:
            clauses.append("period_end <= ?")
            params.append(str(as_of_date))
        with get_db() as conn:
            row = conn.execute(
                f"""
                SELECT source_url
                FROM companyfacts_facts
                WHERE {" AND ".join(clauses)}
                ORDER BY COALESCE(period_end, '1900-01-01') DESC, id DESC
                LIMIT 1
                """,
                params,
            ).fetchone()
    except Exception:
        row = None
    source_url = str(row["source_url"] or "") if row else ""
    match = re.search(r"CIK(\d{10})\.json", source_url)
    return match.group(1) if match else None


@lru_cache(maxsize=512)
def _raw_companyfacts_annual_rows(
    ticker: str,
    as_of_date: str | None,
    line_items: tuple[str, ...],
) -> tuple[tuple[int, str, float], ...]:
    cik = _cached_companyfacts_cik(ticker, as_of_date)
    if not cik:
        return ()
    try:
        from app.config import get_config
        from app.ingest.companyfacts import normalize_annual_facts_from_raw

        cache_path = get_config().cache_dir / "companyfacts" / f"{cik}.json"
        payload = json.loads(cache_path.read_text())
    except Exception:
        return ()
    raw = (
        payload.get("companyfacts")
        if isinstance(payload, dict) and isinstance(payload.get("companyfacts"), dict)
        else payload
    )
    if not isinstance(raw, dict):
        return ()
    requested = set(line_items)
    rows: list[tuple[int, str, float]] = []
    try:
        normalized = normalize_annual_facts_from_raw(raw, cik=cik, years_back=15)
    except Exception:
        return ()
    for row in normalized:
        line_item = str(row.get("line_item") or "")
        fiscal_year = row.get("fiscal_year")
        value = row.get("value")
        period_end = str(row.get("period_end") or "")
        if line_item not in requested:
            continue
        if as_of_date and period_end and period_end > str(as_of_date):
            continue
        if isinstance(fiscal_year, int) and _is_num(value):
            rows.append((fiscal_year, line_item, float(value)))
    return tuple(rows)


def _annual_fact_rows(
    ticker: str,
    *,
    as_of_date: str | None,
    issuer_aware: bool = False,
    issuer_cik: str | None = None,
    aliases: tuple[str, ...] = (),
    require_filed_asof: bool = True,
) -> dict[int, dict[str, float]]:
    values, _provenance = _annual_fact_records(
        ticker,
        as_of_date=as_of_date,
        issuer_aware=issuer_aware,
        issuer_cik=issuer_cik,
        aliases=aliases,
        require_filed_asof=require_filed_asof,
    )
    return values


def _annual_fact_records(
    ticker: str,
    *,
    as_of_date: str | None,
    issuer_aware: bool = False,
    issuer_cik: str | None = None,
    aliases: tuple[str, ...] = (),
    require_filed_asof: bool = True,
) -> tuple[dict[int, dict[str, float]], dict[int, dict[str, dict[str, Any]]]]:
    """Return point-in-time annual values and their exact filing provenance.

    Active packet builders always set ``require_filed_asof``.  Keeping the
    legacy opt-out explicit preserves archaeology helpers without allowing a
    period-end-only row or raw-cache fallback into a current decision packet.
    """

    line_items = (
        "revenue",
        "gross_profit",
        *GROSS_MARGIN_COST_LINE_ITEMS,
        "operating_income",
        "net_income",
        "income_continuing",
        "total_debt",
        "equity",
        "cash",
        "total_assets",
        # Bank-defining line items so infer_issuer_classification can detect
        # a financial issuer ({deposits, loans} -> FINANCIAL) from the fetched
        # values; without these the financial equity-basis ROIC and gross-margin
        # suppression were dead code on every real ticker.
        "deposits",
        "loans",
        "provision_for_credit_losses",
        "current_assets",
        "current_liabilities",
        "total_liabilities",
        "gross_ppe",
        "retained_earnings",
        "cfo",
        "capex",
        "depreciation",
        "depreciation_amortization",
        "intangible_amortization",
        "sga",
        "accounts_receivable",
        "inventory",
        "accounts_payable",
        "shares_outstanding",
        *ROIC_TAX_LINE_ITEMS,
        *ROIC_PRETAX_LINE_ITEMS,
    )
    try:
        from app.db import get_db

        with get_db() as conn:
            if issuer_aware:
                _scope, rows = issuer_companyfacts_rows(
                    conn,
                    ticker,
                    columns=(
                        "fiscal_year",
                        "line_item",
                        "value",
                        "units",
                        "period_end",
                        "filed_date",
                        "source_url",
                        "accession",
                    ),
                    issuer_cik=issuer_cik,
                    aliases=aliases,
                    period_types=ANNUAL_COMPANYFACTS_PERIOD_TYPES,
                    line_items=line_items,
                    as_of_date=as_of_date,
                    value_not_null=True,
                    require_filed_asof=require_filed_asof,
                    order_by="fiscal_year ASC, line_item ASC",
                )
            else:
                rows = companyfacts_rows(
                    conn,
                    ticker,
                    columns=(
                        "fiscal_year",
                        "line_item",
                        "value",
                        "units",
                        "period_end",
                        "filed_date",
                        "source_url",
                        "accession",
                    ),
                    period_types=ANNUAL_COMPANYFACTS_PERIOD_TYPES,
                    line_items=line_items,
                    as_of_date=as_of_date,
                    value_not_null=True,
                    require_filed_asof=require_filed_asof,
                    order_by="fiscal_year ASC, line_item ASC",
                )
    except Exception:
        return {}, {}
    by_year: dict[int, dict[str, float]] = {}
    provenance_by_year: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        fiscal_year = row["fiscal_year"]
        line_item = row["line_item"]
        value = row["value"]
        if isinstance(fiscal_year, int) and isinstance(line_item, str) and _is_num(value):
            by_year.setdefault(fiscal_year, {})[line_item] = float(value)
            source_url = str(row["source_url"] or "").strip()
            accession = str(row["accession"] or "").strip()
            provenance_by_year.setdefault(fiscal_year, {})[line_item] = {
                "value": float(value),
                "unit": str(row["units"] or "").strip(),
                "source": "SEC_COMPANYFACTS",
                "period_end": str(row["period_end"] or "").strip(),
                "filed_date": str(row["filed_date"] or "").strip(),
                "source_reference": source_url
                or (f"SEC_ACCESSION:{accession}" if accession else ""),
                "source_url": source_url or None,
                "accession": accession or None,
            }
    # Raw cached CompanyFacts cannot prove filing visibility or a stable source
    # reference. It remains available only to an explicitly legacy caller.
    if not require_filed_asof:
        for fiscal_year, line_item, value in _raw_companyfacts_annual_rows(
            ticker.upper(),
            str(as_of_date) if as_of_date else None,
            RAW_COMPANYFACTS_FALLBACK_LINE_ITEMS,
        ):
            by_year.setdefault(fiscal_year, {}).setdefault(line_item, value)
    return by_year, provenance_by_year


def _annual_period_ends(
    ticker: str,
    *,
    as_of_date: str | None,
    issuer_aware: bool = False,
    issuer_cik: str | None = None,
    aliases: tuple[str, ...] = (),
    require_filed_asof: bool = False,
) -> dict[int, str]:
    try:
        from app.db import get_db

        with get_db() as conn:
            if issuer_aware:
                _scope, rows = issuer_companyfacts_rows(
                    conn,
                    ticker,
                    columns=("fiscal_year", "period_end"),
                    issuer_cik=issuer_cik,
                    aliases=aliases,
                    period_types=ANNUAL_COMPANYFACTS_PERIOD_TYPES,
                    as_of_date=as_of_date,
                    value_not_null=True,
                    require_filed_asof=require_filed_asof,
                    order_by="fiscal_year ASC, period_end DESC",
                )
            else:
                rows = companyfacts_rows(
                    conn,
                    ticker,
                    columns=("fiscal_year", "period_end"),
                    period_types=ANNUAL_COMPANYFACTS_PERIOD_TYPES,
                    as_of_date=as_of_date,
                    value_not_null=True,
                    require_filed_asof=require_filed_asof,
                    order_by="fiscal_year ASC, period_end DESC",
                )
    except Exception:
        return {}
    period_ends: dict[int, str] = {}
    for row in rows:
        fiscal_year = row["fiscal_year"]
        period_end = str(row["period_end"] or "").strip()
        if isinstance(fiscal_year, int) and period_end and fiscal_year not in period_ends:
            period_ends[fiscal_year] = period_end
    return period_ends


def _packet_annual_fact_rows(
    ticker: str,
    *,
    as_of_date: str | None,
    v2_data_plane: bool,
) -> dict[int, dict[str, float]]:
    issuer_context = _V2_ISSUER_FACT_CONTEXT.get() or {}
    if str(issuer_context.get("ticker") or "").upper() != ticker.upper():
        issuer_context = {}
    return _annual_fact_rows(
        ticker,
        as_of_date=as_of_date,
        issuer_aware=True,
        issuer_cik=issuer_context.get("issuer_cik"),
        aliases=tuple(issuer_context.get("issuer_aliases") or ()),
        require_filed_asof=True,
    )


def _packet_annual_fact_provenance_rows(
    ticker: str,
    *,
    as_of_date: str | None,
    v2_data_plane: bool,
) -> dict[int, dict[str, dict[str, Any]]]:
    _ = v2_data_plane
    issuer_context = _V2_ISSUER_FACT_CONTEXT.get() or {}
    if str(issuer_context.get("ticker") or "").upper() != ticker.upper():
        issuer_context = {}
    _values, provenance = _annual_fact_records(
        ticker,
        as_of_date=as_of_date,
        issuer_aware=True,
        issuer_cik=issuer_context.get("issuer_cik"),
        aliases=tuple(issuer_context.get("issuer_aliases") or ()),
        require_filed_asof=True,
    )
    return provenance


def _packet_annual_period_ends(
    ticker: str,
    *,
    as_of_date: str | None,
    v2_data_plane: bool,
) -> dict[int, str]:
    issuer_context = _V2_ISSUER_FACT_CONTEXT.get() or {}
    if str(issuer_context.get("ticker") or "").upper() != ticker.upper():
        issuer_context = {}
    return _annual_period_ends(
        ticker,
        as_of_date=as_of_date,
        issuer_aware=True,
        issuer_cik=issuer_context.get("issuer_cik"),
        aliases=tuple(issuer_context.get("issuer_aliases") or ()),
        require_filed_asof=True,
    )


def _first_present(values: dict[str, float], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = values.get(key)
        if _is_num(value):
            return float(value)
    return None


def _cost_of_revenue(values: dict[str, float]) -> tuple[float | None, str | None]:
    cost = _first_present(values, GROSS_MARGIN_COST_LINE_ITEMS)
    if cost is not None:
        return cost, "reported_cost_of_revenue"
    revenue = values.get("revenue")
    gross_profit = values.get("gross_profit")
    if _is_num(revenue) and _is_num(gross_profit):
        return float(revenue) - float(gross_profit), "revenue_minus_gross_profit"
    return None, None


def _sic_issuer_classification(ticker: str) -> str | None:
    """The issuer class from the registrant's SEC SIC code, or None when no SIC is
    on file (or the SIC override is off) so the per-year line-item substring rule
    keeps deciding. Resolve once per ticker and pass it down as
    ``issuer_classification``."""
    classification, source = resolve_issuer_classification(ticker=ticker, line_items=())
    return classification if source == "sic" else None


def _gross_margin_year_row(
    fiscal_year: int,
    values: dict[str, float],
    *,
    issuer_classification: str | None = None,
) -> dict[str, Any]:
    reasons: list[str] = []
    revenue = values.get("revenue")
    gross_profit = values.get("gross_profit")
    cost_of_revenue, cost_source = _cost_of_revenue(values)
    classification = issuer_classification or infer_issuer_classification(
        line_items=list(values.keys())
    )
    if classification == ISSUER_CLASS_FINANCIAL:
        # Gross margin is not a meaningful metric for financial issuers; suppress
        # rather than emitting a spurious missing-data reason.
        return {
            "fiscal_year": fiscal_year,
            "gross_margin": None,
            "gross_profit": float(gross_profit) if _is_num(gross_profit) else None,
            "cost_of_revenue": float(cost_of_revenue) if _is_num(cost_of_revenue) else None,
            "cost_of_revenue_source": cost_source,
            "not_computable_reasons": ["GROSS_MARGIN_NOT_APPLICABLE_FINANCIAL_ISSUER"],
        }
    if not _is_num(revenue):
        reasons.append("REVENUE_MISSING")
    elif float(revenue) <= 0:
        reasons.append("REVENUE_NONPOSITIVE")
    if not _is_num(gross_profit) and not _is_num(cost_of_revenue):
        reasons.append("GROSS_PROFIT_OR_COGS_MISSING")
    if reasons:
        return {
            "fiscal_year": fiscal_year,
            "gross_margin": None,
            "gross_profit": float(gross_profit) if _is_num(gross_profit) else None,
            "cost_of_revenue": float(cost_of_revenue) if _is_num(cost_of_revenue) else None,
            "cost_of_revenue_source": cost_source,
            "not_computable_reasons": [
                "GROSS_MARGIN_NOT_COMPUTABLE",
                *list(dict.fromkeys(reasons)),
            ],
        }
    if _is_num(gross_profit):
        numerator = float(gross_profit)
        source = "gross_profit"
    else:
        numerator = float(revenue) - float(cost_of_revenue)
        source = cost_source or "revenue_minus_cost_of_revenue"
    return {
        "fiscal_year": fiscal_year,
        "gross_margin": numerator / float(revenue),
        "gross_profit": numerator,
        "cost_of_revenue": float(cost_of_revenue) if _is_num(cost_of_revenue) else None,
        "cost_of_revenue_source": source,
        "not_computable_reasons": [],
    }


def _gross_margin_metrics(
    ticker: str,
    *,
    as_of_date: str | None = None,
    v2_data_plane: bool = False,
) -> dict[str, Any]:
    by_year = _packet_annual_fact_rows(
        ticker,
        as_of_date=as_of_date,
        v2_data_plane=v2_data_plane,
    )
    if not by_year:
        return {
            "gross_margin": None,
            "gross_margin_trajectory_5y": [],
            "gross_margin_not_computable_reasons": [
                "GROSS_MARGIN_NOT_COMPUTABLE",
                "COMPANYFACTS_UNAVAILABLE",
            ],
        }
    sic_classification = _sic_issuer_classification(ticker)
    trajectory = [
        _gross_margin_year_row(year, by_year[year], issuer_classification=sic_classification)
        for year in sorted(by_year.keys())[-5:]
    ]
    latest = trajectory[-1] if trajectory else {}
    reasons: list[str] = []
    for row in trajectory:
        reasons.extend(str(item) for item in row.get("not_computable_reasons") or [])
    if latest.get("gross_margin") is None and "GROSS_MARGIN_NOT_COMPUTABLE" not in reasons:
        reasons.insert(0, "GROSS_MARGIN_NOT_COMPUTABLE")
    return {
        "gross_margin": latest.get("gross_margin"),
        "gross_margin_trajectory_5y": trajectory,
        "gross_margin_not_computable_reasons": list(dict.fromkeys(reasons)),
    }


def _operating_margin_year_row(fiscal_year: int, values: dict[str, float]) -> dict[str, Any]:
    reasons: list[str] = []
    revenue = values.get("revenue")
    operating_income = values.get("operating_income")
    if not _is_num(revenue):
        reasons.append("REVENUE_MISSING")
    elif float(revenue) <= 0:
        reasons.append("REVENUE_NONPOSITIVE")
    if not _is_num(operating_income):
        reasons.append("OPERATING_INCOME_MISSING")
    margin = float(operating_income) / float(revenue) if not reasons else None
    return {
        "fiscal_year": fiscal_year,
        "operating_margin": margin,
        "not_computable_reasons": ["OPERATING_MARGIN_NOT_COMPUTABLE", *list(dict.fromkeys(reasons))]
        if margin is None
        else [],
    }


def _normalized_operating_margin_metrics(
    ticker: str,
    *,
    as_of_date: str | None = None,
    v2_data_plane: bool = False,
) -> dict[str, Any]:
    by_year = _packet_annual_fact_rows(
        ticker,
        as_of_date=as_of_date,
        v2_data_plane=v2_data_plane,
    )
    if not by_year:
        return {
            "normalized_operating_margin": None,
            "latest_operating_margin": None,
            "operating_margin_trajectory_5y": [],
            "normalized_operating_margin_status": "NORMALIZED_MARGIN_INSUFFICIENT_HISTORY",
            "normalized_operating_margin_not_computable_reasons": [
                "NORMALIZED_MARGIN_INSUFFICIENT_HISTORY",
                "COMPANYFACTS_UNAVAILABLE",
            ],
        }
    trajectory = [
        _operating_margin_year_row(year, by_year[year]) for year in sorted(by_year.keys())[-5:]
    ]
    computable = [row for row in trajectory if _is_num(row.get("operating_margin"))]
    latest = computable[-1] if computable else {}
    reasons: list[str] = []
    for row in trajectory:
        reasons.extend(str(item) for item in row.get("not_computable_reasons") or [])
    if len(computable) < 3:
        reasons.insert(0, "NORMALIZED_MARGIN_INSUFFICIENT_HISTORY")
        return {
            "normalized_operating_margin": None,
            "latest_operating_margin": latest.get("operating_margin"),
            "operating_margin_trajectory_5y": trajectory,
            "normalized_operating_margin_status": "NORMALIZED_MARGIN_INSUFFICIENT_HISTORY",
            "normalized_operating_margin_not_computable_reasons": list(dict.fromkeys(reasons)),
        }
    normalized = sum(float(row["operating_margin"]) for row in computable) / len(computable)
    return {
        "normalized_operating_margin": normalized,
        "latest_operating_margin": latest.get("operating_margin"),
        "operating_margin_trajectory_5y": trajectory,
        "normalized_operating_margin_status": "OK",
        "normalized_operating_margin_not_computable_reasons": [],
    }


def _cash_conversion_cycle_year_row(fiscal_year: int, values: dict[str, float]) -> dict[str, Any]:
    revenue = values.get("revenue")
    receivables = values.get("accounts_receivable")
    inventory = values.get("inventory")
    payables = values.get("accounts_payable")
    cost_of_revenue, cost_source = _cost_of_revenue(values)
    reasons: list[str] = []
    if not _is_num(revenue):
        reasons.append("REVENUE_MISSING")
    elif float(revenue) <= 0:
        reasons.append("REVENUE_NONPOSITIVE")
    if not _is_num(receivables):
        reasons.append("ACCOUNTS_RECEIVABLE_MISSING")
    if not _is_num(inventory):
        reasons.append("INVENTORY_MISSING")
    if not _is_num(payables):
        reasons.append("ACCOUNTS_PAYABLE_MISSING")
    if not _is_num(cost_of_revenue):
        reasons.append("COGS_MISSING")
    elif float(cost_of_revenue) <= 0:
        reasons.append("COGS_NONPOSITIVE")

    dso = (
        float(receivables) / float(revenue) * 365.0
        if _is_num(receivables) and _is_num(revenue) and float(revenue) > 0
        else None
    )
    dio = (
        float(inventory) / float(cost_of_revenue) * 365.0
        if _is_num(inventory) and _is_num(cost_of_revenue) and float(cost_of_revenue) > 0
        else None
    )
    dpo = (
        float(payables) / float(cost_of_revenue) * 365.0
        if _is_num(payables) and _is_num(cost_of_revenue) and float(cost_of_revenue) > 0
        else None
    )
    cycle = dso + dio - dpo if dso is not None and dio is not None and dpo is not None else None
    return {
        "fiscal_year": fiscal_year,
        "cash_conversion_cycle": cycle,
        "days_sales_outstanding": dso,
        "days_inventory_outstanding": dio,
        "days_payable_outstanding": dpo,
        "cost_of_revenue_source": cost_source,
        "not_computable_reasons": ["CCC_NOT_COMPUTABLE", *list(dict.fromkeys(reasons))]
        if cycle is None
        else [],
    }


def _cash_conversion_cycle_metrics(
    ticker: str,
    *,
    as_of_date: str | None = None,
    v2_data_plane: bool = False,
) -> dict[str, Any]:
    by_year = _packet_annual_fact_rows(
        ticker,
        as_of_date=as_of_date,
        v2_data_plane=v2_data_plane,
    )
    if not by_year:
        return {
            "cash_conversion_cycle": None,
            "cash_conversion_cycle_trajectory_5y": [],
            "cash_conversion_cycle_not_computable_reasons": [
                "CCC_NOT_COMPUTABLE",
                "COMPANYFACTS_UNAVAILABLE",
            ],
        }
    trajectory = [
        _cash_conversion_cycle_year_row(year, by_year[year]) for year in sorted(by_year.keys())[-5:]
    ]
    latest = trajectory[-1] if trajectory else {}
    reasons: list[str] = []
    for row in trajectory:
        reasons.extend(str(item) for item in row.get("not_computable_reasons") or [])
    if latest.get("cash_conversion_cycle") is None and "CCC_NOT_COMPUTABLE" not in reasons:
        reasons.insert(0, "CCC_NOT_COMPUTABLE")
    return {
        "cash_conversion_cycle": latest.get("cash_conversion_cycle"),
        "days_sales_outstanding": latest.get("days_sales_outstanding"),
        "days_inventory_outstanding": latest.get("days_inventory_outstanding"),
        "days_payable_outstanding": latest.get("days_payable_outstanding"),
        "cash_conversion_cycle_trajectory_5y": trajectory,
        "cash_conversion_cycle_not_computable_reasons": list(dict.fromkeys(reasons)),
    }


def _share_count_cagr_metrics(
    ticker: str,
    *,
    as_of_date: str | None = None,
    v2_data_plane: bool = False,
) -> dict[str, Any]:
    by_year = _packet_annual_fact_rows(
        ticker,
        as_of_date=as_of_date,
        v2_data_plane=v2_data_plane,
    )
    rows = []
    for year in sorted(by_year.keys())[-5:]:
        value = by_year[year].get("shares_outstanding")
        if not _is_num(value):
            continue
        lineage = by_year[year].get("shares_outstanding_split_lineage")
        rows.append(
            {
                "fiscal_year": year,
                "value": value,
                **(dict(lineage) if isinstance(lineage, dict) else {}),
            }
        )
    if len(rows) < 2:
        return {
            "share_count_cagr": None,
            "share_count_cagr_raw": None,
            "share_count_split_adjustments": [],
            "share_count_trajectory_5y": rows,
            "share_count_cagr_not_computable_reasons": [
                "SHARE_COUNT_CAGR_NOT_COMPUTABLE",
                "SHARE_COUNT_HISTORY_INSUFFICIENT",
            ],
        }
    adjusted_rows, adjustments, split_lineage_complete = _split_adjusted_share_series(rows)
    cagr = _shares_cagr(adjusted_rows) if split_lineage_complete else None
    raw_cagr = _shares_cagr(rows)
    if cagr is None:
        return {
            "share_count_cagr": None,
            "share_count_cagr_raw": raw_cagr,
            "share_count_split_adjustments": adjustments,
            "share_count_trajectory_5y": rows,
            "share_count_cagr_not_computable_reasons": [
                "SHARE_COUNT_CAGR_NOT_COMPUTABLE",
                *(["SHARE_SPLIT_LINEAGE_MISSING"] if not split_lineage_complete else []),
            ],
        }
    direction = "BUYBACKS" if cagr < -0.005 else "DILUTION" if cagr > 0.005 else "STABLE"
    return {
        "share_count_cagr": cagr,
        "share_count_cagr_raw": raw_cagr if adjustments else None,
        "share_count_split_adjustments": adjustments,
        "share_count_trajectory_5y": adjusted_rows if adjustments else rows,
        "share_count_latest": adjusted_rows[-1]["value"],
        "share_count_oldest": adjusted_rows[0]["value"],
        "share_count_cagr_direction": direction,
        "share_count_cagr_not_computable_reasons": [],
    }


def _score_missing(missing_inputs: list[str]) -> dict[str, Any]:
    return {
        "value": None,
        "interpretation": "insufficient data",
        "missing_inputs": list(dict.fromkeys(missing_inputs)),
        "status": "INSUFFICIENT_DATA",
    }


def _require_number(
    values: dict[str, float],
    key: str,
    label: str,
    missing_inputs: list[str],
    *,
    positive: bool = False,
    nonzero: bool = False,
) -> float | None:
    value = values.get(key)
    if not _is_num(value):
        missing_inputs.append(f"{label}_MISSING")
        return None
    value_float = float(value)
    if positive and value_float <= 0:
        missing_inputs.append(f"{label}_NONPOSITIVE")
        return None
    if nonzero and value_float == 0:
        missing_inputs.append(f"{label}_ZERO")
        return None
    return value_float


def _gross_margin_value(
    values: dict[str, float], *, issuer_classification: str | None = None
) -> float | None:
    row = _gross_margin_year_row(0, values, issuer_classification=issuer_classification)
    gross_margin = row.get("gross_margin")
    return float(gross_margin) if _is_num(gross_margin) else None


def _total_liabilities_value(values: dict[str, float]) -> float | None:
    reported = values.get("total_liabilities")
    if _is_num(reported):
        return float(reported)
    total_assets = values.get("total_assets")
    equity = values.get("equity")
    if _is_num(total_assets) and _is_num(equity):
        return float(total_assets) - float(equity)
    return None


def _depreciation_value(values: dict[str, float]) -> float | None:
    depreciation = values.get("depreciation")
    return float(depreciation) if _is_num(depreciation) else None


def _income_continuing_or_net_income(
    values: dict[str, float], label: str, missing_inputs: list[str]
) -> float | None:
    income_continuing = values.get("income_continuing")
    if _is_num(income_continuing):
        return float(income_continuing)
    net_income = values.get("net_income")
    if _is_num(net_income):
        return float(net_income)
    missing_inputs.append(f"INCOME_CONTINUING_OR_NET_INCOME_{label}_MISSING")
    return None


def _piotroski_interpretation(score: int) -> str:
    if score >= 8:
        return "high quality"
    if score >= 6:
        return "moderate"
    return "low quality"


def _beneish_interpretation(value: float) -> str:
    return "potential manipulation" if value > -1.78 else "no manipulation flag"


def _altman_z_interpretation(value: float) -> str:
    if value > 2.60:
        return "safe"
    if value >= 1.10:
        return "gray zone"
    return "distress"


def _latest_two_year_values(
    by_year: dict[int, dict[str, float]],
) -> tuple[dict[str, float], dict[str, float]] | None:
    years = sorted(by_year)
    if len(years) < 2:
        return None
    return by_year[years[-2]], by_year[years[-1]]


def _piotroski_f_score_from_rows(
    by_year: dict[int, dict[str, float]], *, issuer_classification: str | None = None
) -> dict[str, Any]:
    values = _latest_two_year_values(by_year)
    if values is None:
        return _score_missing(["TWO_YEAR_HISTORY_MISSING"])
    prior, current = values
    missing: list[str] = []

    net_income_current = _require_number(current, "net_income", "NET_INCOME_CURRENT", missing)
    cfo_current = _require_number(current, "cfo", "CFO_CURRENT", missing)
    total_assets_current = _require_number(
        current, "total_assets", "TOTAL_ASSETS_CURRENT", missing, positive=True
    )
    total_assets_prior = _require_number(
        prior, "total_assets", "TOTAL_ASSETS_PRIOR", missing, positive=True
    )
    net_income_prior = _require_number(prior, "net_income", "NET_INCOME_PRIOR", missing)
    total_debt_current = _require_number(current, "total_debt", "TOTAL_DEBT_CURRENT", missing)
    total_debt_prior = _require_number(prior, "total_debt", "TOTAL_DEBT_PRIOR", missing)
    current_assets_current = _require_number(
        current, "current_assets", "CURRENT_ASSETS_CURRENT", missing
    )
    current_assets_prior = _require_number(prior, "current_assets", "CURRENT_ASSETS_PRIOR", missing)
    current_liabilities_current = _require_number(
        current,
        "current_liabilities",
        "CURRENT_LIABILITIES_CURRENT",
        missing,
        positive=True,
    )
    current_liabilities_prior = _require_number(
        prior,
        "current_liabilities",
        "CURRENT_LIABILITIES_PRIOR",
        missing,
        positive=True,
    )
    shares_current = _require_number(
        current, "shares_outstanding", "SHARES_OUTSTANDING_CURRENT", missing
    )
    shares_prior = _require_number(prior, "shares_outstanding", "SHARES_OUTSTANDING_PRIOR", missing)
    revenue_current = _require_number(current, "revenue", "REVENUE_CURRENT", missing, positive=True)
    revenue_prior = _require_number(prior, "revenue", "REVENUE_PRIOR", missing, positive=True)
    gross_margin_current = _gross_margin_value(
        current, issuer_classification=issuer_classification
    )
    gross_margin_prior = _gross_margin_value(prior, issuer_classification=issuer_classification)
    if gross_margin_current is None:
        missing.append("GROSS_MARGIN_CURRENT_MISSING")
    if gross_margin_prior is None:
        missing.append("GROSS_MARGIN_PRIOR_MISSING")

    if missing:
        return _score_missing(missing)

    roa_current = float(net_income_current) / float(total_assets_current)
    roa_prior = float(net_income_prior) / float(total_assets_prior)
    current_ratio_current = float(current_assets_current) / float(current_liabilities_current)
    current_ratio_prior = float(current_assets_prior) / float(current_liabilities_prior)
    asset_turnover_current = float(revenue_current) / float(total_assets_current)
    asset_turnover_prior = float(revenue_prior) / float(total_assets_prior)
    signals = [
        float(net_income_current) > 0,
        float(cfo_current) > 0,
        roa_current > roa_prior,
        float(cfo_current) > float(net_income_current),
        float(total_debt_current) < float(total_debt_prior),
        current_ratio_current > current_ratio_prior,
        float(shares_current) <= float(shares_prior),
        float(gross_margin_current) > float(gross_margin_prior),
        asset_turnover_current > asset_turnover_prior,
    ]
    score = int(sum(1 for item in signals if item))
    return {
        "value": score,
        "interpretation": _piotroski_interpretation(score),
        "missing_inputs": [],
        "status": "OK",
    }


def _beneish_m_score_from_rows(
    by_year: dict[int, dict[str, float]], *, issuer_classification: str | None = None
) -> dict[str, Any]:
    values = _latest_two_year_values(by_year)
    if values is None:
        return _score_missing(["TWO_YEAR_HISTORY_MISSING"])
    prior, current = values
    missing: list[str] = []

    sales_current = _require_number(current, "revenue", "REVENUE_CURRENT", missing, positive=True)
    sales_prior = _require_number(prior, "revenue", "REVENUE_PRIOR", missing, positive=True)
    ar_current = _require_number(
        current, "accounts_receivable", "ACCOUNTS_RECEIVABLE_CURRENT", missing
    )
    ar_prior = _require_number(prior, "accounts_receivable", "ACCOUNTS_RECEIVABLE_PRIOR", missing)
    current_assets_current = _require_number(
        current, "current_assets", "CURRENT_ASSETS_CURRENT", missing
    )
    current_assets_prior = _require_number(prior, "current_assets", "CURRENT_ASSETS_PRIOR", missing)
    gross_ppe_current = _require_number(current, "gross_ppe", "GROSS_PPE_CURRENT", missing)
    gross_ppe_prior = _require_number(prior, "gross_ppe", "GROSS_PPE_PRIOR", missing)
    total_assets_current = _require_number(
        current, "total_assets", "TOTAL_ASSETS_CURRENT", missing, positive=True
    )
    total_assets_prior = _require_number(
        prior, "total_assets", "TOTAL_ASSETS_PRIOR", missing, positive=True
    )
    depreciation_current = _require_number(current, "depreciation", "DEPRECIATION_CURRENT", missing)
    depreciation_prior = _require_number(prior, "depreciation", "DEPRECIATION_PRIOR", missing)
    sga_current = _require_number(current, "sga", "SGA_CURRENT", missing)
    sga_prior = _require_number(prior, "sga", "SGA_PRIOR", missing)
    total_debt_current = _require_number(current, "total_debt", "TOTAL_DEBT_CURRENT", missing)
    total_debt_prior = _require_number(prior, "total_debt", "TOTAL_DEBT_PRIOR", missing)
    current_liabilities_current = _require_number(
        current,
        "current_liabilities",
        "CURRENT_LIABILITIES_CURRENT",
        missing,
    )
    current_liabilities_prior = _require_number(
        prior, "current_liabilities", "CURRENT_LIABILITIES_PRIOR", missing
    )
    cfo_current = _require_number(current, "cfo", "CFO_CURRENT", missing)
    income_continuing_current = _income_continuing_or_net_income(current, "CURRENT", missing)
    gross_margin_current = _gross_margin_value(
        current, issuer_classification=issuer_classification
    )
    gross_margin_prior = _gross_margin_value(prior, issuer_classification=issuer_classification)
    if gross_margin_current is None:
        missing.append("GROSS_MARGIN_CURRENT_MISSING")
    elif gross_margin_current <= 0:
        missing.append("GROSS_MARGIN_CURRENT_NONPOSITIVE")
    if gross_margin_prior is None:
        missing.append("GROSS_MARGIN_PRIOR_MISSING")
    elif gross_margin_prior <= 0:
        missing.append("GROSS_MARGIN_PRIOR_NONPOSITIVE")

    if missing:
        return _score_missing(missing)

    ar_sales_prior = float(ar_prior) / float(sales_prior)
    if ar_sales_prior == 0:
        return _score_missing(["DSRI_PRIOR_AR_SALES_ZERO"])
    dsri = (float(ar_current) / float(sales_current)) / ar_sales_prior
    gmi = float(gross_margin_prior) / float(gross_margin_current)
    aqi_denominator_prior = 1.0 - (
        (float(current_assets_prior) + float(gross_ppe_prior)) / float(total_assets_prior)
    )
    aqi_denominator_current = 1.0 - (
        (float(current_assets_current) + float(gross_ppe_current)) / float(total_assets_current)
    )
    if aqi_denominator_prior == 0:
        return _score_missing(["AQI_PRIOR_DENOMINATOR_ZERO"])
    sgi = float(sales_current) / float(sales_prior)
    if float(depreciation_current) + float(gross_ppe_current) == 0:
        return _score_missing(["DEPRECIATION_RATE_CURRENT_DENOMINATOR_ZERO"])
    if float(depreciation_prior) + float(gross_ppe_prior) == 0:
        return _score_missing(["DEPRECIATION_RATE_PRIOR_DENOMINATOR_ZERO"])
    depreciation_rate_current = float(depreciation_current) / (
        float(depreciation_current) + float(gross_ppe_current)
    )
    depreciation_rate_prior = float(depreciation_prior) / (
        float(depreciation_prior) + float(gross_ppe_prior)
    )
    if depreciation_rate_current <= 0:
        return _score_missing(["DEPRECIATION_RATE_CURRENT_NONPOSITIVE"])
    sga_sales_prior = float(sga_prior) / float(sales_prior)
    if sga_sales_prior == 0:
        return _score_missing(["SGAI_PRIOR_SGA_SALES_ZERO"])
    leverage_prior = (float(total_debt_prior) + float(current_liabilities_prior)) / float(
        total_assets_prior
    )
    if leverage_prior == 0:
        return _score_missing(["LVGI_PRIOR_LEVERAGE_ZERO"])
    aqi = aqi_denominator_current / aqi_denominator_prior
    depi = depreciation_rate_prior / depreciation_rate_current
    sgai = (float(sga_current) / float(sales_current)) / sga_sales_prior
    lvgi = (
        (float(total_debt_current) + float(current_liabilities_current))
        / float(total_assets_current)
    ) / leverage_prior
    tata = (float(income_continuing_current) - float(cfo_current)) / float(total_assets_current)
    value = round(
        -4.84
        + 0.92 * dsri
        + 0.528 * gmi
        + 0.404 * aqi
        + 0.892 * sgi
        + 0.115 * depi
        - 0.172 * sgai
        + 4.679 * tata
        - 0.327 * lvgi,
        6,
    )
    return {
        "value": value,
        "interpretation": _beneish_interpretation(value),
        "missing_inputs": [],
        "status": "OK",
    }


def _altman_z_score_from_rows(
    by_year: dict[int, dict[str, float]],
    *,
    current_price: float | None,
) -> dict[str, Any]:
    years = sorted(by_year)
    if not years:
        return _score_missing(["ANNUAL_HISTORY_MISSING"])
    current = by_year[years[-1]]
    missing: list[str] = []
    current_assets = _require_number(current, "current_assets", "CURRENT_ASSETS", missing)
    current_liabilities = _require_number(
        current, "current_liabilities", "CURRENT_LIABILITIES", missing
    )
    total_assets = _require_number(current, "total_assets", "TOTAL_ASSETS", missing, positive=True)
    retained_earnings = _require_number(current, "retained_earnings", "RETAINED_EARNINGS", missing)
    operating_income = _require_number(current, "operating_income", "OPERATING_INCOME", missing)
    shares = _require_number(
        current, "shares_outstanding", "SHARES_OUTSTANDING", missing, positive=True
    )
    total_liabilities = _total_liabilities_value(current)
    if total_liabilities is None:
        missing.append("TOTAL_LIABILITIES_MISSING")
    elif total_liabilities <= 0:
        missing.append("TOTAL_LIABILITIES_NONPOSITIVE")
    if not _is_num(current_price) or float(current_price) <= 0:
        missing.append("CURRENT_PRICE_MISSING")

    if missing:
        return _score_missing(missing)

    working_capital = float(current_assets) - float(current_liabilities)
    market_value_equity = float(current_price) * float(shares)
    value = round(
        6.56 * (working_capital / float(total_assets))
        + 3.26 * (float(retained_earnings) / float(total_assets))
        + 6.72 * (float(operating_income) / float(total_assets))
        + 1.05 * (market_value_equity / float(total_liabilities)),
        6,
    )
    return {
        "value": value,
        "interpretation": _altman_z_interpretation(value),
        "missing_inputs": [],
        "status": "OK",
    }


def _forensic_quality_scores(
    ticker: str,
    *,
    as_of_date: str | None,
    current_price: float | None,
    v2_data_plane: bool = False,
) -> dict[str, Any]:
    by_year = _packet_annual_fact_rows(
        ticker,
        as_of_date=as_of_date,
        v2_data_plane=v2_data_plane,
    )
    sic_classification = _sic_issuer_classification(ticker)
    piotroski = _piotroski_f_score_from_rows(by_year, issuer_classification=sic_classification)
    beneish = _beneish_m_score_from_rows(by_year, issuer_classification=sic_classification)
    altman = _altman_z_score_from_rows(by_year, current_price=current_price)
    return {
        "piotroski_f_score": piotroski["value"],
        "piotroski_interpretation": piotroski["interpretation"],
        "piotroski_missing_inputs": list(piotroski.get("missing_inputs") or []),
        "piotroski_status": piotroski["status"],
        "beneish_m_score": beneish["value"],
        "beneish_interpretation": beneish["interpretation"],
        "beneish_missing_inputs": list(beneish.get("missing_inputs") or []),
        "beneish_status": beneish["status"],
        "altman_z_score": altman["value"],
        "altman_interpretation": altman["interpretation"],
        "altman_missing_inputs": list(altman.get("missing_inputs") or []),
        "altman_status": altman["status"],
    }


def _effective_tax_rate(values: dict[str, float]) -> float | None:
    tax = _first_present(values, ROIC_TAX_LINE_ITEMS)
    pretax = _first_present(values, ROIC_PRETAX_LINE_ITEMS)
    if tax is None or pretax is None or pretax <= 0:
        return None
    return max(0.0, min(float(tax) / float(pretax), 0.50))


def _roic_row(
    values: dict[str, float], *, issuer_classification: str | None = None
) -> dict[str, Any]:
    reasons: list[str] = []
    operating_income = values.get("operating_income")
    debt = values.get("total_debt")
    equity = values.get("equity")
    cash = values.get("cash")
    tax_rate = _effective_tax_rate(values)
    classification = issuer_classification or infer_issuer_classification(
        line_items=list(values.keys())
    )
    financial_equity_basis = classification == ISSUER_CLASS_FINANCIAL and not _is_num(debt)
    if not _is_num(operating_income):
        reasons.append("NOPAT_DATA_MISSING")
    if tax_rate is None:
        reasons.append("TRAILING_EFFECTIVE_TAX_RATE_MISSING")
    if not _is_num(debt) and not financial_equity_basis:
        reasons.append("TOTAL_DEBT_MISSING")
    if not _is_num(equity):
        reasons.append("TOTAL_EQUITY_MISSING")
    if not _is_num(cash) and not financial_equity_basis:
        reasons.append("EXCESS_CASH_MISSING")
    if reasons:
        return {
            "roic": None,
            "nopat": None,
            "invested_capital": None,
            "reasons": list(dict.fromkeys(reasons)),
        }

    # Financial issuers (deposits/loans present) lack a meaningful total-debt
    # figure; use reported equity as the invested-capital basis instead.
    if financial_equity_basis:
        invested_capital = float(equity)
        invested_capital_basis = "equity_only_financial"
        # The financial-issuer equity basis is a *basis annotation*, not a
        # computation failure. Carry it in a dedicated basis_notes field so it
        # never leaks into the not-computable reasons stream (a successful ROIC
        # must not be mislabeled "not computable").
        basis_notes = ["FINANCIAL_ISSUER_EQUITY_BASIS"]
    else:
        # V1 uses invested capital with goodwill: total debt + total equity - cash.
        # WACC is a conservative sector-agnostic default until sector-specific tuning lands.
        invested_capital = float(debt) + float(equity) - float(cash)
        invested_capital_basis = ROIC_INVESTED_CAPITAL_BASIS
        basis_notes = []
    if invested_capital <= 0:
        return {
            "roic": None,
            "nopat": None,
            "invested_capital": invested_capital,
            "reasons": ["NEGATIVE_OR_ZERO_INVESTED_CAPITAL"],
        }
    nopat = float(operating_income) * (1.0 - float(tax_rate))
    return {
        "roic": nopat / invested_capital,
        "nopat": nopat,
        "invested_capital": invested_capital,
        "invested_capital_basis": invested_capital_basis,
        "effective_tax_rate": tax_rate,
        "basis_notes": basis_notes,
        "reasons": [],
    }


def _returns_on_capital_metrics(
    ticker: str,
    *,
    as_of_date: str | None,
    v2_data_plane: bool = False,
) -> dict[str, Any]:
    by_year = _packet_annual_fact_rows(
        ticker,
        as_of_date=as_of_date,
        v2_data_plane=v2_data_plane,
    )
    if not by_year:
        return {
            "roic": None,
            "roic_wacc_spread": None,
            "incremental_roic_3y": None,
            "roic_trajectory_5y": [],
            "wacc_default": ROIC_WACC_DEFAULT,
            "wacc_default_reason": ROIC_WACC_DEFAULT_REASON,
            "invested_capital_basis": ROIC_INVESTED_CAPITAL_BASIS,
            "roic_not_computable_reasons": ["ROIC_NOT_COMPUTABLE", "COMPANYFACTS_UNAVAILABLE"],
        }
    trajectory: list[dict[str, Any]] = []
    reasons: list[str] = []
    sic_classification = _sic_issuer_classification(ticker)
    for fiscal_year in sorted(by_year.keys())[-5:]:
        row = _roic_row(by_year[fiscal_year], issuer_classification=sic_classification)
        trajectory.append(
            {
                "fiscal_year": fiscal_year,
                "roic": row.get("roic"),
                "nopat": row.get("nopat"),
                "invested_capital": row.get("invested_capital"),
                "invested_capital_basis": row.get("invested_capital_basis"),
                "effective_tax_rate": row.get("effective_tax_rate"),
                "basis_notes": list(row.get("basis_notes") or []),
                "not_computable_reasons": row.get("reasons") or [],
            }
        )
        reasons.extend(str(item) for item in (row.get("reasons") or []))
    # Prefer the most-recent COMPLETE fiscal year for the headline ROIC: the
    # strict-max year is frequently a partially-filed stub (operating_income only)
    # whose _roic_row returns None. Walk back to the most-recent year that
    # actually computes, and flag when we had to fall back off the strict max.
    latest = trajectory[-1] if trajectory else {}
    chosen = next(
        (entry for entry in reversed(trajectory) if entry.get("roic") is not None),
        None,
    )
    if chosen is not None:
        latest = chosen
        if chosen is not (trajectory[-1] if trajectory else None):
            reasons.append("LATEST_YEAR_STUB_USED_PRIOR_COMPLETE_YEAR")
    latest_roic = latest.get("roic")
    incremental_roic_3y = None
    if len(trajectory) >= 4:
        start = trajectory[-4]
        end = trajectory[-1]
        start_nopat = start.get("nopat")
        end_nopat = end.get("nopat")
        start_ic = start.get("invested_capital")
        end_ic = end.get("invested_capital")
        if all(_is_num(value) for value in (start_nopat, end_nopat, start_ic, end_ic)) and float(
            end_ic
        ) != float(start_ic):
            incremental_roic_3y = (float(end_nopat) - float(start_nopat)) / (
                float(end_ic) - float(start_ic)
            )
        else:
            reasons.append("INCREMENTAL_ROIC_NOT_COMPUTABLE")
    elif trajectory:
        reasons.append("INCREMENTAL_ROIC_HISTORY_INSUFFICIENT")
    reason_codes = (
        ["ROIC_NOT_COMPUTABLE", *list(dict.fromkeys(reasons))]
        if latest_roic is None
        else list(dict.fromkeys(reasons))
    )
    # Report the headline basis from the chosen (computed) year so a financial
    # issuer whose ROIC used the equity-only basis is LABELED equity_only_financial
    # at the consumed level, instead of the default with_goodwill basis.
    headline_basis = (
        latest.get("invested_capital_basis")
        if _is_num(latest_roic)
        else ROIC_INVESTED_CAPITAL_BASIS
    )
    if headline_basis is None:
        headline_basis = ROIC_INVESTED_CAPITAL_BASIS
    return {
        "roic": latest_roic,
        "roic_wacc_spread": (float(latest_roic) - ROIC_WACC_DEFAULT)
        if _is_num(latest_roic)
        else None,
        "incremental_roic_3y": incremental_roic_3y,
        "roic_trajectory_5y": trajectory,
        "wacc_default": ROIC_WACC_DEFAULT,
        "wacc_default_reason": ROIC_WACC_DEFAULT_REASON,
        "invested_capital_basis": headline_basis,
        "roic_basis_notes": list(latest.get("basis_notes") or []) if _is_num(latest_roic) else [],
        "roic_not_computable_reasons": reason_codes,
    }


def _latest_shares_outstanding(ticker: str, *, as_of_date: str | None) -> float | None:
    try:
        from app.db import get_db

        with get_db() as conn:
            return latest_companyfacts_value(
                conn,
                ticker,
                line_item="shares_outstanding",
                period_types=ANNUAL_COMPANYFACTS_PERIOD_TYPES,
                as_of_date=as_of_date,
                value_not_null=True,
                require_filed_asof=True,
            )
    except Exception:
        return None


def _market_cap_from_cache(
    ticker: str, *, as_of_date: str | None, current_price: float | None = None
) -> float | None:
    """Read only market caps whose persisted USD-millions unit is explicit."""

    _ = current_price
    try:
        from app.db import get_db

        with get_db() as conn:
            columns = {
                str(row["name"])
                for row in conn.execute("PRAGMA table_info(market_caps)").fetchall()
            }
            if "market_cap_unit" not in columns:
                return None
            date_filter = str(as_of_date or date.today().isoformat())
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
                (ticker.upper(), MARKET_CAP_UNIT_USD_MILLIONS, date_filter),
            ).fetchone()
            value = row["market_cap"] if row else None
            if _is_finite_num(value) and float(value) > 0:
                return float(value)
    except Exception:
        return None
    return None


def _fcf_yield_metrics(
    ticker: str,
    *,
    as_of_date: str | None,
    current_price: float | None = None,
    v2_data_plane: bool = False,
    market_cap_override_mm: float | None = None,
    market_cap_unit: str | None = None,
    quote_snapshot_id: str | None = None,
) -> dict[str, Any]:
    by_year = _packet_annual_fact_rows(
        ticker,
        as_of_date=as_of_date,
        v2_data_plane=v2_data_plane,
    )
    provenance_by_year = _packet_annual_fact_provenance_rows(
        ticker,
        as_of_date=as_of_date,
        v2_data_plane=v2_data_plane,
    )
    latest_year = next(
        (
            year
            for year in sorted(by_year, reverse=True)
            if _is_num(by_year[year].get("cfo")) or _is_num(by_year[year].get("capex"))
        ),
        None,
    )
    latest = by_year.get(latest_year, {}) if latest_year is not None else {}
    latest_provenance = provenance_by_year.get(latest_year, {}) if latest_year is not None else {}
    cfo = latest.get("cfo")
    capex = latest.get("capex")
    explicit_override = _is_num(market_cap_override_mm) and (
        v2_data_plane or market_cap_unit == MARKET_CAP_UNIT_USD_MILLIONS
    )
    if explicit_override:
        try:
            market_cap = float(market_cap_override_mm)
        except OverflowError:
            market_cap = market_cap_override_mm
    elif v2_data_plane or market_cap_override_mm is not None:
        market_cap = None
    else:
        market_cap = _market_cap_from_cache(
            ticker,
            as_of_date=as_of_date,
            current_price=current_price,
        )
    # Canonical contract: every accepted value is already USD millions.
    # There is deliberately no magnitude-based conversion here.
    market_cap_statement_units = market_cap
    reasons: list[str] = []
    if not _is_num(cfo):
        reasons.append("CFO_MISSING")
    elif not _is_finite_num(cfo):
        reasons.append("CFO_NON_FINITE")
    if not _is_num(capex):
        reasons.append("CAPEX_MISSING")
    elif not _is_finite_num(capex):
        reasons.append("CAPEX_NON_FINITE")
    if not _is_num(market_cap_statement_units):
        reasons.append("MARKET_CAP_MISSING")
    elif not _is_finite_num(market_cap_statement_units):
        reasons.append("MARKET_CAP_NON_FINITE")
    elif float(market_cap_statement_units) <= 0:
        reasons.append("MARKET_CAP_MISSING")
    if reasons:
        return {
            "fcf_yield": None,
            "ttm_fcf": None,
            "market_cap": market_cap,
            "market_cap_unit": (MARKET_CAP_UNIT_USD_MILLIONS if market_cap is not None else None),
            "fcf_yield_reasons": reasons,
        }
    fcf, fcf_yield = _fcf_yield_from_inputs(cfo, capex, market_cap_statement_units)
    if fcf is None or fcf_yield is None:
        return {
            "fcf_yield": None,
            "ttm_fcf": None,
            "market_cap": market_cap,
            "market_cap_unit": (MARKET_CAP_UNIT_USD_MILLIONS if market_cap is not None else None),
            "fcf_yield_reasons": ["FCF_YIELD_NON_FINITE"],
        }
    metric_trace = canonical_metric_trace(
        metric="fcf_yield",
        formula="(cfo_usd_millions - abs(capex_usd_millions)) / market_cap_usd_millions",
        inputs={
            "cfo_usd_millions": float(cfo),
            "capex_usd_millions": float(capex),
            "free_cash_flow_usd_millions": fcf,
            "market_cap_usd_millions": float(market_cap_statement_units),
        },
        output=fcf_yield,
        recomputed_output=fcf / float(market_cap_statement_units),
        output_unit="ratio",
        quote_snapshot_id=quote_snapshot_id,
        input_provenance={
            "cfo_usd_millions": latest_provenance.get("cfo", {}),
            "capex_usd_millions": latest_provenance.get("capex", {}),
            "free_cash_flow_usd_millions": _derived_trace_provenance(
                fcf,
                unit=MARKET_CAP_UNIT_USD_MILLIONS,
                source="DERIVED_FREE_CASH_FLOW",
                components={
                    "cfo_usd_millions": latest_provenance.get("cfo", {}),
                    "capex_usd_millions": latest_provenance.get("capex", {}),
                },
            ),
            "market_cap_usd_millions": _market_cap_trace_provenance(
                float(market_cap_statement_units),
                as_of_date=as_of_date,
                quote_snapshot_id=quote_snapshot_id,
            ),
        },
    )
    return {
        "fcf_yield": fcf_yield,
        "ttm_fcf": fcf,
        "market_cap": market_cap,
        "market_cap_unit": MARKET_CAP_UNIT_USD_MILLIONS,
        "fcf_yield_reasons": [],
        "fcf_yield_source": "latest_annual_companyfacts_as_ttm_proxy",
        "metric_trace": metric_trace,
    }


def _fcf_yield_from_inputs(
    cfo: Any, capex: Any, market_cap_statement_units: Any
) -> tuple[float | None, float | None]:
    if (
        not _is_finite_num(cfo)
        or not _is_finite_num(capex)
        or not _is_finite_num(market_cap_statement_units)
        or float(market_cap_statement_units) <= 0
    ):
        return None, None
    fcf = float(cfo) - abs(float(capex))
    if not math.isfinite(fcf):
        return None, None
    fcf_yield = fcf / float(market_cap_statement_units)
    if not math.isfinite(fcf_yield):
        return None, None
    return fcf, fcf_yield


def _historical_price_provider() -> Any:
    from app.config import get_config
    from app.market.price_provider import build_price_provider

    cfg = get_config()
    timeout_seconds = min(
        HISTORICAL_MULTIPLE_PRICE_TIMEOUT_SECONDS,
        max(
            1.0,
            float(
                getattr(cfg, "http_timeout_seconds", HISTORICAL_MULTIPLE_PRICE_TIMEOUT_SECONDS)
                or HISTORICAL_MULTIPLE_PRICE_TIMEOUT_SECONDS
            ),
        ),
    )
    try:
        historical_cfg = cfg.model_copy(update={"http_timeout_seconds": timeout_seconds})
    except AttributeError:  # pragma: no cover - pydantic v1 compatibility
        historical_cfg = cfg.copy(update={"http_timeout_seconds": timeout_seconds})
    return build_price_provider(
        cfg=historical_cfg,
        with_prices=True,
        fallback_days=HISTORICAL_MULTIPLE_PRICE_FALLBACK_DAYS,
        max_retries=HISTORICAL_MULTIPLE_PRICE_MAX_RETRIES,
    )


def _historical_fiscal_year_prices(
    ticker: str,
    period_ends: dict[int, str],
    *,
    as_of_date: str | None,
) -> dict[int, float]:
    if not as_of_date or not period_ends:
        return {}
    provider = _historical_price_provider()
    prices: dict[int, float] = {}
    for fiscal_year, period_end in sorted(period_ends.items(), reverse=True):
        if str(period_end) > str(as_of_date):
            continue
        try:
            snapshot = provider.get_price_asof(ticker.upper(), str(period_end))
        except Exception:
            snapshot = None
        price = getattr(snapshot, "price", None)
        if _is_num(price) and float(price) > 0:
            prices[int(fiscal_year)] = float(price)
            continue
        break
    return dict(sorted(prices.items()))


def _depreciation_amortization(values: dict[str, float]) -> float | None:
    reported = values.get("depreciation_amortization")
    if _is_num(reported):
        return float(reported)
    parts = [
        value
        for value in (values.get("depreciation"), values.get("intangible_amortization"))
        if _is_num(value)
    ]
    return sum(float(value) for value in parts) if parts else None


def _market_cap_statement_units(price: Any, values: dict[str, float]) -> float | None:
    shares = values.get("shares_outstanding")
    if not _is_num(price) or not _is_num(shares) or float(price) <= 0 or float(shares) <= 0:
        return None
    return float(price) * float(shares)


def _enterprise_value_statement_units(price: Any, values: dict[str, float]) -> float | None:
    market_cap = _market_cap_statement_units(price, values)
    if market_cap is None:
        return None
    debt_value = values.get("total_debt")
    cash_value = values.get("cash")
    if not _is_finite_num(debt_value) or not _is_finite_num(cash_value):
        return None
    debt = float(debt_value)
    cash = float(cash_value)
    return market_cap + debt - cash


def _complete_fact_provenance(
    record: dict[str, Any] | None,
    *,
    as_of_date: str | None,
) -> bool:
    if not isinstance(record, dict):
        return False
    if not _is_finite_num(record.get("value")):
        return False
    if str(record.get("unit") or "").strip() != MARKET_CAP_UNIT_USD_MILLIONS:
        return False
    period_end = str(record.get("period_end") or "").strip()
    filed_date = str(record.get("filed_date") or "").strip()
    source = str(record.get("source") or "").strip()
    source_reference = str(record.get("source_reference") or "").strip()
    if not period_end or not filed_date or not source or not source_reference:
        return False
    try:
        period_date = date.fromisoformat(period_end[:10])
        filed = date.fromisoformat(filed_date[:10])
        run_date = date.fromisoformat(str(as_of_date)[:10]) if as_of_date else None
    except ValueError:
        return False
    return bool(
        period_date <= filed
        and (run_date is None or (period_date <= run_date and filed <= run_date))
    )


def _market_cap_trace_provenance(
    value: float,
    *,
    as_of_date: str | None,
    quote_snapshot_id: str | None,
    source: str = "DERIVED_PRICE_TIMES_SHARES",
) -> dict[str, Any]:
    effective_date = str(as_of_date or "").strip()[:10]
    snapshot_id = str(quote_snapshot_id or "").strip()
    if not effective_date or not snapshot_id:
        return {}
    return {
        "value": float(value),
        "unit": MARKET_CAP_UNIT_USD_MILLIONS,
        "source": source,
        "period_end": effective_date,
        "filed_date": effective_date,
        "source_reference": f"QUOTE_SNAPSHOT:{snapshot_id}",
        "quote_snapshot_id": snapshot_id,
    }


def _derived_trace_provenance(
    value: float,
    *,
    unit: str,
    source: str,
    components: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if not components:
        return {}
    normalized_components: dict[str, dict[str, Any]] = {}
    period_ends: list[str] = []
    filed_dates: list[str] = []
    source_references: list[str] = []
    for name, raw_record in components.items():
        if not isinstance(raw_record, dict):
            return {}
        record = dict(raw_record)
        period_end = str(record.get("period_end") or "").strip()[:10]
        filed_date = str(record.get("filed_date") or "").strip()[:10]
        source_reference = str(record.get("source_reference") or "").strip()
        if not (
            record.get("value") is not None
            and str(record.get("unit") or "").strip()
            and str(record.get("source") or "").strip()
            and period_end
            and filed_date
            and source_reference
        ):
            return {}
        try:
            period_date = date.fromisoformat(period_end)
            filed = date.fromisoformat(filed_date)
        except ValueError:
            return {}
        if period_date > filed:
            return {}
        normalized_components[name] = record
        period_ends.append(period_end)
        filed_dates.append(filed_date)
        source_references.append(source_reference)
    period_end = max(period_ends)
    filed_date = max(filed_dates)
    if period_end > filed_date:
        return {}
    return {
        "value": float(value),
        "unit": unit,
        "source": source,
        "period_end": period_end,
        "filed_date": filed_date,
        "source_reference": (f"{source}:" + "|".join(sorted(dict.fromkeys(source_references)))),
        "components": normalized_components,
    }


def _canonical_current_valuation_metrics(
    ticker: str,
    *,
    as_of_date: str | None,
    market_cap_mm: float | None,
    market_cap_unit: str | None,
    quote_snapshot_id: str | None,
    v2_data_plane: bool = False,
    fundamental_values: dict[str, float] | None = None,
    fundamental_provenance: dict[str, dict[str, Any]] | None = None,
    market_cap_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Recompute current investment ratios from literal USD-million inputs."""

    if (
        market_cap_unit != MARKET_CAP_UNIT_USD_MILLIONS
        or not _is_finite_num(market_cap_mm)
        or float(market_cap_mm) <= 0
    ):
        return {
            "status": "NEEDS_DATA",
            "reason": "MARKET_CAP_USD_MILLIONS_REQUIRED",
            "metric_traces": {},
        }
    values = fundamental_values
    provenance = fundamental_provenance
    require_provenance = values is None
    if values is None:
        by_year = _packet_annual_fact_rows(
            ticker,
            as_of_date=as_of_date,
            v2_data_plane=v2_data_plane,
        )
        provenance_by_year = _packet_annual_fact_provenance_rows(
            ticker,
            as_of_date=as_of_date,
            v2_data_plane=v2_data_plane,
        )
        latest_year = max(by_year) if by_year else None
        values = by_year[latest_year] if latest_year is not None else {}
        provenance = provenance_by_year.get(latest_year, {}) if latest_year is not None else {}
    provenance = provenance or {}

    def usable_fact(name: str) -> float | None:
        value = values.get(name)
        if not _is_finite_num(value):
            return None
        if require_provenance and not _complete_fact_provenance(
            provenance.get(name),
            as_of_date=as_of_date,
        ):
            return None
        return float(value)

    market_cap = float(market_cap_mm)
    issuer_context = _V2_ISSUER_FACT_CONTEXT.get() or {}
    price_snapshot = (
        issuer_context.get("price_snapshot")
        if isinstance(issuer_context.get("price_snapshot"), dict)
        else {}
    )
    market_cap_input_provenance = dict(market_cap_provenance or {})
    if not market_cap_input_provenance:
        market_cap_input_provenance = _market_cap_trace_provenance(
            market_cap,
            as_of_date=price_snapshot.get("as_of_date") or as_of_date,
            quote_snapshot_id=quote_snapshot_id,
        )

    def fact_input_provenance(name: str, value: float | None) -> dict[str, Any]:
        record = provenance.get(name)
        if (
            value is None
            or not _complete_fact_provenance(record, as_of_date=as_of_date)
            or not math.isclose(
                float(record.get("value")),
                float(value),
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
        ):
            return {}
        return dict(record)

    debt = usable_fact("total_debt")
    cash = usable_fact("cash")
    missing_balance_inputs = [
        name for name, value in (("TOTAL_DEBT", debt), ("CASH", cash)) if value is None
    ]
    enterprise_value = market_cap + debt - cash if debt is not None and cash is not None else None
    operating_income = usable_fact("operating_income")
    da = usable_fact("depreciation_amortization")
    da_provenance = fact_input_provenance("depreciation_amortization", da)
    if da is None:
        depreciation = usable_fact("depreciation")
        intangible_amortization = usable_fact("intangible_amortization")
        da_parts = [value for value in (depreciation, intangible_amortization) if value is not None]
        da = sum(da_parts) if da_parts else None
        if da is not None:
            da_components = {
                name: record
                for name, value, record in (
                    (
                        "depreciation",
                        depreciation,
                        fact_input_provenance("depreciation", depreciation),
                    ),
                    (
                        "intangible_amortization",
                        intangible_amortization,
                        fact_input_provenance(
                            "intangible_amortization",
                            intangible_amortization,
                        ),
                    ),
                )
                if value is not None and record
            }
            if len(da_components) == len(da_parts):
                da_provenance = _derived_trace_provenance(
                    da,
                    unit=MARKET_CAP_UNIT_USD_MILLIONS,
                    source="DERIVED_DEPRECIATION_AND_AMORTIZATION",
                    components=da_components,
                )
    ebitda = (
        float(operating_income) + float(da)
        if _is_finite_num(operating_income) and _is_finite_num(da)
        else None
    )
    net_income = usable_fact("net_income")
    equity = usable_fact("equity")
    cfo = usable_fact("cfo")
    capex = usable_fact("capex")
    fcf, fcf_yield = _fcf_yield_from_inputs(cfo, capex, market_cap)
    enterprise_value_provenance = (
        _derived_trace_provenance(
            enterprise_value,
            unit=MARKET_CAP_UNIT_USD_MILLIONS,
            source="DERIVED_ENTERPRISE_VALUE",
            components={
                "market_cap_usd_millions": market_cap_input_provenance,
                "debt_usd_millions": fact_input_provenance("total_debt", debt),
                "cash_usd_millions": fact_input_provenance("cash", cash),
            },
        )
        if enterprise_value is not None
        else {}
    )
    ebitda_provenance = (
        _derived_trace_provenance(
            ebitda,
            unit=MARKET_CAP_UNIT_USD_MILLIONS,
            source="DERIVED_EBITDA",
            components={
                "operating_income_usd_millions": fact_input_provenance(
                    "operating_income",
                    operating_income,
                ),
                "depreciation_amortization_usd_millions": da_provenance,
            },
        )
        if ebitda is not None
        else {}
    )

    ev_to_ebitda = (
        float(enterprise_value) / float(ebitda)
        if _is_finite_num(enterprise_value)
        and _is_finite_num(ebitda)
        and not math.isclose(float(ebitda), 0.0)
        else None
    )
    pe = (
        market_cap / float(net_income)
        if _is_finite_num(net_income) and not math.isclose(float(net_income), 0.0)
        else None
    )
    price_to_book = (
        market_cap / float(equity)
        if _is_finite_num(equity) and not math.isclose(float(equity), 0.0)
        else None
    )
    traces: dict[str, Any] = {}
    if enterprise_value is not None and debt is not None and cash is not None:
        traces["enterprise_value"] = canonical_metric_trace(
            metric="enterprise_value",
            formula="market_cap_usd_millions + debt_usd_millions - cash_usd_millions",
            inputs={
                "market_cap_usd_millions": market_cap,
                "debt_usd_millions": debt,
                "cash_usd_millions": cash,
            },
            output=enterprise_value,
            recomputed_output=market_cap + debt - cash,
            output_unit=MARKET_CAP_UNIT_USD_MILLIONS,
            quote_snapshot_id=quote_snapshot_id,
            input_provenance={
                "market_cap_usd_millions": market_cap_input_provenance,
                "debt_usd_millions": fact_input_provenance("total_debt", debt),
                "cash_usd_millions": fact_input_provenance("cash", cash),
            },
        )
    if ev_to_ebitda is not None:
        traces["ev_to_ebitda"] = canonical_metric_trace(
            metric="ev_to_ebitda",
            formula="enterprise_value_usd_millions / ebitda_usd_millions",
            inputs={
                "enterprise_value_usd_millions": enterprise_value,
                "ebitda_usd_millions": ebitda,
            },
            output=ev_to_ebitda,
            recomputed_output=enterprise_value / float(ebitda),
            output_unit="ratio",
            quote_snapshot_id=quote_snapshot_id,
            input_provenance={
                "enterprise_value_usd_millions": enterprise_value_provenance,
                "ebitda_usd_millions": ebitda_provenance,
            },
        )
    if pe is not None:
        traces["pe"] = canonical_metric_trace(
            metric="pe",
            formula="market_cap_usd_millions / net_income_usd_millions",
            inputs={
                "market_cap_usd_millions": market_cap,
                "net_income_usd_millions": float(net_income),
            },
            output=pe,
            recomputed_output=market_cap / float(net_income),
            output_unit="ratio",
            quote_snapshot_id=quote_snapshot_id,
            input_provenance={
                "market_cap_usd_millions": market_cap_input_provenance,
                "net_income_usd_millions": fact_input_provenance(
                    "net_income",
                    net_income,
                ),
            },
        )
    if price_to_book is not None:
        traces["price_to_book"] = canonical_metric_trace(
            metric="price_to_book",
            formula="market_cap_usd_millions / equity_usd_millions",
            inputs={
                "market_cap_usd_millions": market_cap,
                "equity_usd_millions": float(equity),
            },
            output=price_to_book,
            recomputed_output=market_cap / float(equity),
            output_unit="ratio",
            quote_snapshot_id=quote_snapshot_id,
            input_provenance={
                "market_cap_usd_millions": market_cap_input_provenance,
                "equity_usd_millions": fact_input_provenance("equity", equity),
            },
        )
    if fcf_yield is not None and fcf is not None:
        traces["fcf_yield"] = canonical_metric_trace(
            metric="fcf_yield",
            formula=("(cfo_usd_millions - abs(capex_usd_millions)) / market_cap_usd_millions"),
            inputs={
                "cfo_usd_millions": float(cfo),
                "capex_usd_millions": float(capex),
                "market_cap_usd_millions": market_cap,
            },
            output=fcf_yield,
            recomputed_output=fcf / market_cap,
            output_unit="ratio",
            quote_snapshot_id=quote_snapshot_id,
            input_provenance={
                "cfo_usd_millions": fact_input_provenance("cfo", cfo),
                "capex_usd_millions": fact_input_provenance("capex", capex),
                "market_cap_usd_millions": market_cap_input_provenance,
            },
        )
    return {
        "status": "NEEDS_DATA" if missing_balance_inputs else "OK",
        "reason": (
            "MISSING_OR_UNPROVEN_" + "_AND_".join(missing_balance_inputs)
            if missing_balance_inputs
            else None
        ),
        "market_cap_mm": market_cap,
        "enterprise_value": enterprise_value,
        "ebitda": ebitda,
        "ev_to_ebitda": ev_to_ebitda,
        "pe": pe,
        "price_to_book": price_to_book,
        "fcf_yield": fcf_yield,
        "metric_traces": traces,
    }


def _historical_multiple_value(metric: str, price: Any, values: dict[str, float]) -> float | None:
    market_cap = _market_cap_statement_units(price, values)
    if metric == "pe":
        net_income = values.get("net_income")
        return (
            market_cap / float(net_income)
            if market_cap is not None and _is_num(net_income) and float(net_income) > 0
            else None
        )
    if metric == "ev_to_ebitda":
        enterprise_value = _enterprise_value_statement_units(price, values)
        operating_income = values.get("operating_income")
        da = _depreciation_amortization(values)
        ebitda = (
            float(operating_income) + float(da)
            if _is_num(operating_income) and _is_num(da)
            else None
        )
        return (
            enterprise_value / ebitda
            if enterprise_value is not None and _is_num(ebitda) and float(ebitda) > 0
            else None
        )
    if metric == "price_to_book":
        equity = values.get("equity")
        return (
            market_cap / float(equity)
            if market_cap is not None and _is_num(equity) and float(equity) > 0
            else None
        )
    if metric == "fcf_yield":
        cfo = values.get("cfo")
        capex = values.get("capex")
        _, fcf_yield = _fcf_yield_from_inputs(cfo, capex, market_cap)
        return fcf_yield
    return None


def _quantile(sorted_values: list[float], percentile: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * max(0.0, min(1.0, percentile))
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return (
        float(sorted_values[lower])
        + (float(sorted_values[upper]) - float(sorted_values[lower])) * weight
    )


def _percentile_within_history(current_value: Any, sorted_values: list[float]) -> float | None:
    if not _is_num(current_value) or not sorted_values:
        return None
    current = float(current_value)
    if len(sorted_values) == 1:
        return (
            50.0
            if current == float(sorted_values[0])
            else 0.0
            if current < float(sorted_values[0])
            else 100.0
        )
    if current <= float(sorted_values[0]):
        return 0.0
    if current >= float(sorted_values[-1]):
        return 100.0
    for idx in range(len(sorted_values) - 1):
        low = float(sorted_values[idx])
        high = float(sorted_values[idx + 1])
        if low <= current <= high:
            base = idx / (len(sorted_values) - 1) * 100.0
            next_base = (idx + 1) / (len(sorted_values) - 1) * 100.0
            if high == low:
                return base
            return base + ((current - low) / (high - low)) * (next_base - base)
    return None


def _historical_band_payload(
    metric: str, current_value: Any, history: list[dict[str, Any]]
) -> dict[str, Any]:
    values = [float(row["value"]) for row in history if _is_num(row.get("value"))]
    sorted_values = sorted(values)
    years = len(values)
    status = "OK" if years >= 5 else "INSUFFICIENT_HISTORY" if years > 0 else "DATA_MISSING"
    reasons: list[str] = []
    if years == 0:
        reasons.append("HISTORICAL_PRICE_OR_FUNDAMENTAL_DATA_MISSING")
    elif years < 5:
        reasons.append("HISTORICAL_MULTIPLE_HISTORY_INSUFFICIENT")
    if not _is_num(current_value):
        reasons.append("CURRENT_MULTIPLE_NOT_COMPUTABLE")
    return {
        "current_value": float(current_value) if _is_num(current_value) else None,
        "history": history,
        "range_min": _quantile(sorted_values, 0.0),
        "range_q1": _quantile(sorted_values, 0.25),
        "range_median": _quantile(sorted_values, 0.50),
        "range_q3": _quantile(sorted_values, 0.75),
        "range_max": _quantile(sorted_values, 1.0),
        "current_percentile": _percentile_within_history(current_value, sorted_values),
        "years_of_history": years,
        "status": status,
        "not_computable_reasons": reasons,
    }


def _historical_multiple_bands_from_inputs(
    *,
    by_year: dict[int, dict[str, float]],
    period_ends: dict[int, str],
    prices_by_year: dict[int, float],
    current_price: float | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    candidate_years = sorted(set(by_year) & set(period_ends))[-10:]
    for metric in HISTORICAL_MULTIPLE_KEYS:
        history: list[dict[str, Any]] = []
        for fiscal_year in candidate_years:
            price = prices_by_year.get(fiscal_year)
            value = _historical_multiple_value(metric, price, by_year[fiscal_year])
            if value is not None:
                history.append({"year": fiscal_year, "value": value})
        latest_values = by_year[candidate_years[-1]] if candidate_years else {}
        current_value = (
            _historical_multiple_value(metric, current_price, latest_values)
            if latest_values
            else None
        )
        payload[metric] = _historical_band_payload(metric, current_value, history)
    return payload


def _historical_multiple_bands(
    ticker: str,
    *,
    as_of_date: str | None,
    current_price: float | None,
    v2_data_plane: bool = False,
    require_explicit_split_lineage: bool = False,
) -> dict[str, Any]:
    if require_explicit_split_lineage:
        return {
            metric: {
                **_historical_band_payload(metric, None, []),
                "status": "NEEDS_DATA",
                "not_computable_reasons": ["HISTORICAL_PRICE_SHARE_SPLIT_BASIS_UNVERIFIED"],
            }
            for metric in HISTORICAL_MULTIPLE_KEYS
        }
    if not as_of_date:
        return {
            metric: _historical_band_payload(metric, None, [])
            for metric in HISTORICAL_MULTIPLE_KEYS
        }
    by_year = _packet_annual_fact_rows(
        ticker,
        as_of_date=as_of_date,
        v2_data_plane=v2_data_plane,
    )
    period_ends = _packet_annual_period_ends(
        ticker,
        as_of_date=as_of_date,
        v2_data_plane=v2_data_plane,
    )
    candidate_period_ends = {
        year: period_ends[year] for year in sorted(set(by_year) & set(period_ends))[-10:]
    }
    prices_by_year = _historical_fiscal_year_prices(
        ticker, candidate_period_ends, as_of_date=as_of_date
    )
    bands = _historical_multiple_bands_from_inputs(
        by_year=by_year,
        period_ends=period_ends,
        prices_by_year=prices_by_year,
        current_price=current_price,
    )
    if not prices_by_year:
        for metric_payload in bands.values():
            reasons = list(metric_payload.get("not_computable_reasons") or [])
            if "HISTORICAL_PRICE_NOT_AVAILABLE" not in reasons:
                reasons.insert(0, "HISTORICAL_PRICE_NOT_AVAILABLE")
            metric_payload["not_computable_reasons"] = reasons
    return bands


def _peer_valuation_context(
    ticker: str,
    *,
    sector: str | None,
    as_of_date: str | None,
    canonical_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if isinstance(canonical_metrics, dict) and canonical_metrics.get("status") == "OK":
        return {
            "status": "PEER_DISTRIBUTION_SUPPRESSED_AMBIGUOUS_UNITS",
            "sector": sector,
            "peer_count": 0,
            "peer_set_used": [],
            "metrics": {
                "ev_ebitda": {
                    "label": PEER_METRIC_DISPLAY_NAMES.get("ev_ebitda", "EV/EBITDA"),
                    "stock_value": canonical_metrics.get("ev_to_ebitda"),
                    "sector_median": None,
                    "sector_q1": None,
                    "sector_q3": None,
                    "percentile_rank": None,
                },
                "fcf_yield": {
                    "label": PEER_METRIC_DISPLAY_NAMES.get("fcf_yield", "FCF yield"),
                    "stock_value": canonical_metrics.get("fcf_yield"),
                    "sector_median": None,
                    "sector_q1": None,
                    "sector_q3": None,
                    "percentile_rank": None,
                },
            },
            "metric_traces": dict(canonical_metrics.get("metric_traces") or {}),
        }
    try:
        peer = compute_peer_relative_metrics(
            ticker, as_of_date or date.today().isoformat(), sector=sector
        )
    except Exception as exc:
        return {"status": "ERROR", "error": str(exc)}
    metrics: dict[str, Any] = {}
    ticker_metrics = (
        peer.get("ticker_metrics") if isinstance(peer.get("ticker_metrics"), dict) else {}
    )
    medians = peer.get("sector_medians") if isinstance(peer.get("sector_medians"), dict) else {}
    q1 = peer.get("sector_q1") if isinstance(peer.get("sector_q1"), dict) else {}
    q3 = peer.get("sector_q3") if isinstance(peer.get("sector_q3"), dict) else {}
    percentiles = (
        peer.get("percentile_ranks") if isinstance(peer.get("percentile_ranks"), dict) else {}
    )
    for raw_metric in ("EV/EBITDA", "FCF yield"):
        metric_key = normalize_peer_metric(raw_metric)
        if not metric_key:
            continue
        metrics[metric_key] = {
            "label": PEER_METRIC_DISPLAY_NAMES.get(metric_key, raw_metric),
            "stock_value": ticker_metrics.get(metric_key),
            "sector_median": medians.get(metric_key),
            "sector_q1": q1.get(metric_key),
            "sector_q3": q3.get(metric_key),
            "percentile_rank": percentiles.get(metric_key),
        }
    return {
        "status": peer.get("status"),
        "sector": peer.get("sector"),
        "peer_count": peer.get("peer_count"),
        "peer_set_used": list(peer.get("peer_set_used") or []),
        "metrics": metrics,
    }


def _returns_on_capital(
    packet: TickerSignalPacket,
    *,
    as_of_date: str | None = None,
    v2_data_plane: bool = False,
) -> dict[str, Any]:
    computed = _returns_on_capital_metrics(
        packet.ticker,
        as_of_date=as_of_date,
        v2_data_plane=v2_data_plane,
    )
    return {
        "peer_position": packet.peer_position,
        "roic_vs_median": packet.roic_vs_median,
        "operating_margin_vs_median": packet.op_margin_vs_median,
        "revenue_growth_vs_median": packet.revenue_growth_vs_median,
        "moat_score": packet.moat_score,
        **computed,
    }


def _cash_conversion(
    packet: TickerSignalPacket,
    *,
    as_of_date: str | None = None,
    v2_data_plane: bool = False,
) -> dict[str, Any]:
    qctx = _quality_context(packet)
    keys = [
        "cash_conversion",
        "cash_conversion_ratio",
        "fcf_margin",
        "owner_earnings_yield",
        "owner_earnings_yield_ev_3y",
    ]
    payload = {key: qctx.get(key) for key in keys if key in qctx}
    payload.update(
        _cash_conversion_cycle_metrics(
            packet.ticker,
            as_of_date=as_of_date,
            v2_data_plane=v2_data_plane,
        )
    )
    has_computed_metric = any(
        _is_num(payload.get(key))
        for key in ("cash_conversion_ratio", "fcf_margin", "cash_conversion_cycle")
    )
    payload["cash_conversion_status"] = "AVAILABLE" if has_computed_metric else "UNKNOWN"
    return payload


def _balance_sheet(packet: TickerSignalPacket) -> dict[str, Any]:
    solvency = _solvency_payload(packet)
    return {
        "solvency_risk": packet.solvency_risk,
        "solvency_signals": list(solvency.get("signals") or []),
        "solvency_details": solvency.get("details"),
        "negative_equity": solvency.get("negative_equity"),
        "current_ratio": solvency.get("current_ratio"),
        "cash_runway_quarters": solvency.get("cash_runway_quarters"),
        "going_concern_language": solvency.get("going_concern_language"),
        "going_concern_asserted": going_concern_asserted(solvency),
        "going_concern_assertions": [
            dict(item)
            for item in solvency.get("going_concern_assertions") or []
            if isinstance(item, dict)
        ],
        "no_assurance_financing": solvency.get("no_assurance_financing"),
        "debt_due_within_12mo": solvency.get("debt_due_within_12mo"),
        "downside_risk_class": packet.downside_risk_class,
    }


def _capital_allocation(
    packet: TickerSignalPacket,
    *,
    as_of_date: str | None = None,
    v2_data_plane: bool = False,
) -> dict[str, Any]:
    qctx = _quality_context(packet)
    keys = [
        "dilution_rate_shares_cagr",
        "capital_allocation_score",
        "capital_allocation_class",
        "capital_allocation_reason_codes",
    ]
    payload = {key: qctx.get(key) for key in keys if key in qctx}
    share_count = _share_count_cagr_metrics(
        packet.ticker,
        as_of_date=as_of_date,
        v2_data_plane=v2_data_plane,
    )
    payload.update(share_count)
    if (
        payload.get("dilution_rate_shares_cagr") is None
        and share_count.get("share_count_cagr") is not None
    ):
        payload["dilution_rate_shares_cagr"] = share_count["share_count_cagr"]
    payload["valuation_supports"] = list(packet.valuation_supports or [])
    payload["valuation_headwinds"] = list(packet.valuation_headwinds or [])
    return payload


def _accounting_quality(packet: TickerSignalPacket) -> dict[str, Any]:
    qctx = _quality_context(packet)
    filing_metadata = (
        packet.filing_risk_metadata if isinstance(packet.filing_risk_metadata, dict) else {}
    )
    return {
        "confidence_class": packet.confidence_class,
        "earnings_quality": qctx.get("earnings_quality"),
        "filing_risk_status": packet.filing_risk_status,
        "filing_risk_evidence_status": filing_metadata.get("evidence_status"),
        "filing_risk_source_accession": filing_metadata.get("source_accession"),
        "filing_risk_source_form_type": filing_metadata.get("source_form_type"),
        "filing_risk_source_filing_date": filing_metadata.get("source_filing_date"),
        "filing_risk_source_filing_age_days": filing_metadata.get("source_filing_age_days"),
        "filing_risk_text_chars": filing_metadata.get("risk_text_chars"),
        "filing_risk_warnings": list(filing_metadata.get("warnings") or []),
        "filing_risk_signals": dict(packet.filing_risk_signals or {}),
        "anomaly_count": packet.anomaly_count,
        "research_status": packet.research_status,
        "model_fit_warnings": list(packet.model_fit_warnings or []),
        "valuation_provenance_status": packet.valuation_provenance_status,
        "valuation_provenance_blockers": list(packet.valuation_provenance_blockers or []),
    }


def _expectations_gap_from_db(
    ticker: str,
    *,
    as_of_date: str | None,
    pipeline_version: str = "v1",
    issuer_cik: str | None = None,
    issuer_aliases: tuple[str, ...] = (),
    price_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Read the canonical expectations-gap sub-dict from the persisted reverse_dcf row.

    Mirrors the valuations-table read pattern used elsewhere in this module. The
    expectations_gap dict is persisted at the top level of the reverse_dcf
    outputs_json blob (see valuation_writer._run_reverse_dcf). Returns
    ``{'bucket': 'EXPECTATIONS_GAP_UNRELIABLE'}`` whenever no row exists or the
    persisted outputs lack the key, so the signal is silent rather than wrong.
    """
    default = {"bucket": "EXPECTATIONS_GAP_UNRELIABLE"}
    try:
        from app.db import get_db

        with get_db() as conn:
            if str(pipeline_version or "v1").lower() == "v2":
                if not as_of_date:
                    return {
                        **default,
                        "provenance_status": "INVALID",
                        "provenance_blockers": ["V2_REVERSE_DCF_ASOF_MISSING"],
                    }
                provenance = validate_v2_valuation_method_provenance(
                    conn,
                    ticker,
                    method="reverse_dcf",
                    as_of_date=as_of_date,
                    issuer_cik=issuer_cik,
                    issuer_aliases=issuer_aliases,
                    price_snapshot=price_snapshot,
                )
                blockers = [str(item) for item in provenance.get("mismatch_reasons") or []]
                if blockers:
                    return {
                        **default,
                        "provenance_status": "INVALID",
                        "provenance_blockers": blockers,
                    }
                payload = provenance.get("outputs")
                if not isinstance(payload, dict):
                    return {
                        **default,
                        "provenance_status": "INVALID",
                        "provenance_blockers": ["V2_REVERSE_DCF_OUTPUTS_INVALID"],
                    }
            else:
                row = latest_decision_eligible_valuation_row(
                    conn,
                    ticker=ticker,
                    method="reverse_dcf",
                    as_of_date=as_of_date,
                    exact_as_of_date=as_of_date is not None,
                )
                payload = None
    except Exception as exc:
        if str(pipeline_version or "v1").lower() == "v2":
            return {
                **default,
                "provenance_status": "ERROR",
                "provenance_blockers": [f"V2_REVERSE_DCF_PROVENANCE_ERROR:{type(exc).__name__}"],
            }
        return dict(default)
    is_v2 = str(pipeline_version or "v1").lower() == "v2"
    if not is_v2:
        if not row or not row["outputs_json"]:
            return dict(default)
        try:
            payload = json.loads(row["outputs_json"])
        except Exception:
            return dict(default)
    gap = payload.get("expectations_gap") if isinstance(payload, dict) else None
    if not isinstance(gap, dict) or not gap.get("bucket"):
        if is_v2:
            return {
                **default,
                "provenance_status": "VALIDATED_NO_SIGNAL",
                "provenance_blockers": [],
            }
        return dict(default)
    enriched = dict(gap)
    if "implied_growth" not in enriched:
        outputs = payload.get("outputs") if isinstance(payload.get("outputs"), dict) else {}
        enriched["implied_growth"] = outputs.get("implied_growth")
    # Coerce the non-numeric 'UNKNOWN' sentinel (per the reverse_dcf fix) to
    # None so the raw implied_growth consumed by the cross-sectional gap
    # factor is always either a float or None, never a string.
    enriched["implied_growth"] = _optional_float(enriched.get("implied_growth"))
    # Never surface a saturated / known-unreliable implied_growth. When the
    # reverse-DCF solver saturated, the bucket is UNRELIABLE (or the saturation
    # flag is set) while outputs.implied_growth still holds the clipped bound;
    # feeding that bound to the gap factor systematically rewards distressed
    # names. Drop it to None so the member is treated as gap-absent (the
    # factor renormalizes over its remaining present factors).
    if enriched.get("bucket") == "EXPECTATIONS_GAP_UNRELIABLE" or enriched.get(
        "implied_growth_saturated"
    ):
        enriched["implied_growth"] = None
    if is_v2:
        enriched["provenance_status"] = "VALIDATED"
        enriched["provenance_blockers"] = []
    return enriched


def _valuation(
    packet: TickerSignalPacket,
    anchor_method: str | None,
    anchor: float | None,
    *,
    sector: str | None,
    as_of_date: str | None,
    v2_data_plane: bool = False,
    market_cap_mm: float | None = None,
    market_cap_unit: str | None = None,
    quote_snapshot_id: str | None = None,
    require_explicit_split_lineage: bool = False,
) -> dict[str, Any]:
    pzd = _pricing_zone_detail(packet)
    discount = _discount_to_anchor(packet.current_price, anchor)
    generic_anchor_method, generic_anchor = _generic_valuation_anchor(packet)
    tech_method, tech_anchor = _technology_adjusted_anchor(packet)
    fcf_yield = _fcf_yield_metrics(
        packet.ticker,
        as_of_date=as_of_date,
        current_price=_optional_float(packet.current_price),
        v2_data_plane=v2_data_plane,
        market_cap_override_mm=market_cap_mm,
        market_cap_unit=market_cap_unit,
        quote_snapshot_id=quote_snapshot_id,
    )
    canonical_metrics = _canonical_current_valuation_metrics(
        packet.ticker,
        as_of_date=as_of_date,
        market_cap_mm=market_cap_mm,
        market_cap_unit=market_cap_unit,
        quote_snapshot_id=quote_snapshot_id,
        v2_data_plane=v2_data_plane,
    )
    if canonical_metrics.get("status") == "OK":
        fcf_yield = {
            **fcf_yield,
            "fcf_yield": canonical_metrics.get("fcf_yield"),
            "metric_trace": (canonical_metrics.get("metric_traces", {}).get("fcf_yield")),
        }
    peer_relative_valuation = _peer_valuation_context(
        packet.ticker,
        sector=sector,
        as_of_date=as_of_date,
        canonical_metrics=(canonical_metrics if require_explicit_split_lineage else None),
    )
    historical_multiples = _historical_multiple_bands(
        packet.ticker,
        as_of_date=as_of_date,
        current_price=_optional_float(packet.current_price),
        v2_data_plane=v2_data_plane,
        require_explicit_split_lineage=require_explicit_split_lineage,
    )
    issuer_context = _V2_ISSUER_FACT_CONTEXT.get() or {}
    if v2_data_plane and packet.valuation_provenance_status != "VALIDATED":
        expectations_gap = {
            "bucket": "EXPECTATIONS_GAP_UNRELIABLE",
            "provenance_status": "SUPPRESSED_INVALID_SCORECARD",
            "provenance_blockers": list(packet.valuation_provenance_blockers or []),
        }
    else:
        expectations_gap = _expectations_gap_from_db(
            packet.ticker,
            as_of_date=as_of_date,
            pipeline_version="v2" if v2_data_plane else "v1",
            issuer_cik=issuer_context.get("issuer_cik"),
            issuer_aliases=tuple(issuer_context.get("issuer_aliases") or ()),
            price_snapshot=(issuer_context.get("price_snapshot") if v2_data_plane else None),
        )
    return {
        "current_price": packet.current_price,
        "anchor_method": anchor_method,
        "valuation_anchor": anchor,
        "discount_to_anchor": discount,
        "generic_anchor_method": generic_anchor_method,
        "generic_anchor_value": generic_anchor,
        "sector_specific_anchor_method": tech_method if tech_anchor is not None else None,
        "sector_specific_anchor_value": tech_anchor,
        "available_methods": _available_valuation_methods(packet, sector=sector),
        "dcf_value": packet.dcf_value,
        "epv_value": packet.epv_value,
        "graham_value": packet.graham_value,
        "ncav_value": packet.ncav_value,
        "insurance_value": packet.insurance_value,
        "margin_of_safety_verdict": packet.margin_of_safety_verdict,
        "gate_action": packet.gate_verdict or pzd.get("gate_action"),
        "method_tension_type": packet.method_tension_type,
        "growth_dependency_ratio": packet.growth_dependency_ratio,
        "consensus_direction": packet.consensus_direction,
        "intrinsic_range_low": packet.intrinsic_range_low,
        "intrinsic_range_high": packet.intrinsic_range_high,
        "fcf_yield": fcf_yield.get("fcf_yield"),
        "ttm_fcf": fcf_yield.get("ttm_fcf"),
        "fcf_yield_reasons": fcf_yield.get("fcf_yield_reasons"),
        "fcf_yield_source": fcf_yield.get("fcf_yield_source"),
        "enterprise_value": canonical_metrics.get("enterprise_value"),
        "ev_to_ebitda": canonical_metrics.get("ev_to_ebitda"),
        "pe": canonical_metrics.get("pe"),
        "price_to_book": canonical_metrics.get("price_to_book"),
        "canonical_metric_status": canonical_metrics.get("status"),
        "canonical_metric_reason": canonical_metrics.get("reason"),
        "metric_traces": dict(canonical_metrics.get("metric_traces") or {}),
        "peer_relative_valuation": peer_relative_valuation,
        "historical_multiples": historical_multiples,
        "implied_growth": expectations_gap.get("implied_growth"),
        "supportable_growth": expectations_gap.get("supportable_growth"),
        "expectations_gap": expectations_gap.get("gap"),
        # Saturated-LOW CHEAP gaps are FLOORS (gap <= value), not exact —
        # downstream renders must not present them as point estimates.
        "expectations_gap_is_upper_bound": bool(expectations_gap.get("gap_is_upper_bound", False)),
        "expectations_gap_bucket": expectations_gap.get("bucket"),
        "expectations_gap_provenance_status": expectations_gap.get("provenance_status"),
        "expectations_gap_provenance_blockers": list(
            expectations_gap.get("provenance_blockers") or []
        ),
        "generic_valuation_valid": (
            packet.insurance_packet.get("generic_valuation_valid")
            if isinstance(packet.insurance_packet, dict)
            else None
        ),
    }


def _expected_return(
    packet: TickerSignalPacket, anchor_method: str | None, anchor: float | None
) -> dict[str, Any]:
    discount = _discount_to_anchor(packet.current_price, anchor)
    status = "SCENARIO_REQUIRED"
    if discount is None:
        status = "INSUFFICIENT_INPUTS"
    elif discount < 0:
        status = "ANCHOR_BELOW_PRICE"
    return {
        "status": status,
        "base_anchor_method": anchor_method,
        "base_anchor_discount": discount,
        "scenario_engine_required": True,
        "horizon_years": [5, 10],
    }


def _score_components(packet: TickerSignalPacket, discount: float | None) -> dict[str, Any]:
    return {
        "moat_score": packet.moat_score,
        "peer_position": packet.peer_position,
        "solvency_risk": packet.solvency_risk,
        "discount_to_anchor": discount,
        "anomaly_count": packet.anomaly_count,
        "growth_dependency_ratio": packet.growth_dependency_ratio,
    }


def _cap_context_from_signal_packet(
    packet: TickerSignalPacket,
) -> dict[str, Any]:
    """Carry the canonical cap/quote snapshot when no separate classification is supplied."""

    if (
        packet.market_cap_mm is None
        and not packet.market_cap_method
        and not packet.cap_stage_quote_snapshot_id
    ):
        return {}

    return {
        "market_cap_mm": packet.market_cap_mm,
        "market_cap_unit": packet.market_cap_unit,
        "cap_source": packet.market_cap_source,
        "cap_effective_as_of_date": packet.market_cap_effective_as_of_date,
        "cap_source_kind": packet.market_cap_source_kind,
        "cap_source_name": packet.market_cap_source_name,
        "cap_source_url": packet.market_cap_source_url,
        "cap_confidence": packet.market_cap_confidence,
        "cap_method": packet.market_cap_method,
        "market_cap_derivation": dict(packet.market_cap_derivation),
        "current_price": packet.current_price,
        "current_price_as_of_date": packet.current_price_as_of_date,
        "current_price_currency": packet.current_price_currency,
        "current_price_source": packet.current_price_source,
        "current_price_source_url": packet.current_price_source_url,
        "current_price_basis": packet.price_basis,
        "current_raw_price": packet.raw_price,
        "current_split_adjustment_factor": packet.split_adjustment_factor,
        "current_split_effective_date": packet.split_effective_date,
        "price_used": packet.cap_stage_price,
        "price_as_of_date": packet.cap_stage_price_as_of_date,
        "price_currency": packet.cap_stage_price_currency,
        "price_source": packet.cap_stage_price_source,
        "price_source_url": packet.cap_stage_price_source_url,
        "quote_snapshot_id": packet.cap_stage_quote_snapshot_id,
        "price_basis": packet.price_basis,
        "raw_price": packet.raw_price,
        "split_adjustment_factor": packet.split_adjustment_factor,
        "split_effective_date": packet.split_effective_date,
        "split_lineage_proof": packet.split_lineage_proof,
        "shares_mm": packet.shares_outstanding_mm,
        "raw_shares_outstanding_mm": packet.raw_shares_outstanding_mm,
        "raw_shares_source_value": packet.raw_shares_source_value,
        "raw_shares_source_unit": packet.raw_shares_source_unit,
        "shares_unit": packet.shares_unit,
        "shares_basis": packet.shares_basis,
        "shares_period_end": packet.shares_as_of_date,
        "shares_filed_date": packet.shares_filed_date,
        "shares_source": packet.shares_source,
        "shares_source_url": packet.shares_source_url,
        "issuer_quote_ratio": packet.issuer_quote_ratio,
        "issuer_cik": packet.issuer_cik,
        "issuer_primary_ticker": packet.issuer_primary_ticker,
        "issuer_listed_tickers": list(packet.issuer_listed_tickers),
        "security_role": packet.security_role,
        "is_secondary_class": packet.is_secondary_class,
        "is_adr": packet.is_adr,
        "adr_ratio": packet.adr_ratio,
        "share_class_ratio": packet.share_class_ratio,
        "identity_source": packet.identity_source,
        "identity_source_url": packet.identity_source_url,
        "identity_as_of_date": packet.identity_as_of_date,
        "identity_confidence": packet.identity_confidence,
        "ratio_source_url": packet.ratio_source_url,
        "ratio_source_accession": packet.ratio_source_accession,
        "ratio_security_symbol": packet.ratio_security_symbol,
    }


def build_sector_company_financial_packet(
    packet: TickerSignalPacket,
    *,
    sector: str | None = None,
    as_of_date: str | None = None,
    cap_classification: dict[str, Any] | None = None,
    pipeline_version: str = "v1",
) -> SectorCompanyFinancialPacket:
    """Map one alpha signal packet into the sector financial contract.

    ``cap_classification`` is the band-filter chain result computed once at
    candidate loading (candidate_selection["cap_classifications"][ticker]);
    it is propagated, never recomputed here.
    """

    cap_info = (
        cap_classification
        if isinstance(cap_classification, dict) and cap_classification
        else _cap_context_from_signal_packet(packet)
    )
    v2_data_plane = str(pipeline_version or "v1").lower() == "v2"
    cap_stage_price = _optional_float(cap_info.get("price_used"))
    current_price_currency = str(cap_info.get("current_price_currency") or "").strip().upper()
    valuation_price = _optional_float(cap_info.get("current_price"))
    if valuation_price is not None and current_price_currency != "USD":
        valuation_price = None
    valuation_price_from_current = valuation_price is not None
    cap_price_currency = str(cap_info.get("price_currency") or "").strip().upper()
    valuation_price_from_cap = valuation_price is None and cap_price_currency == "USD"
    if valuation_price_from_cap:
        valuation_price = cap_stage_price
    if cap_info or v2_data_plane:
        # Both active pipelines consume the cap-stage quote contract. Missing
        # or contradictory evidence stays missing so the integrity gate can
        # stop paid reasoning instead of falling back to a scorecard/live quote.
        packet = replace(packet, current_price=valuation_price)

    anchor_method, anchor = _valuation_anchor(packet, sector=sector)
    blockers = _blockers(packet, anchor)
    caps = _confidence_caps(packet)
    discount = _discount_to_anchor(packet.current_price, anchor)
    cap_mm = cap_info.get("market_cap_mm")
    issuer_aliases = tuple(
        dict.fromkeys(
            str(item).strip().upper()
            for item in (
                packet.ticker,
                cap_info.get("issuer_primary_ticker"),
                cap_info.get("primary_ticker"),
                *(cap_info.get("issuer_listed_tickers") or ()),
                *(cap_info.get("issuer_aliases") or ()),
            )
            if str(item or "").strip()
        )
    )
    issuer_context_token = None
    if cap_info or v2_data_plane:
        issuer_context_token = _V2_ISSUER_FACT_CONTEXT.set(
            {
                "ticker": packet.ticker,
                "issuer_cik": cap_info.get("issuer_cik") or cap_info.get("cik"),
                "issuer_aliases": issuer_aliases,
                "price_snapshot": {
                    "price": valuation_price,
                    "as_of_date": (
                        cap_info.get("current_price_as_of_date")
                        if valuation_price_from_current
                        else cap_info.get("price_as_of_date")
                        if valuation_price_from_cap
                        else None
                    ),
                    "currency": (
                        cap_info.get("current_price_currency")
                        if valuation_price_from_current
                        else cap_info.get("price_currency")
                        if valuation_price_from_cap
                        else None
                    ),
                    "source": (
                        cap_info.get("current_price_source")
                        if valuation_price_from_current
                        else cap_info.get("price_source")
                        if valuation_price_from_cap
                        else None
                    ),
                    "url": (
                        cap_info.get("current_price_source_url")
                        if valuation_price_from_current
                        else cap_info.get("price_source_url")
                        if valuation_price_from_cap
                        else None
                    ),
                },
            }
        )
    try:
        business_quality = _business_quality(
            packet,
            as_of_date=as_of_date,
            v2_data_plane=v2_data_plane,
        )
        returns_on_capital = _returns_on_capital(
            packet,
            as_of_date=as_of_date,
            v2_data_plane=v2_data_plane,
        )
        cash_conversion = _cash_conversion(
            packet,
            as_of_date=as_of_date,
            v2_data_plane=v2_data_plane,
        )
        capital_allocation = _capital_allocation(
            packet,
            as_of_date=as_of_date,
            v2_data_plane=v2_data_plane,
        )
        valuation = _valuation(
            packet,
            anchor_method,
            anchor,
            sector=sector,
            as_of_date=as_of_date,
            v2_data_plane=v2_data_plane,
            market_cap_mm=(float(cap_mm) if isinstance(cap_mm, (int, float)) else None),
            market_cap_unit=cap_info.get("market_cap_unit"),
            quote_snapshot_id=cap_info.get("quote_snapshot_id"),
            require_explicit_split_lineage=bool(cap_info),
        )
    finally:
        if issuer_context_token is not None:
            _V2_ISSUER_FACT_CONTEXT.reset(issuer_context_token)
    current_price_as_of_date = (
        cap_info.get("current_price_as_of_date")
        if valuation_price_from_current
        else cap_info.get("price_as_of_date")
        if valuation_price_from_cap
        else None
    )
    selected_price_currency = (
        cap_info.get("current_price_currency")
        if valuation_price_from_current
        else cap_info.get("price_currency")
        if valuation_price_from_cap
        else None
    )
    selected_price_source = (
        cap_info.get("current_price_source")
        if valuation_price_from_current
        else cap_info.get("price_source")
        if valuation_price_from_cap
        else None
    )
    selected_price_source_url = (
        cap_info.get("current_price_source_url")
        if valuation_price_from_current
        else cap_info.get("price_source_url")
        if valuation_price_from_cap
        else None
    )
    selected_price_basis = (
        cap_info.get("current_price_basis")
        if valuation_price_from_current
        else cap_info.get("price_basis")
    ) or None
    selected_raw_price = _optional_float(
        cap_info.get("current_raw_price")
        if valuation_price_from_current
        else cap_info.get("raw_price")
    )
    selected_split_factor = _optional_float(
        cap_info.get("current_split_adjustment_factor")
        if valuation_price_from_current
        else cap_info.get("split_adjustment_factor")
    )
    selected_split_effective_date = (
        cap_info.get("current_split_effective_date")
        if valuation_price_from_current
        else cap_info.get("split_effective_date")
    )
    current_quote_snapshot_id = None
    if valuation_price is not None:
        current_quote_snapshot_id = stable_quote_hash(
            ticker=packet.ticker,
            price=valuation_price,
            as_of_date=current_price_as_of_date,
            currency=selected_price_currency,
            source=selected_price_source,
            source_url=selected_price_source_url,
            price_basis=selected_price_basis,
            raw_price=selected_raw_price,
            split_adjustment_factor=selected_split_factor,
            split_effective_date=selected_split_effective_date,
        )
    cap_stage_quote_snapshot_id = cap_info.get("quote_snapshot_id")
    if cap_stage_price is not None and not cap_stage_quote_snapshot_id:
        cap_stage_quote_snapshot_id = stable_quote_hash(
            ticker=packet.ticker,
            price=cap_stage_price,
            as_of_date=cap_info.get("price_as_of_date"),
            currency=cap_info.get("price_currency"),
            source=cap_info.get("price_source"),
            source_url=cap_info.get("price_source_url"),
            price_basis=cap_info.get("price_basis"),
            raw_price=cap_info.get("raw_price"),
            split_adjustment_factor=cap_info.get("split_adjustment_factor"),
            split_effective_date=cap_info.get("split_effective_date"),
        )
    shares_mm = _optional_float(
        cap_info.get("shares_mm")
        if cap_info.get("shares_mm") is not None
        else cap_info.get("shares_outstanding_mm")
    )
    raw_shares_mm = _optional_float(cap_info.get("raw_shares_outstanding_mm"))
    raw_shares_source_value = _optional_float(cap_info.get("raw_shares_source_value"))
    raw_shares_source_unit = cap_info.get("raw_shares_source_unit")
    if raw_shares_mm is None and cap_info.get("shares_basis") == SHARES_BASIS_UNADJUSTED:
        raw_shares_mm = shares_mm
    issuer_quote_ratio = _optional_float(cap_info.get("issuer_quote_ratio"))
    if issuer_quote_ratio is None and cap_info.get("is_adr") is True:
        issuer_quote_ratio = _optional_float(cap_info.get("adr_ratio"))
    if issuer_quote_ratio is None and cap_info.get("is_secondary_class") is True:
        issuer_quote_ratio = _optional_float(cap_info.get("share_class_ratio"))
    metric_traces = dict(valuation.get("metric_traces") or {})
    if (
        _is_finite_num(cap_mm)
        and float(cap_mm) > 0
        and _is_finite_num(valuation_price)
        and _is_finite_num(shares_mm)
        and issuer_quote_ratio is not None
        and issuer_quote_ratio > 0
    ):
        recomputed_cap = float(valuation_price) * float(shares_mm) / issuer_quote_ratio
        metric_traces["market_cap_mm"] = canonical_metric_trace(
            metric="market_cap_mm",
            formula="current_price * shares_outstanding_mm / issuer_quote_ratio",
            inputs={
                "current_price": float(valuation_price),
                "shares_outstanding_mm": float(shares_mm),
                "issuer_quote_ratio": issuer_quote_ratio,
            },
            output=float(cap_mm),
            recomputed_output=recomputed_cap,
            output_unit=MARKET_CAP_UNIT_USD_MILLIONS,
            quote_snapshot_id=current_quote_snapshot_id,
            input_provenance={
                "current_price": {
                    "value": float(valuation_price),
                    "unit": PRICE_UNIT_USD_PER_SHARE,
                    "source": selected_price_source,
                    "period_end": current_price_as_of_date,
                    "filed_date": current_price_as_of_date,
                    "source_reference": selected_price_source_url,
                    "basis": selected_price_basis,
                    "raw_value": selected_raw_price,
                    "split_adjustment_factor": selected_split_factor,
                    "split_effective_date": selected_split_effective_date,
                },
                "shares_outstanding_mm": {
                    "value": float(shares_mm),
                    "raw_value": raw_shares_mm,
                    "raw_source_value": raw_shares_source_value,
                    "raw_source_unit": raw_shares_source_unit,
                    "normalized_value": float(shares_mm),
                    "normalized_unit": cap_info.get("shares_unit"),
                    "unit": cap_info.get("shares_unit"),
                    "source": cap_info.get("shares_source") or cap_info.get("cap_source_name"),
                    "period_end": cap_info.get("shares_period_end")
                    or cap_info.get("shares_as_of_date"),
                    "filed_date": cap_info.get("shares_filed_date"),
                    "source_reference": cap_info.get("shares_source_url")
                    or cap_info.get("cap_source_url"),
                    "basis": cap_info.get("shares_basis"),
                    "split_adjustment_factor": selected_split_factor,
                    "split_effective_date": selected_split_effective_date,
                },
                "issuer_quote_ratio": {
                    "value": issuer_quote_ratio,
                    "unit": "ratio",
                    "source": (cap_info.get("identity_source") or "ISSUER_SECURITY_IDENTITY"),
                    "period_end": (cap_info.get("identity_as_of_date") or current_price_as_of_date),
                    "filed_date": (cap_info.get("identity_as_of_date") or current_price_as_of_date),
                    "source_reference": (
                        cap_info.get("ratio_source_url")
                        or cap_info.get("identity_source_url")
                        or f"ISSUER_SECURITY_IDENTITY:{packet.ticker}"
                    ),
                },
            },
            rel_tol=1e-6,
            abs_tol=1e-6,
        )
    cap_derivation = cap_info.get("market_cap_derivation")
    if not isinstance(cap_derivation, dict):
        raw_derivation = cap_info.get("cap_derivation_json")
        try:
            cap_derivation = json.loads(raw_derivation) if raw_derivation else {}
        except (TypeError, ValueError):
            cap_derivation = {}
    return SectorCompanyFinancialPacket(
        ticker=packet.ticker,
        financial_status=_financial_status(blockers, caps),
        model_fit_status=_model_fit_status(packet, sector=sector),
        data_quality_status=_data_quality_status(packet, anchor),
        market_cap_category=cap_info.get("cap_band") or None,
        market_cap_mm=float(cap_mm) if isinstance(cap_mm, (int, float)) else None,
        market_cap_unit=cap_info.get("market_cap_unit"),
        market_cap_source=cap_info.get("cap_source") or None,
        market_cap_effective_as_of_date=cap_info.get("cap_effective_as_of_date"),
        market_cap_source_kind=cap_info.get("cap_source_kind"),
        market_cap_source_name=cap_info.get("cap_source_name"),
        market_cap_source_url=cap_info.get("cap_source_url"),
        market_cap_confidence=cap_info.get("cap_confidence"),
        cap_stage_price=cap_stage_price,
        cap_stage_price_as_of_date=cap_info.get("price_as_of_date"),
        cap_stage_price_currency=cap_info.get("price_currency"),
        cap_stage_price_source=cap_info.get("price_source"),
        cap_stage_price_source_url=cap_info.get("price_source_url"),
        cap_stage_price_confidence=cap_info.get("price_confidence"),
        cap_stage_quote_snapshot_id=cap_stage_quote_snapshot_id,
        issuer_cik=cap_info.get("issuer_cik") or cap_info.get("cik"),
        issuer_primary_ticker=cap_info.get("issuer_primary_ticker")
        or cap_info.get("primary_ticker"),
        issuer_listed_tickers=[
            str(item).upper()
            for item in cap_info.get("issuer_listed_tickers") or []
            if str(item).strip()
        ],
        security_role=cap_info.get("security_role") or cap_info.get("security_type"),
        is_secondary_class=(
            bool(cap_info["is_secondary_class"])
            if isinstance(cap_info.get("is_secondary_class"), bool)
            else None
        ),
        is_adr=(bool(cap_info["is_adr"]) if isinstance(cap_info.get("is_adr"), bool) else None),
        adr_ratio=(
            float(cap_info["adr_ratio"])
            if isinstance(cap_info.get("adr_ratio"), (int, float))
            and not isinstance(cap_info.get("adr_ratio"), bool)
            else None
        ),
        share_class_ratio=(
            float(cap_info["share_class_ratio"])
            if isinstance(cap_info.get("share_class_ratio"), (int, float))
            and not isinstance(cap_info.get("share_class_ratio"), bool)
            else None
        ),
        identity_source=cap_info.get("identity_source"),
        identity_source_url=cap_info.get("identity_source_url"),
        identity_as_of_date=cap_info.get("identity_as_of_date"),
        identity_confidence=cap_info.get("identity_confidence"),
        ratio_source_url=cap_info.get("ratio_source_url"),
        ratio_source_accession=cap_info.get("ratio_source_accession"),
        ratio_security_symbol=cap_info.get("ratio_security_symbol"),
        current_price=_optional_float(packet.current_price),
        current_price_unit=(
            PRICE_UNIT_USD_PER_SHARE
            if valuation_price is not None and str(selected_price_currency or "").upper() == "USD"
            else None
        ),
        current_price_as_of_date=(current_price_as_of_date),
        current_price_currency=(selected_price_currency),
        current_price_source=(selected_price_source),
        current_price_source_url=(selected_price_source_url),
        current_price_confidence=(
            cap_info.get("current_price_confidence")
            if valuation_price_from_current
            else cap_info.get("price_confidence")
            if valuation_price_from_cap
            else None
        ),
        quote_snapshot_id=current_quote_snapshot_id,
        price_basis=selected_price_basis,
        raw_price=selected_raw_price,
        shares_outstanding_mm=shares_mm,
        raw_shares_outstanding_mm=raw_shares_mm,
        raw_shares_source_value=raw_shares_source_value,
        raw_shares_source_unit=raw_shares_source_unit,
        shares_unit=cap_info.get("shares_unit"),
        shares_basis=cap_info.get("shares_basis"),
        shares_as_of_date=cap_info.get("shares_period_end") or cap_info.get("shares_as_of_date"),
        shares_filed_date=cap_info.get("shares_filed_date"),
        shares_source=cap_info.get("shares_source") or cap_info.get("cap_source_name"),
        shares_source_url=cap_info.get("shares_source_url") or cap_info.get("cap_source_url"),
        issuer_quote_ratio=issuer_quote_ratio,
        split_adjustment_factor=selected_split_factor,
        split_effective_date=selected_split_effective_date,
        split_lineage_proof=(
            dict(cap_info["split_lineage_proof"])
            if isinstance(cap_info.get("split_lineage_proof"), dict)
            else None
        ),
        market_cap_method=cap_info.get("cap_method") or cap_info.get("market_cap_method"),
        market_cap_derivation=cap_derivation,
        metric_traces=metric_traces,
        business_quality=business_quality,
        reinvestment=_reinvestment(packet),
        returns_on_capital=returns_on_capital,
        cash_conversion=cash_conversion,
        balance_sheet=_balance_sheet(packet),
        capital_allocation=capital_allocation,
        accounting_quality=_accounting_quality(packet),
        valuation=valuation,
        expected_return=_expected_return(packet, anchor_method, anchor),
        score_components=_score_components(packet, discount),
        blockers=blockers,
        confidence_caps=caps,
    )


def build_sector_company_financial_packets_from_signal_packets(
    packets: dict[str, TickerSignalPacket],
    *,
    sector: str | None = None,
    as_of_date: str | None = None,
    cap_classifications: dict[str, dict[str, Any]] | None = None,
    pipeline_version: str = "v1",
) -> list[SectorCompanyFinancialPacket]:
    """Build comparable financial packets from already assembled alpha packets."""

    classifications = cap_classifications if isinstance(cap_classifications, dict) else {}
    return [
        build_sector_company_financial_packet(
            packet,
            sector=sector,
            as_of_date=as_of_date,
            cap_classification=classifications.get(str(ticker).upper()),
            pipeline_version=pipeline_version,
        )
        for ticker, packet in sorted(packets.items(), key=lambda item: item[0])
    ]


def build_sector_company_financial_packets(
    tickers: list[str],
    *,
    sector: str | None = None,
    as_of_date: str | None = None,
    market_cap_focus: str = "small_cap",
    pipeline_version: str = "v1",
) -> list[SectorCompanyFinancialPacket]:
    """Assemble signal packets and map them into sector financial packets.

    ``sector``, ``as_of_date``, and ``market_cap_focus`` are accepted now so
    the eventual autonomous sector runtime can call this through a stable seam.
    The current deterministic builder does not need them beyond preserving the
    interface shape.
    """

    _ = market_cap_focus
    if str(pipeline_version or "v1").lower() == "v2":
        packets = assemble_sector_packets(
            tickers,
            filing_risk_use_llm=False,
            as_of_date=as_of_date,
            pipeline_version=pipeline_version,
        )
    else:
        packets = assemble_sector_packets(
            tickers,
            as_of_date=as_of_date,
        )
    return build_sector_company_financial_packets_from_signal_packets(
        packets,
        sector=sector,
        as_of_date=as_of_date,
        pipeline_version=pipeline_version,
    )


__all__ = [
    "build_sector_company_financial_packet",
    "build_sector_company_financial_packets",
    "build_sector_company_financial_packets_from_signal_packets",
]
