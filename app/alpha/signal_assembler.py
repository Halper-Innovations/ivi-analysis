"""Assembles all available signals for a ticker into a TickerSignalPacket."""

from __future__ import annotations

import json
import logging
import math
from contextlib import closing
from datetime import date
from pathlib import Path
from typing import Any, Sequence

from app.alpha.anomaly_detector import detect_anomalies
from app.alpha.filing_risk_scan import scan_filing_risks
from app.alpha.schemas import TickerSignalPacket
from app.autonomous.financial_integrity import (
    MARKET_CAP_UNIT_USD_MILLIONS,
    PRICE_UNIT_USD_PER_SHARE,
    SHARES_BASIS_UNADJUSTED,
    canonical_metric_trace,
    stable_quote_hash,
)
from app.alpha.solvency_scanner import assess_solvency
from app.config import AppConfig, get_config
from app.db import connect, get_db
from app.insurance.packet import build_insurance_packet
from app.util.financial_data_access import (
    companyfacts_rows,
    foreign_normalized_facts_gap_reason,
    issuer_companyfacts_rows,
    resolve_filing_issuer_scope,
)
from app.valuation.lineage import (
    latest_decision_eligible_valuation_row,
    latest_decision_eligible_valuation_rows,
)
from app.valuation.method_tension import analyze_method_tensions
from app.valuation.mos_conventions import graham_value_from_textbook_discount
from app.valuation.provenance import validate_v2_scorecard_provenance

logger = logging.getLogger(__name__)


def _db_context(
    db_path: str | Path | None,
    *,
    cfg: AppConfig | None = None,
):
    """Open an explicitly requested DB instead of the configured global DB."""

    return get_db(cfg) if db_path is None else closing(connect(db_path, cfg=cfg))


def _normalize_ticker_for_price(ticker: str) -> list[str]:
    """Return ticker variants to try for price lookup.

    Share class suffixes (BIO-B, HPE-PC) often fail on price providers
    but the base ticker works. Warrants (-WT, -WS) are kept as-is.
    """
    candidates = [ticker]
    if "-" in ticker and not ticker.endswith(("-WT", "-WS", "-UN")):
        base = ticker.split("-")[0]
        if len(base) >= 2:
            candidates.append(base)
    return candidates


def _fetch_live_price(ticker: str) -> float | None:
    """Attempt to fetch a live price via the configured price provider.

    Tries the exact ticker first, then falls back to the base ticker
    (without share class suffix) if the first attempt fails.
    Returns price as float, or None if unavailable/disabled.
    """
    try:
        from app.config import get_config

        cfg = get_config()
        if cfg.price_provider == "disabled":
            return None
        from app.valuation.price_provider import StooqPriceProvider

        provider = StooqPriceProvider()
        today = date.today().isoformat()
        for variant in _normalize_ticker_for_price(ticker):
            try:
                quote = provider.get_quote(variant, today)
                if quote and isinstance(quote.price, (int, float)) and quote.price > 0:
                    return float(quote.price)
            except Exception:
                continue
        return None
    except Exception as exc:
        logger.debug("signal_assembler: live price fetch failed for %s: %s", ticker, exc)
    return None


def _load_scorecard_record(
    ticker: str,
    *,
    as_of_date: str | None = None,
    issuer_cik: str | None = None,
    issuer_aliases: Sequence[str] = (),
    require_exact_issuer_binding: bool = False,
    db_path: str | Path | None = None,
    cfg: AppConfig | None = None,
) -> tuple[str | None, dict[str, Any]]:
    """Load the latest authorized scorecard visible at ``as_of_date``.

    The newest database candidate is selected before exact-source
    authorization. An invalid newest row suppresses the scorecard rather than
    resurrecting older evidence.
    """
    try:
        with _db_context(db_path, cfg=cfg) as conn:
            row = latest_decision_eligible_valuation_row(
                conn,
                ticker=ticker,
                method="scorecard",
                as_of_date=as_of_date,
                expected_issuer_cik=issuer_cik,
                expected_issuer_aliases=issuer_aliases,
                require_exact_issuer_binding=require_exact_issuer_binding,
            )
        if row and row["outputs_json"]:
            return str(row["as_of_date"]), json.loads(row["outputs_json"])
    except Exception as exc:
        logger.debug("signal_assembler: scorecard load failed for %s: %s", ticker, exc)
    return None, {}


def _load_scorecard(ticker: str) -> dict[str, Any]:
    """Load the latest scorecard valuation outputs for a ticker."""
    _, scorecard = _load_scorecard_record(ticker)
    return scorecard


def _positive_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if number > 0 else None


def _apply_financial_snapshot_contract(
    packet: TickerSignalPacket,
    snapshot: dict[str, Any],
    *,
    current_price_override: float | None,
) -> None:
    """Attach the explicit cap/quote/share lineage supplied by the caller."""

    current_price = _positive_float(current_price_override)
    if current_price is None:
        current_price = _positive_float(
            snapshot.get("current_price")
            if snapshot.get("current_price") is not None
            else snapshot.get("price")
            if snapshot.get("price") is not None
            else snapshot.get("price_used")
        )
    packet.current_price = current_price
    packet.current_price_currency = (
        str(
            snapshot.get("current_price_currency")
            or snapshot.get("price_currency")
            or snapshot.get("currency")
            or ""
        )
        .strip()
        .upper()
        or None
    )
    packet.current_price_unit = str(snapshot.get("current_price_unit") or "").strip() or (
        PRICE_UNIT_USD_PER_SHARE
        if current_price is not None and packet.current_price_currency == "USD"
        else None
    )
    packet.current_price_as_of_date = (
        snapshot.get("current_price_as_of_date")
        or snapshot.get("price_as_of_date")
        or snapshot.get("as_of_date")
    )
    packet.current_price_source = (
        snapshot.get("current_price_source")
        or snapshot.get("price_source")
        or snapshot.get("source")
        or snapshot.get("provider")
    )
    packet.current_price_source_url = (
        snapshot.get("current_price_source_url")
        or snapshot.get("price_source_url")
        or snapshot.get("source_url")
        or snapshot.get("url")
    )
    packet.price_basis = snapshot.get("price_basis")
    packet.raw_price = _positive_float(snapshot.get("raw_price"))
    if packet.raw_price is None and packet.price_basis == "UNADJUSTED":
        packet.raw_price = current_price
    packet.split_adjustment_factor = _positive_float(snapshot.get("split_adjustment_factor"))
    packet.split_effective_date = snapshot.get("split_effective_date")
    packet.split_lineage_proof = (
        dict(snapshot["split_lineage_proof"])
        if isinstance(snapshot.get("split_lineage_proof"), dict)
        else None
    )
    if current_price is not None:
        packet.quote_snapshot_id = stable_quote_hash(
            ticker=packet.ticker,
            price=current_price,
            as_of_date=packet.current_price_as_of_date,
            currency=packet.current_price_currency,
            source=packet.current_price_source,
            source_url=packet.current_price_source_url,
            price_basis=packet.price_basis,
            raw_price=packet.raw_price,
            split_adjustment_factor=packet.split_adjustment_factor,
            split_effective_date=packet.split_effective_date,
        )

    packet.market_cap_mm = _positive_float(snapshot.get("market_cap_mm"))
    packet.market_cap_unit = snapshot.get("market_cap_unit")
    packet.market_cap_source = snapshot.get("cap_source") or snapshot.get("market_cap_source")
    packet.market_cap_effective_as_of_date = snapshot.get(
        "cap_effective_as_of_date"
    ) or snapshot.get("market_cap_effective_as_of_date")
    packet.market_cap_source_kind = snapshot.get("cap_source_kind") or snapshot.get(
        "market_cap_source_kind"
    )
    packet.market_cap_source_name = snapshot.get("cap_source_name") or snapshot.get(
        "market_cap_source_name"
    )
    packet.market_cap_source_url = snapshot.get("cap_source_url") or snapshot.get(
        "market_cap_source_url"
    )
    packet.market_cap_confidence = snapshot.get("cap_confidence") or snapshot.get(
        "market_cap_confidence"
    )
    packet.market_cap_method = snapshot.get("cap_method") or snapshot.get("market_cap_method")
    derivation = snapshot.get("market_cap_derivation")
    packet.market_cap_derivation = dict(derivation) if isinstance(derivation, dict) else {}
    packet.shares_outstanding_mm = _positive_float(
        snapshot.get("shares_mm")
        if snapshot.get("shares_mm") is not None
        else snapshot.get("shares_outstanding_mm")
    )
    packet.raw_shares_outstanding_mm = _positive_float(snapshot.get("raw_shares_outstanding_mm"))
    packet.raw_shares_source_value = _positive_float(snapshot.get("raw_shares_source_value"))
    packet.raw_shares_source_unit = snapshot.get("raw_shares_source_unit")
    packet.shares_unit = snapshot.get("shares_unit")
    packet.shares_basis = snapshot.get("shares_basis")
    if packet.raw_shares_outstanding_mm is None and packet.shares_basis == SHARES_BASIS_UNADJUSTED:
        packet.raw_shares_outstanding_mm = packet.shares_outstanding_mm
    packet.shares_as_of_date = snapshot.get("shares_period_end") or snapshot.get(
        "shares_as_of_date"
    )
    packet.shares_filed_date = snapshot.get("shares_filed_date")
    packet.shares_source = snapshot.get("shares_source") or snapshot.get("cap_source_name")
    packet.shares_source_url = snapshot.get("shares_source_url") or snapshot.get("cap_source_url")
    packet.issuer_quote_ratio = _positive_float(snapshot.get("issuer_quote_ratio"))
    packet.issuer_cik = snapshot.get("issuer_cik")
    packet.issuer_primary_ticker = snapshot.get("issuer_primary_ticker")
    packet.issuer_listed_tickers = [
        str(item).strip().upper()
        for item in snapshot.get("issuer_listed_tickers") or []
        if str(item).strip()
    ]
    packet.security_role = snapshot.get("security_role")
    packet.is_secondary_class = snapshot.get("is_secondary_class")
    packet.is_adr = snapshot.get("is_adr")
    packet.adr_ratio = _positive_float(snapshot.get("adr_ratio"))
    packet.share_class_ratio = _positive_float(snapshot.get("share_class_ratio"))
    packet.identity_source = snapshot.get("identity_source")
    packet.identity_source_url = snapshot.get("identity_source_url")
    packet.identity_as_of_date = snapshot.get("identity_as_of_date")
    packet.identity_confidence = snapshot.get("identity_confidence")
    packet.ratio_source_url = snapshot.get("ratio_source_url")
    packet.ratio_source_accession = snapshot.get("ratio_source_accession")
    packet.ratio_security_symbol = snapshot.get("ratio_security_symbol")
    packet.cap_scope_status = snapshot.get("scope_status")
    packet.cap_scope_reason = snapshot.get("scope_reason")

    cap_stage_price = _positive_float(snapshot.get("price_used"))
    if cap_stage_price is None and snapshot.get("cap_stage_price") is not None:
        cap_stage_price = _positive_float(snapshot.get("cap_stage_price"))
    packet.cap_stage_price = cap_stage_price
    packet.cap_stage_price_as_of_date = snapshot.get("price_as_of_date") or snapshot.get(
        "cap_stage_price_as_of_date"
    )
    packet.cap_stage_price_currency = snapshot.get("price_currency") or snapshot.get(
        "cap_stage_price_currency"
    )
    packet.cap_stage_price_source = snapshot.get("price_source") or snapshot.get(
        "cap_stage_price_source"
    )
    packet.cap_stage_price_source_url = snapshot.get("price_source_url") or snapshot.get(
        "cap_stage_price_source_url"
    )
    if packet.cap_stage_price is not None:
        packet.cap_stage_quote_snapshot_id = stable_quote_hash(
            ticker=packet.ticker,
            price=packet.cap_stage_price,
            as_of_date=packet.cap_stage_price_as_of_date,
            currency=packet.cap_stage_price_currency,
            source=packet.cap_stage_price_source,
            source_url=packet.cap_stage_price_source_url,
            price_basis=packet.price_basis,
            raw_price=packet.raw_price,
            split_adjustment_factor=packet.split_adjustment_factor,
            split_effective_date=packet.split_effective_date,
        )

    cap = packet.market_cap_mm
    shares = packet.shares_outstanding_mm
    ratio = packet.issuer_quote_ratio
    if cap is not None and current_price is not None and shares is not None:
        if ratio is None:
            return
        recomputed_cap = current_price * shares / ratio
        packet.metric_traces["market_cap_mm"] = canonical_metric_trace(
            metric="market_cap_mm",
            formula="current_price * shares_outstanding_mm / issuer_quote_ratio",
            inputs={
                "current_price": current_price,
                "shares_outstanding_mm": shares,
                "issuer_quote_ratio": ratio,
            },
            output=cap,
            recomputed_output=recomputed_cap,
            output_unit=MARKET_CAP_UNIT_USD_MILLIONS,
            quote_snapshot_id=packet.quote_snapshot_id,
            input_provenance={
                "current_price": {
                    "value": current_price,
                    "unit": packet.current_price_unit,
                    "source": packet.current_price_source,
                    "period_end": packet.current_price_as_of_date,
                    "filed_date": packet.current_price_as_of_date,
                    "source_reference": packet.current_price_source_url,
                    "basis": packet.price_basis,
                    "raw_value": packet.raw_price,
                    "split_adjustment_factor": packet.split_adjustment_factor,
                    "split_effective_date": packet.split_effective_date,
                },
                "shares_outstanding_mm": {
                    "value": shares,
                    "raw_value": packet.raw_shares_outstanding_mm,
                    "raw_source_value": packet.raw_shares_source_value,
                    "raw_source_unit": packet.raw_shares_source_unit,
                    "normalized_value": shares,
                    "normalized_unit": packet.shares_unit,
                    "unit": packet.shares_unit,
                    "source": packet.shares_source,
                    "period_end": packet.shares_as_of_date,
                    "filed_date": packet.shares_filed_date,
                    "source_reference": packet.shares_source_url,
                    "basis": packet.shares_basis,
                    "split_adjustment_factor": packet.split_adjustment_factor,
                    "split_effective_date": packet.split_effective_date,
                },
                "issuer_quote_ratio": {
                    "value": ratio,
                    "unit": "ratio",
                    "source": (snapshot.get("identity_source") or "ISSUER_SECURITY_IDENTITY"),
                    "period_end": (
                        snapshot.get("identity_as_of_date") or packet.current_price_as_of_date
                    ),
                    "filed_date": (
                        snapshot.get("identity_as_of_date") or packet.current_price_as_of_date
                    ),
                    "source_reference": (
                        snapshot.get("ratio_source_url")
                        or snapshot.get("identity_source_url")
                        or f"ISSUER_SECURITY_IDENTITY:{packet.ticker}"
                    ),
                },
            },
            rel_tol=1e-6,
            abs_tol=1e-6,
        )


def _apply_insurance_packet(packet: TickerSignalPacket, insurance_packet: dict[str, Any]) -> None:
    """Attach insurance routing/valuation and suppress invalid generic anchors."""
    if not isinstance(insurance_packet, dict) or not insurance_packet:
        return
    routing = (
        insurance_packet.get("routing") if isinstance(insurance_packet.get("routing"), dict) else {}
    )
    valuation = (
        insurance_packet.get("valuation")
        if isinstance(insurance_packet.get("valuation"), dict)
        else {}
    )
    if insurance_packet.get("model_status") == "NOT_APPLICABLE" and routing.get(
        "security_type"
    ) in (None, "common"):
        return
    if (
        insurance_packet.get("generic_valuation_valid") is not False
        and insurance_packet.get("model_status") != "OK"
    ):
        return

    packet.security_type = routing.get("security_type")
    packet.issuer_type = routing.get("issuer_type")
    packet.insurance_subtype = routing.get("insurance_subtype")
    packet.insurance_packet = insurance_packet
    packet.insurance_valuation = valuation
    packet.model_status = insurance_packet.get("model_status")
    packet.model_blockers = [str(item) for item in insurance_packet.get("model_blockers") or []]
    packet.model_fit_warnings = [
        str(item) for item in insurance_packet.get("model_fit_warnings") or []
    ]

    if insurance_packet.get("generic_valuation_valid") is False:
        packet.dcf_value = None
        packet.epv_value = None
        packet.graham_value = None
        packet.ncav_value = None
        if "GENERIC_DCF_EPV_SUPPRESSED" not in packet.valuation_headwinds:
            packet.valuation_headwinds.append("GENERIC_DCF_EPV_SUPPRESSED")

    anchor = valuation.get("valuation_anchor")
    if valuation.get("model_status") == "OK" and isinstance(anchor, (int, float)):
        packet.insurance_value = float(anchor)
        packet.insurance_method = str(valuation.get("method") or "insurance")


def _load_quarterly_revenue_trend(
    ticker: str,
    *,
    as_of_date: str | None = None,
    require_filed_asof: bool = False,
    issuer_cik: str | None = None,
    aliases: Sequence[str] = (),
    db_path: str | Path | None = None,
    cfg: AppConfig | None = None,
) -> tuple[float | None, str | None, str | None]:
    """Return (latest_quarterly_revenue, period_label, trend)."""
    try:
        with _db_context(db_path, cfg=cfg) as conn:
            kwargs = {
                "exclude_period_types": ("FY",),
                "line_items": ("revenue",),
                "as_of_date": as_of_date,
                "value_not_null": True,
                "require_filed_asof": require_filed_asof,
                "order_by": "fiscal_year DESC, period_type DESC",
                "limit": 4,
            }
            if issuer_cik is not None or aliases:
                _scope, rows = issuer_companyfacts_rows(
                    conn,
                    ticker,
                    columns=("fiscal_year", "period_type", "value"),
                    issuer_cik=issuer_cik,
                    aliases=aliases,
                    **kwargs,
                )
            else:
                rows = companyfacts_rows(
                    conn,
                    ticker,
                    columns=("fiscal_year", "period_type", "value"),
                    **kwargs,
                )
        if not rows:
            return None, None, "UNKNOWN"
        latest = rows[0]
        latest_val = float(latest["value"])
        period_label = f"FY{latest['fiscal_year']}{latest['period_type']}"
        if len(rows) >= 2:
            prev_val = float(rows[1]["value"])
            if prev_val > 0:
                qoq_change = (latest_val - prev_val) / prev_val
                if qoq_change > 0.05:
                    trend = "ACCELERATING"
                elif qoq_change < -0.05:
                    trend = "DECELERATING"
                else:
                    trend = "STABLE"
            else:
                trend = "UNKNOWN"
        else:
            trend = "UNKNOWN"
        return latest_val, period_label, trend
    except Exception:
        return None, None, "UNKNOWN"


def assemble_signal_packet(
    ticker: str,
    *,
    filing_risk_use_llm: bool = True,
    as_of_date: str | None = None,
    pipeline_version: str = "v1",
    current_price_override: float | None = None,
    price_snapshot: dict[str, Any] | None = None,
    issuer_cik: str | None = None,
    issuer_aliases: tuple[str, ...] = (),
    db_path: str | Path | None = None,
    cfg: AppConfig | None = None,
    allowed_filing_roots: tuple[str | Path, ...] = (),
) -> TickerSignalPacket:
    """Build a complete signal packet for one ticker from all available data sources."""
    upper = ticker.upper()
    packet = TickerSignalPacket(ticker=upper)
    normalized_pipeline = str(pipeline_version or "v1").strip().lower()
    if normalized_pipeline not in {"v1", "v2"}:
        raise ValueError("pipeline_version must be v1 or v2")
    is_v2 = normalized_pipeline == "v2"
    # V2 filing-risk evidence is deliberately deterministic. The legacy LLM
    # classifier calls a provider outside the autonomous usage ledger and can
    # therefore neither be authorized nor reconciled by the whole-run gate.
    if is_v2:
        filing_risk_use_llm = False
    effective_as_of = str(as_of_date or "").strip()[:10] or None
    if is_v2 and effective_as_of is None:
        raise ValueError("v2 signal assembly requires as_of_date")
    # Autonomous sector callers provide an explicit run boundary.  Filing
    # classification must remain deterministic until the financial-integrity
    # gate has authorized a paid reasoning call.
    if effective_as_of is not None:
        filing_risk_use_llm = False
    filing_cfg = cfg or get_config()
    if db_path is not None:
        filing_cfg = filing_cfg.model_copy(update={"db_path": Path(db_path)})
    with _db_context(db_path, cfg=cfg) as conn:
        issuer_scope = resolve_filing_issuer_scope(
            conn,
            upper,
            issuer_cik=issuer_cik,
            aliases=issuer_aliases,
        )
    issuer_cik = issuer_scope.issuer_cik
    issuer_aliases = issuer_scope.aliases

    # Scorecard / valuation
    # Preserve the historical monkeypatch seam on _load_scorecard while still
    # keeping the real DB as-of date available for insurance packet routing.
    valuation_provenance_blockers: list[str] = []
    if is_v2:
        with _db_context(db_path, cfg=cfg) as conn:
            scorecard_state = validate_v2_scorecard_provenance(
                conn,
                upper,
                as_of_date=str(effective_as_of),
                issuer_cik=issuer_cik,
                issuer_aliases=issuer_aliases,
                price_snapshot=price_snapshot,
            )
        scorecard_as_of_date = scorecard_state.get("validated_asof")
        scorecard_record = scorecard_state.get("outputs") or {}
        scorecard = scorecard_record
        valuation_provenance_blockers = [
            str(item) for item in scorecard_state.get("mismatch_reasons") or []
        ]
        packet.valuation_provenance_status = (
            "VALIDATED" if not valuation_provenance_blockers else "INVALID"
        )
        packet.valuation_provenance_blockers = valuation_provenance_blockers
    else:
        if effective_as_of is None:
            scorecard = _load_scorecard(upper)
            scorecard_as_of_date, scorecard_record = _load_scorecard_record(upper)
        else:
            scorecard_as_of_date, scorecard_record = _load_scorecard_record(
                upper,
                as_of_date=effective_as_of,
                issuer_cik=issuer_cik,
                issuer_aliases=issuer_aliases,
                require_exact_issuer_binding=True,
                db_path=db_path,
                cfg=cfg,
            )
            scorecard = scorecard_record
    scorecard_from_record = bool(scorecard) and scorecard == scorecard_record
    if scorecard:
        # Intrinsic values live in pricing_zone_detail (scorecard method)
        pzd = scorecard.get("pricing_zone_detail") or {}
        packet.pricing_zone = (
            str(scorecard.get("pricing_zone")) if scorecard.get("pricing_zone") else None
        )
        # ONE deployable population with the backtest (review ANCHOR-2 /
        # GZ-2 / zone-allowlist): anchors come only from the three surviving
        # zones the backtest measures. VALUATION_ANOMALY (negative EPV/DCF, the
        # original leak), INSUFFICIENT_DATA (pzd still carries a positive
        # dcf_base or graham/ncav per-share), blocked, and zoneless pre-zone
        # scorecards contribute NO anchor methods — raw_valuation keeps the
        # full scorecard for LLM context.
        zone_suppressed = packet.pricing_zone not in (
            "MARGIN_OF_SAFETY",
            "GROWTH_DEPENDENT",
            "SPECULATIVE_PREMIUM",
        )
        if zone_suppressed:
            packet.dcf_value = None
            packet.epv_value = None
        else:
            packet.dcf_value = pzd.get("dcf_base")
            packet.epv_value = pzd.get("epv_adjusted")
        # Per-method values persisted by the writer (post-audit) — preferred
        # over back-computing from discounts; symmetric with the backtest's
        # pzd reads (review issue 2b/2c).
        if not zone_suppressed and isinstance(pzd.get("ncav_value_per_share"), (int, float)):
            packet.ncav_value = float(pzd["ncav_value_per_share"])
        price = pzd.get("current_price")
        packet.current_price = price if isinstance(price, (int, float)) and price > 0 else None
        # Graham: prefer the writer-persisted per-share value (post-audit);
        # fall back to inverting the TEXTBOOK discount d=(iv-price)/iv =>
        # iv = price/(1-d) for older scorecards
        # (audit: graham-discount-inversion — price/(1+d) sign-inverted it).
        discounts = scorecard.get("discounts") or {}
        if zone_suppressed:
            packet.graham_value = None
        elif isinstance(pzd.get("graham_value_per_share"), (int, float)):
            packet.graham_value = float(pzd["graham_value_per_share"])
        elif isinstance(discounts.get("graham"), (int, float)) and packet.current_price:
            packet.graham_value = graham_value_from_textbook_discount(
                packet.current_price, discounts["graham"]
            )
        packet.margin_of_safety_verdict = scorecard.get("legacy_signal")  # OVERVALUED / UNDERVALUED
        packet.raw_valuation = scorecard

        # Quality context — field is "quality_context" not "quality_ctx"
        qctx = scorecard.get("quality_context") or {}
        packet.gate_verdict = qctx.get("gate_action") or pzd.get("gate_action")
        packet.confidence_class = qctx.get("confidence_class") or qctx.get("earnings_quality")
        # Moat lives in moat_strength
        moat = scorecard.get("moat_strength") or {}
        packet.moat_score = moat.get("moat_score")
        packet.moat_classification = moat.get("moat_class")
        # Downside
        downside = scorecard.get("downside_scenario") or {}
        packet.downside_risk_class = downside.get("downside_risk_class")
        packet.valuation_headwinds = (
            qctx.get("valuation_headwinds") or scorecard.get("valuation_headwinds") or []
        )
        packet.valuation_supports = (
            qctx.get("valuation_supports") or scorecard.get("valuation_supports") or []
        )
        packet.raw_quality_ctx = qctx

        # Signal context
        signal_ctx = scorecard.get("signal_context") or scorecard.get("signal_context_val") or {}
        if isinstance(signal_ctx, dict):
            packet.filing_diff_changes = signal_ctx.get("filing_diff_changes") or []
            packet.high_materiality_changes = sum(
                1
                for c in packet.filing_diff_changes
                if isinstance(c, dict) and c.get("materiality") == "HIGH"
            )
            packet.pattern_hits = signal_ctx.get("pattern_hits") or []
            packet.patterns_with_signal = signal_ctx.get("patterns_with_signal") or []

        # Peer context
        peer_ctx = scorecard.get("peer_context") or {}
        if isinstance(peer_ctx, dict):
            packet.peer_position = peer_ctx.get("position")
            packet.roic_vs_median = peer_ctx.get("roic_vs_median")
            packet.op_margin_vs_median = peer_ctx.get("operating_margin_vs_median")
            packet.revenue_growth_vs_median = peer_ctx.get("revenue_growth_vs_median")

    # An explicit override is the immutable cap-stage quote for either active
    # pipeline. It always wins over a scorecard price and suppresses a later
    # live-price substitution.
    if current_price_override is not None or is_v2:
        packet.current_price = (
            float(current_price_override)
            if isinstance(current_price_override, (int, float))
            and not isinstance(current_price_override, bool)
            and current_price_override > 0
            else None
        )
    # Live price fallback — legacy callers retain the existing behavior.
    elif effective_as_of is None and packet.current_price is None:
        live = _fetch_live_price(upper)
        if live is not None:
            packet.current_price = live
    if isinstance(price_snapshot, dict) and price_snapshot:
        _apply_financial_snapshot_contract(
            packet,
            price_snapshot,
            current_price_override=current_price_override,
        )

    # Insurance and non-common securities need a different valuation contract.
    if scorecard_from_record or is_v2:
        try:
            if is_v2:
                insurance_packet = build_insurance_packet(
                    upper,
                    as_of_date=effective_as_of,
                    scorecard=scorecard,
                    persist=False,
                    pipeline_version="v2",
                    current_price_override=packet.current_price,
                    issuer_cik=issuer_cik,
                    aliases=issuer_aliases,
                    db_path=db_path,
                    cfg=cfg,
                )
            elif effective_as_of is not None:
                insurance_packet = build_insurance_packet(
                    upper,
                    as_of_date=effective_as_of,
                    scorecard=scorecard,
                    persist=False,
                    pipeline_version="v1",
                    current_price_override=packet.current_price,
                    issuer_cik=issuer_cik,
                    aliases=issuer_aliases,
                    db_path=db_path,
                    cfg=cfg,
                )
            else:
                insurance_packet = build_insurance_packet(
                    upper,
                    as_of_date=scorecard_as_of_date or date.today().isoformat(),
                    scorecard=scorecard,
                    persist=False,
                    pipeline_version="v1",
                    current_price_override=packet.current_price,
                    issuer_cik=issuer_cik,
                    aliases=issuer_aliases,
                    db_path=db_path,
                    cfg=cfg,
                )
            _apply_insurance_packet(packet, insurance_packet)
        except Exception as exc:
            if is_v2 or effective_as_of is not None:
                raise
            logger.debug("signal_assembler: insurance packet build failed for %s: %s", upper, exc)

    # Quarterly freshness
    if is_v2:
        q_rev, q_period, q_trend = _load_quarterly_revenue_trend(
            upper,
            as_of_date=effective_as_of,
            require_filed_asof=True,
            issuer_cik=issuer_cik,
            aliases=issuer_aliases,
            db_path=db_path,
            cfg=cfg,
        )
    elif effective_as_of is not None:
        q_rev, q_period, q_trend = _load_quarterly_revenue_trend(
            upper,
            as_of_date=effective_as_of,
            require_filed_asof=True,
            issuer_cik=issuer_cik,
            aliases=issuer_aliases,
            db_path=db_path,
            cfg=cfg,
        )
    else:
        q_rev, q_period, q_trend = _load_quarterly_revenue_trend(upper)
    packet.latest_quarterly_revenue = q_rev
    packet.latest_quarterly_period = q_period
    packet.quarterly_revenue_trend = q_trend

    # Filing risk scan. Preserve the historical one-argument monkeypatch seam
    # for default callers while letting provider-free cache targeting opt out
    # of LLM classification explicitly.
    def _scoped_filing_risk_scan(**kwargs: Any) -> dict[str, Any]:
        if db_path is None and cfg is None:
            return scan_filing_risks(upper, **kwargs)
        with _db_context(db_path, cfg=cfg) as connection:
            return scan_filing_risks(
                upper,
                connection=connection,
                cfg=filing_cfg,
                allowed_filing_roots=allowed_filing_roots,
                **kwargs,
            )

    if is_v2:
        risk_result = _scoped_filing_risk_scan(
            use_llm=filing_risk_use_llm,
            as_of_date=effective_as_of,
            issuer_cik=issuer_cik,
            aliases=issuer_aliases,
            issuer_aware=True,
            allow_network_materialization=False,
        )
    elif filing_risk_use_llm:
        risk_result = _scoped_filing_risk_scan()
    elif effective_as_of is not None:
        risk_result = _scoped_filing_risk_scan(
            use_llm=False,
            as_of_date=effective_as_of,
            issuer_cik=issuer_cik,
            aliases=issuer_aliases,
            issuer_aware=True,
            allow_network_materialization=False,
        )
    else:
        risk_result = _scoped_filing_risk_scan(use_llm=False)
    foreign_gap_reason: str | None = None
    if is_v2 and str(risk_result.get("source_form_type") or "").upper() in {
        "20-F",
        "20-F/A",
        "40-F",
        "40-F/A",
    }:
        try:
            with _db_context(db_path, cfg=cfg) as conn:
                foreign_gap_reason = foreign_normalized_facts_gap_reason(
                    conn,
                    upper,
                    issuer_cik=issuer_cik or risk_result.get("source_issuer_cik"),
                    form_type=risk_result.get("source_form_type"),
                    as_of_date=(
                        risk_result.get("analysis_as_of_date")
                        or scorecard_as_of_date
                        or date.today().isoformat()
                    ),
                    aliases=issuer_aliases,
                    require_filed_asof=True,
                )
        except Exception as exc:
            logger.debug(
                "signal_assembler: foreign facts gap classification failed for %s: %s",
                upper,
                exc,
            )
    packet.filing_risk_status = risk_result.get("status")
    packet.filing_risk_signals = {
        k: v
        for k, v in risk_result.items()
        if k
        in (
            "competitive_disruption",
            "secular_decline",
            "regulatory_legal",
            "customer_concentration",
            "summary",
        )
    }
    if foreign_gap_reason is not None:
        packet.filing_risk_signals["foreign_facts_gap_reason"] = foreign_gap_reason
    packet.filing_risk_metadata = {
        k: v
        for k, v in risk_result.items()
        if k
        in (
            "evidence_status",
            "warnings",
            "source_accession",
            "source_form_type",
            "source_filing_date",
            "source_filing_age_days",
            "risk_text_chars",
        )
    }

    # Cheap research: anomaly detection + solvency (deterministic, no LLM)
    if is_v2:
        anomalies = detect_anomalies(
            upper,
            as_of_date=effective_as_of,
            require_filed_asof=True,
            issuer_cik=issuer_cik,
            aliases=issuer_aliases,
            db_path=db_path,
        )
        solvency = assess_solvency(
            upper,
            as_of_date=effective_as_of,
            require_filed_asof=True,
            issuer_cik=issuer_cik,
            aliases=issuer_aliases,
            db_path=db_path,
        )
    else:
        if effective_as_of is None:
            anomalies = detect_anomalies(upper)
            solvency = assess_solvency(upper)
        else:
            anomalies = detect_anomalies(
                upper,
                as_of_date=effective_as_of,
                require_filed_asof=True,
                issuer_cik=issuer_cik,
                aliases=issuer_aliases,
                db_path=db_path,
            )
            solvency = assess_solvency(
                upper,
                as_of_date=effective_as_of,
                require_filed_asof=True,
                issuer_cik=issuer_cik,
                aliases=issuer_aliases,
                db_path=db_path,
            )

    packet.anomaly_count = len(anomalies)
    packet.solvency_risk = solvency.solvency_risk
    packet.research_status = "OK" if anomalies or solvency.signals else "NO_ANOMALIES"
    packet.research_report = {
        "anomalies": [
            {
                "type": a.anomaly_type,
                "severity": a.severity,
                "description": a.description,
                "question": a.question,
            }
            for a in anomalies
        ],
        "investigations": [],  # populated later by comparator for finalists only
        "solvency": {
            "risk": solvency.solvency_risk,
            "signals": solvency.signals,
            "details": solvency.details,
            "negative_equity": solvency.negative_equity,
            "current_ratio": solvency.current_ratio,
            "cash_runway_quarters": solvency.cash_runway_quarters,
            "going_concern_language": solvency.going_concern_language,
            "going_concern_assertions": [
                assertion.to_dict() for assertion in solvency.going_concern_assertions
            ],
            "no_assurance_financing": solvency.no_assurance_financing,
            "valuation_allowance_full": solvency.valuation_allowance_full,
            "debt_due_within_12mo": solvency.debt_due_within_12mo,
        },
    }

    # Method tension analysis. V2 may use only values carried by the validated
    # scorecard above; loading separate unbound method rows would reintroduce
    # stale/wrong-issuer valuation evidence into ranking and prompts.
    if is_v2 and valuation_provenance_blockers:
        return packet

    try:
        if is_v2:
            dcf_val = packet.dcf_value
            epv_val = packet.epv_value
            graham_val = packet.graham_value
            ncav_val = packet.ncav_value
        else:
            with _db_context(db_path, cfg=cfg) as conn:
                method_values = _legacy_method_values(
                    conn,
                    ticker=upper,
                    as_of_date=effective_as_of,
                    issuer_cik=issuer_cik,
                    issuer_aliases=issuer_aliases,
                )
                dcf_val = method_values.get("dcf") or method_values.get("dcf_adjusted")
                epv_val = method_values.get("epv") or method_values.get("epv_adjusted")
                graham_val = method_values.get("graham")
                ncav_val = method_values.get("ncav")

        qctx = packet.raw_quality_ctx or {}
        tensions = analyze_method_tensions(
            dcf_value=dcf_val,
            epv_value=epv_val,
            graham_value=graham_val,
            ncav_value=ncav_val,
            current_price=packet.current_price,
            revenue_cagr_5y=qctx.get("revenue_cagr_5y"),
            wacc=0.10,
            terminal_growth=0.015,
        )
        packet.method_tension_type = tensions.get("tension_type")
        packet.growth_dependency_ratio = tensions.get("growth_value_pct")
        packet.methods_agree = tensions.get("methods_agree")
        packet.consensus_direction = tensions.get("consensus_direction")
        ir = tensions.get("intrinsic_range", {})
        packet.intrinsic_range_low = ir.get("low")
        packet.intrinsic_range_high = ir.get("high")
    except Exception as exc:
        logger.debug("signal_assembler: tension analysis failed for %s: %s", upper, exc)

    return packet


_LEGACY_METHOD_VALUE_KEYS = {
    "dcf": "base",
    "dcf_adjusted": "base",
    "epv": "value_per_share",
    "epv_adjusted": "value_per_share",
    "graham": "value_per_share",
    "ncav": "value_per_share",
}


def _legacy_method_values(
    conn: Any,
    *,
    ticker: str,
    as_of_date: str | None,
    issuer_cik: str | None,
    issuer_aliases: Sequence[str],
) -> dict[str, float]:
    """Per-share method values for the legacy (v1) tension analysis, from ONE run.

    Each method used to be read on its own as "newest row at or before the as-of date", so a
    tension could compare a DCF from one run against an EPV from another (or a stale DCF
    against a fresh EPV). The newest authorized method row now names the run: only rows with
    the same as-of date and source run count, and any method missing from that run is simply
    absent. Only a status of OK with a positive finite value is admitted — an EPV_NEGATIVE
    row's negative "value" is not an intrinsic value and stretched the intrinsic range and
    the growth-dependency ratio (its status stays on the valuation row).
    """
    rows = latest_decision_eligible_valuation_rows(
        conn,
        ticker=ticker,
        methods=tuple(_LEGACY_METHOD_VALUE_KEYS),
        as_of_date=as_of_date,
        expected_issuer_cik=issuer_cik,
        expected_issuer_aliases=issuer_aliases,
        require_exact_issuer_binding=as_of_date is not None,
    )
    if not rows:
        return {}
    newest = max(
        rows,
        key=lambda row: (
            str(row["as_of_date"] or ""),
            str(row["created_at"] or ""),
            int(row["id"] or 0),
        ),
    )
    run = (str(newest["as_of_date"] or ""), str(newest["source_run_id"] or ""))
    values: dict[str, float] = {}
    for row in rows:
        if (str(row["as_of_date"] or ""), str(row["source_run_id"] or "")) != run:
            continue
        method = str(row["method"])
        try:
            outputs = json.loads(row["outputs_json"])
        except (TypeError, ValueError):
            continue
        if not isinstance(outputs, dict) or outputs.get("status") != "OK":
            continue
        value = outputs.get(_LEGACY_METHOD_VALUE_KEYS[method])
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if math.isfinite(float(value)) and float(value) > 0:
                values[method] = float(value)
    return values


def assemble_sector_packets(
    tickers: list[str],
    *,
    filing_risk_use_llm: bool = True,
    as_of_date: str | None = None,
    pipeline_version: str = "v1",
    current_prices: dict[str, float | None] | None = None,
    issuer_contexts: dict[str, dict[str, Any]] | None = None,
    db_path: str | Path | None = None,
    cfg: AppConfig | None = None,
    allowed_filing_roots: tuple[str | Path, ...] = (),
) -> dict[str, TickerSignalPacket]:
    """Assemble signal packets for all tickers in a sector."""
    normalized_pipeline = str(pipeline_version or "v1").strip().lower()
    if normalized_pipeline not in {"v1", "v2"}:
        raise ValueError("pipeline_version must be v1 or v2")
    if normalized_pipeline == "v2" or as_of_date is not None:
        filing_risk_use_llm = False
    prices = current_prices if isinstance(current_prices, dict) else {}
    identities = issuer_contexts if isinstance(issuer_contexts, dict) else {}
    packets: dict[str, TickerSignalPacket] = {}
    for ticker in tickers:
        upper = ticker.upper()
        identity = identities.get(upper) if isinstance(identities.get(upper), dict) else {}
        aliases = tuple(
            dict.fromkeys(
                str(item).strip().upper()
                for item in (
                    upper,
                    identity.get("issuer_primary_ticker"),
                    identity.get("primary_ticker"),
                    *(identity.get("issuer_listed_tickers") or ()),
                    *(identity.get("issuer_aliases") or ()),
                )
                if str(item or "").strip()
            )
        )
        legacy_price_snapshot = {
            "price": prices.get(upper),
            "as_of_date": identity.get("current_price_as_of_date")
            or identity.get("price_as_of_date"),
            "currency": identity.get("current_price_currency") or identity.get("price_currency"),
            "source": identity.get("current_price_source") or identity.get("price_source"),
            "url": identity.get("current_price_source_url") or identity.get("price_source_url"),
            "confidence": identity.get("current_price_confidence")
            or identity.get("price_confidence"),
        }
        integrity_keys = {
            "market_cap_mm",
            "market_cap_unit",
            "quote_snapshot_id",
            "price_basis",
            "raw_price",
            "split_adjustment_factor",
            "shares_mm",
            "raw_shares_outstanding_mm",
            "raw_shares_source_value",
            "raw_shares_source_unit",
            "shares_unit",
            "shares_basis",
            "cap_method",
        }
        has_integrity_context = any(key in identity for key in integrity_keys)
        has_legacy_quote_context = prices.get(upper) is not None or any(
            value not in (None, "")
            for key, value in legacy_price_snapshot.items()
            if key != "price"
        )
        canonical_price_snapshot = (
            {**identity, **legacy_price_snapshot}
            if has_integrity_context
            else legacy_price_snapshot
            if has_legacy_quote_context
            else None
        )
        assembly_kwargs: dict[str, Any] = {
            "filing_risk_use_llm": filing_risk_use_llm,
            "as_of_date": as_of_date,
            "pipeline_version": pipeline_version,
            "current_price_override": prices.get(upper),
            "price_snapshot": canonical_price_snapshot,
            "issuer_cik": identity.get("issuer_cik") or identity.get("cik"),
            "issuer_aliases": aliases,
            "db_path": db_path,
            "allowed_filing_roots": allowed_filing_roots,
        }
        if cfg is not None:
            assembly_kwargs["cfg"] = cfg
        packets[upper] = assemble_signal_packet(upper, **assembly_kwargs)
    return packets
