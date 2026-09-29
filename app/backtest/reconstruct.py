"""Reconstruct the deterministic buy signal as-of a past date (approach gamma).

Drives the production valuation (ensure_valuation, now as-of-clean per the
_load_facts fix) with the historical price injected as price_override, then
consumes the exact scorecard + reverse-DCF records returned by that writer
invocation. No LLM, no grade.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from app.backtest.asof import effective_asof_cutoff
from app.backtest.universe import cap_category_asof
from app.valuation.anchor_policy import is_decline_class, select_anchor
from app.valuation.lineage import valuation_source_record
from app.valuation.valuation_writer import ensure_valuation
from app.watchlist.anchor_sanity import evaluate_anchor


@dataclass
class ReconstructedSignal:
    ticker: str
    as_of_date: str
    anchor: float
    price: float  # adjusted close at T — the ledger entry/return leg
    buy_price_target: float  # round(anchor * _NEUTRAL_DISCOUNT, 4) — single source of truth
    mos: float  # (anchor - classification_price) / anchor
    deploy_ready: bool  # classification_price <= buy_price_target
    expectations_gap_bucket: str
    cap_category: str | None
    # Scorecard pricing zone at the as-of date — lets the diagnostic stratify
    # deploy_ready by zone (MARGIN_OF_SAFETY vs GROWTH_DEPENDENT etc.).
    pricing_zone: str | None = None
    # Split-basis re-basing: deploy classification compares the as-of
    # per-share anchor against the RAW close at T (what live saw), while
    # `price` stays adjusted so entry/exit return legs share one series.
    # Legacy cached snapshots lack the raw close — basis records the fallback.
    classification_price: float | None = None
    price_basis: str = "raw_close"


@dataclass
class ReconstructResult:
    ticker: str
    as_of_date: str
    signal: ReconstructedSignal | None
    skip_reason: str | None = None


_NEUTRAL_DISCOUNT = 0.75  # WATCHLIST_ONLY base; deploy_ready = price <= anchor * 0.75

# Per-process memo for the sector-model routing exclusion: issuer type is
# effectively static per ticker, and route_security reads cached profiles /
# filing text (too costly per (ticker, date)).
_SECTOR_MODEL_ROUTING_CACHE: dict[str, bool] = {}


def _live_sector_model_routed(ticker: str, as_of_date: str) -> bool:
    """True when the live surface routes this name away from generic dcf/epv
    anchors — non-common security or insurance underwriter (the
    generic_invalid predicate in app.insurance.packet). The backtest cannot
    reconstruct the insurance/preferred sector model point-in-time, so these
    rows are EXCLUDED from measurement with a granular skip reason rather
    than silently measured under a rule production never runs (review
    ANCHOR-5). UNKNOWN routing keeps the row: live keeps generic anchors for
    unknown security types."""
    upper = ticker.upper()
    if upper in _SECTOR_MODEL_ROUTING_CACHE:
        return _SECTOR_MODEL_ROUTING_CACHE[upper]
    try:
        import app.insurance.routing as insurance_routing

        routing = insurance_routing.route_security(upper, as_of_date=as_of_date)
        routed = (
            routing.security_type
            not in (insurance_routing.SECURITY_COMMON, insurance_routing.SECURITY_UNKNOWN)
            or routing.issuer_type == insurance_routing.ISSUER_INSURANCE_UNDERWRITER
        )
    except Exception:  # noqa: BLE001 - routing failure must not abort the run
        routed = False
    _SECTOR_MODEL_ROUTING_CACHE[upper] = routed
    return routed


def _writer_output_payloads(
    records: Any,
    *,
    ticker: str,
    as_of_date: str,
    run_id: str,
) -> dict[str, dict[str, Any]]:
    """Validate and unpack the exact records returned by one writer invocation."""

    if not isinstance(records, list) or not records:
        raise RuntimeError("valuation writer returned no source records")

    expected_ticker = ticker.upper()
    payloads: dict[str, dict[str, Any]] = {}
    for raw_record in records:
        if (
            not isinstance(raw_record, Mapping)
            or set(raw_record) != {"schema_version", "row"}
            or not isinstance(raw_record.get("row"), Mapping)
        ):
            raise RuntimeError("valuation writer returned a malformed source record")

        row = raw_record["row"]
        canonical = valuation_source_record(row)
        if canonical is None or canonical != dict(raw_record):
            raise RuntimeError("valuation writer returned a noncanonical source record")
        if row.get("ticker") != expected_ticker:
            raise RuntimeError("valuation writer record ticker mismatch")
        if row.get("as_of_date") != as_of_date:
            raise RuntimeError("valuation writer record as-of mismatch")
        if row.get("source_run_id") != run_id:
            raise RuntimeError("valuation writer record run identity mismatch")

        method = row.get("method")
        if not isinstance(method, str) or not method or method != method.strip():
            raise RuntimeError("valuation writer record method is invalid")
        if method in payloads:
            raise RuntimeError("valuation writer returned duplicate method records")

        outputs_json = row.get("outputs_json")
        if not isinstance(outputs_json, str):
            raise RuntimeError("valuation writer record outputs are invalid")
        try:
            payload = json.loads(outputs_json)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("valuation writer record outputs are invalid") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("valuation writer record outputs are not an object")
        payloads[method] = payload

    return payloads


def _scorecard_pricing_zone(
    payload: dict[str, Any],
) -> tuple[str, dict[str, Any] | None, tuple[str, float] | None, str | None]:
    """Return (pricing_zone, pricing_zone_detail, sector_specific_anchor,
    revenue_trend_class) from the exact returned scorecard. The sector-specific
    leg mirrors the live packet rule: the technology-adjusted anchor applies
    when the divergence diagnostics are OK and positive (review issue 2a —
    the live watchlist anchors ~15% of names on sector-specific rules; the
    backtest must measure the same rule where the data supports it).
    revenue_trend_class feeds the decline-cap anchor policy (ONE predicate
    with live via anchor_policy.is_decline_class)."""
    pzd = payload.get("pricing_zone_detail")
    zone = str(payload.get("pricing_zone") or "")
    quality_ctx = payload.get("quality_context")
    revenue_trend_class = (
        str(quality_ctx.get("revenue_trend_class"))
        if isinstance(quality_ctx, dict) and quality_ctx.get("revenue_trend_class")
        else None
    )
    sector_specific: tuple[str, float] | None = None
    diagnostics = payload.get("tech_valuation_divergence_diagnostics")
    if isinstance(diagnostics, dict) and diagnostics.get("status") in (None, "OK"):
        adjusted_anchor = diagnostics.get("adjusted_anchor")
        if isinstance(adjusted_anchor, (int, float)) and float(adjusted_anchor) > 0:
            sector_specific = ("technology_adjusted_dcf", float(adjusted_anchor))
    return zone, (pzd if isinstance(pzd, dict) else None), sector_specific, revenue_trend_class


def _expectations_gap(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return a usable gap only when the exact reverse-DCF payload names a bucket."""

    gap = payload.get("expectations_gap") if isinstance(payload, dict) else None
    if not isinstance(gap, dict) or not gap.get("bucket"):
        return None
    return dict(gap)


def reconstruct_signal_asof(
    ticker: str,
    as_of_date: str,
    *,
    provider: Any | None = None,
    filing_lag_days: int = 90,
) -> ReconstructResult:
    """Compute the deterministic cheapness signal for ``ticker`` as-of ``as_of_date``.

    Steps: (1) fetch the historical price at T; (2) run ensure_valuation as-of the
    filing-lag cutoff with the price injected; (3) validate the exact records
    returned by that invocation; (4) derive the anchor and expectations-gap
    bucket from those records. Returns ReconstructResult(signal=None,
    skip_reason=...) when any leg is missing -- never raises for ordinary
    missing-data cases.
    """
    from app.market.price_provider import get_default_provider

    if provider is None:
        provider = get_default_provider()

    snap = provider.get_price_asof(ticker, as_of_date)
    if (
        snap is None
        or isinstance(snap.price, bool)
        or not isinstance(snap.price, (int, float))
        or not math.isfinite(float(snap.price))
        or snap.price <= 0
    ):
        return ReconstructResult(ticker, as_of_date, None, "NO_PRICE_AT_AS_OF")
    price = float(snap.price)
    raw_price = getattr(snap, "raw_price", None)
    if (
        isinstance(raw_price, bool)
        or not isinstance(raw_price, (int, float))
        or not math.isfinite(float(raw_price))
        or raw_price <= 0
    ):
        # Historical shares are filed on their literal as-of basis. An
        # adjusted close without its matching raw close cannot prove the same
        # split basis, so it is not a valid valuation or classification input.
        return ReconstructResult(ticker, as_of_date, None, "NEEDS_DATA:RAW_PRICE_BASIS")
    classification_price = float(raw_price)
    price_basis = "raw_close"

    cutoff = effective_asof_cutoff(as_of_date, filing_lag_days=filing_lag_days)
    # Sector-model routing exclusion (review ANCHOR-5): names live anchors on
    # the insurance model (or suppresses generic methods for) are unmeasurable
    # here — skip with a granular reason so the diagnostic can quantify them.
    if _live_sector_model_routed(ticker, cutoff):
        return ReconstructResult(ticker, as_of_date, None, "FINANCIAL_ISSUER_SECTOR_MODEL")
    run_id = f"backtest_recon_{as_of_date}"
    # Everything from here reads/writes the measurement sibling table —
    # reconstruction can never overwrite the production valuation record.
    from app.valuation.measurement import measurement_scope

    with measurement_scope():
        try:
            records = ensure_valuation(
                ticker,
                cutoff,
                provider=provider,
                run_id=run_id,
                # The valuation writer multiplies this quote by filed as-of
                # shares and uses it for pricing zones / reverse DCF. Feed the
                # unadjusted close on the same share basis; retain ``price``
                # only for adjusted entry/exit return legs.
                price_override=classification_price,
                force_refresh=True,
                require_filed_asof=True,
                raise_on_error=True,
            )
            payloads = _writer_output_payloads(
                records,
                ticker=ticker,
                as_of_date=cutoff,
                run_id=run_id,
            )
        except Exception as exc:  # noqa: BLE001 - a single ticker must not abort the run
            return ReconstructResult(
                ticker, as_of_date, None, f"VALUATION_ERROR:{type(exc).__name__}"
            )

        scorecard = payloads.get("scorecard")
        if scorecard is None:
            return ReconstructResult(ticker, as_of_date, None, "NO_VALUATION_ANCHOR")
        zone, pzd, sector_specific, revenue_trend_class = _scorecard_pricing_zone(scorecard)
        gap = _expectations_gap(payloads.get("reverse_dcf"))

    if not pzd:
        return ReconstructResult(ticker, as_of_date, None, "NO_VALUATION_ANCHOR")
    # Zone gate (audit: anomaly-zone-still-anchors-in-backtest): production
    # suppresses VALUATION_ANOMALY names (negative EPV or DCF), so the backtest
    # must not lift the surviving positive method from the anomaly detail dict
    # as a full-confidence anchor. Mirrors the production signal map.
    if zone == "VALUATION_ANOMALY":
        return ReconstructResult(ticker, as_of_date, None, "VALUATION_ANOMALY")
    if zone not in ("MARGIN_OF_SAFETY", "GROWTH_DEPENDENT", "SPECULATIVE_PREMIUM"):
        # Granular reason (review issue: sample-composition shifts must be
        # quantifiable in the diagnostic, not lumped under NO_VALUATION_ANCHOR).
        reason = f"ZONE_{zone}" if zone else "NO_VALUATION_ANCHOR"
        return ReconstructResult(ticker, as_of_date, None, reason)
    # Share-count stability gate (audit: stable-shares-stale-on-capital-events):
    # an uncorroborated >50% share jump means the per-share anchor may be wrong
    # by the full event factor — exclude from deploy_ready classification.
    if pzd.get("shares_flag") == "SHARES_LATEST_FY_OUTLIER":
        return ReconstructResult(ticker, as_of_date, None, "SHARES_UNSTABLE")
    # ONE shared anchor rule with the live packet layer (audit:
    # anchor-rule-backtest-vs-live-divergence): max of positive (dcf, epv),
    # graham->ncav fallback, via app.valuation.anchor_policy.
    selection = select_anchor(
        dcf=pzd.get("dcf_base") if isinstance(pzd.get("dcf_base"), (int, float)) else None,
        epv=pzd.get("epv_adjusted") if isinstance(pzd.get("epv_adjusted"), (int, float)) else None,
        graham=(
            pzd.get("graham_value_per_share")
            if isinstance(pzd.get("graham_value_per_share"), (int, float))
            else None
        ),
        ncav=(
            pzd.get("ncav_value_per_share")
            if isinstance(pzd.get("ncav_value_per_share"), (int, float))
            else None
        ),
        sector_specific=sector_specific,
        decline_class=is_decline_class(revenue_trend_class),
    )
    if selection.value is None:
        return ReconstructResult(ticker, as_of_date, None, "NO_VALUATION_ANCHOR")
    anchor = selection.value
    # Anchor-sanity band, mirroring the live watchlist anchor quarantine (same helper,
    # same [0.2x, 5.0x] band, same verdict vocabulary): an anchor orders of
    # magnitude away from the quote at T is a magnitude/units/junk-quote
    # artifact, not deep value — one $0.009 OTC quote resolved +11,184pp and
    # single-handedly inflated a headline mean. Evaluated against the
    # CLASSIFICATION price (raw close when available — the basis the anchor
    # is compared on).
    sanity = evaluate_anchor(anchor, classification_price)
    if sanity.verdict != "OK":
        return ReconstructResult(ticker, as_of_date, None, sanity.verdict)
    buy_target = round(anchor * _NEUTRAL_DISCOUNT, 4)
    mos = (anchor - classification_price) / anchor
    deploy_ready = classification_price <= buy_target
    bucket = str((gap or {}).get("bucket") or "EXPECTATIONS_GAP_UNRELIABLE")

    signal = ReconstructedSignal(
        ticker=ticker.upper(),
        as_of_date=as_of_date,
        anchor=anchor,
        price=price,
        buy_price_target=buy_target,
        mos=round(mos, 6),
        deploy_ready=deploy_ready,
        expectations_gap_bucket=bucket,
        # as-of cap band (raw price_T x shares<=cutoff — same share basis)
        # -> drives SPY vs IWM benchmark selection
        cap_category=cap_category_asof(ticker, cutoff, classification_price),
        pricing_zone=zone,
        classification_price=classification_price,
        price_basis=price_basis,
    )
    return ReconstructResult(ticker, as_of_date, signal, None)
