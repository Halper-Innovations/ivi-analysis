from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config import AppConfig, get_config
from app.db import get_db, utc_now_iso
from app.diff.schemas import FilingDiffReport
from app.patterns.catalog import get_pattern_definition
from app.patterns.scanner import load_pattern_scan_report
from app.patterns.schemas import PatternScanReport
from app.synthesis.schemas import SignalEvidence, VariantPerception, VariantPerceptionReport
from app.valuation.anchor_policy import published_dcf_base
from app.valuation.facts import _dedupe_refs
from app.valuation.lineage import latest_decision_eligible_valuation_rows

_ALL_SOURCES: tuple[str, ...] = ("VALUATION", "FILING_DIFF", "PATTERN", "INTANGIBLE_ECONOMICS")


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _get_num(mapping: dict[str, Any] | None, key: str) -> float | None:
    if not isinstance(mapping, dict):
        return None
    value = mapping.get(key)
    return float(value) if _is_num(value) else None


def _safe_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def variant_perception_report_path(
    ticker: str, as_of_date: str, *, cfg: AppConfig | None = None
) -> Path:
    cfg = cfg or get_config()
    return (
        cfg.outputs_dir
        / "variant_perceptions"
        / f"{str(ticker).strip().upper()}_{str(as_of_date).strip()}.json"
    )


def load_variant_perception_report(
    *,
    ticker: str,
    as_of_date: str,
    cfg: AppConfig | None = None,
) -> VariantPerceptionReport | None:
    path = variant_perception_report_path(ticker, as_of_date, cfg=cfg)
    if not path.exists():
        return None
    payload = _safe_json(path)
    if not payload:
        return None
    try:
        return VariantPerceptionReport.model_validate(payload)
    except Exception:
        return None


def _method_payload(valuations: dict[str, Any], name: str) -> dict[str, Any]:
    payload = valuations.get(name)
    return payload if isinstance(payload, dict) else {}


def _method_outputs(valuations: dict[str, Any], name: str) -> dict[str, Any]:
    payload = _method_payload(valuations, name)
    outputs = payload.get("outputs")
    return outputs if isinstance(outputs, dict) else {}


def _method_inputs(valuations: dict[str, Any], name: str) -> dict[str, Any]:
    payload = _method_payload(valuations, name)
    inputs = payload.get("inputs")
    return inputs if isinstance(inputs, dict) else {}


def _valuation_path(*parts: str) -> list[str]:
    return ["valuations." + ".".join(parts)]


def _signal_strength(rank: float) -> str:
    if rank >= 0.75:
        return "HIGH"
    if rank >= 0.40:
        return "MEDIUM"
    return "LOW"


def _diff_path(ticker: str, run_id: str, *, cfg: AppConfig) -> Path:
    return cfg.outputs_dir / "diffs" / f"{ticker.upper()}_{run_id}_diff.json"


def _load_diff_report(*, ticker: str, run_id: str, cfg: AppConfig) -> FilingDiffReport | None:
    path = _diff_path(ticker, run_id, cfg=cfg)
    if not path.exists():
        return None
    payload = _safe_json(path)
    if not payload:
        return None
    try:
        return FilingDiffReport.model_validate(payload)
    except Exception:
        return None


def _load_valuation_payload(*, ticker: str, as_of_date: str) -> dict[str, Any]:
    with get_db() as conn:
        rows = latest_decision_eligible_valuation_rows(
            conn,
            ticker=ticker,
            as_of_date=as_of_date,
            exact_as_of_date=True,
        )
    valuations: dict[str, Any] = {}
    for row in rows:
        try:
            valuations[str(row["method"])] = {
                "inputs": json.loads(row["inputs_json"] or "{}"),
                "outputs": json.loads(row["outputs_json"] or "{}"),
                "warnings": json.loads(row["warnings_json"] or "[]"),
            }
        except Exception:
            continue
    return valuations


def _load_intangible_payload(*, ticker: str, run_id: str, cfg: AppConfig) -> dict[str, Any] | None:
    candidates = [
        cfg.outputs_dir / "universe" / run_id / "intangible_economics.json",
        cfg.sectors_dir / run_id / "intangible_economics.json",
    ]
    for path in candidates:
        payload = _safe_json(path)
        rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
        for row in rows:
            if (
                isinstance(row, dict)
                and str(row.get("ticker") or "").strip().upper() == ticker.upper()
            ):
                return row
    return None


def _coerce_diff_report(
    payload: FilingDiffReport | dict[str, Any] | None,
) -> FilingDiffReport | None:
    if isinstance(payload, FilingDiffReport):
        return payload
    if isinstance(payload, dict) and payload:
        try:
            return FilingDiffReport.model_validate(payload)
        except Exception:
            return None
    return None


def _coerce_pattern_report(
    payload: PatternScanReport | dict[str, Any] | None,
) -> PatternScanReport | None:
    if isinstance(payload, PatternScanReport):
        return payload
    if isinstance(payload, dict) and payload:
        try:
            return PatternScanReport.model_validate(payload)
        except Exception:
            return None
    return None


def _category_payload(
    valuations: dict[str, Any], override: dict[str, Any] | None
) -> dict[str, Any]:
    if isinstance(override, dict) and override:
        return override
    tech_outputs = _method_outputs(valuations, "tech_adjustment")
    category = tech_outputs.get("category_classification")
    return category if isinstance(category, dict) else {}


def _supports_under(text: str) -> bool:
    lowered = text.lower()
    return any(
        token in lowered
        for token in (
            "removed",
            "completed investment",
            "investment completion",
            "efficiency",
            "margin expansion",
            "capacity utilization",
            "customer concentration risk factor removed",
            "diversified",
        )
    )


def _supports_over(text: str) -> bool:
    lowered = text.lower()
    return any(
        token in lowered
        for token in (
            "competitive",
            "pricing pressure",
            "risk",
            "litigation",
            "customer concentration",
            "margin compression",
            "weaker",
            "deteriorat",
            "dilution",
            "demand softness",
            "material weakness",
        )
    )


def _extract_valuation_signals(
    *,
    ticker: str,
    valuations: dict[str, Any],
) -> tuple[list[SignalEvidence], float | None, float | None]:
    signals: list[SignalEvidence] = []
    reverse_outputs = _method_outputs(valuations, "reverse_dcf")
    reverse_inputs = _method_inputs(valuations, "reverse_dcf")
    reverse_detail = (
        reverse_outputs.get("outputs") if isinstance(reverse_outputs.get("outputs"), dict) else {}
    )
    market_implied_growth = _get_num(reverse_detail, "implied_growth")
    if reverse_detail.get("implied_growth_saturated"):
        # A clipped bound is not a solve — do not emit HIGH_MARKET_IMPLIED_
        # GROWTH evidence or derive fair growth from it (audit:
        # saturated-bound-leaks-to-flag-ignoring-consumers).
        market_implied_growth = None
    market_price = _get_num(reverse_inputs, "price") or _get_num(reverse_inputs, "market_price")

    scorecard = _method_outputs(valuations, "scorecard")
    pricing_zone_detail = (
        scorecard.get("pricing_zone_detail")
        if isinstance(scorecard.get("pricing_zone_detail"), dict)
        else {}
    )
    # Memo evidence is built from the DCF the platform stands behind. Reading the
    # raw dcf row turned a spike-inflated $50 against a $28 price into a HIGH
    # "about 79% upside" argument for a name the writer had anchored at $30,
    # where the honest upside is 7% and no signal is due.
    #
    # The durable base corrects the plain dcf row only, and only when that row has
    # a measured base to correct. It is never applied to dcf_adjusted (a different
    # method with its own value) and never conjures a DCF when no dcf/dcf_adjusted
    # row exists — a phantom "DCF_ADJUSTED_DISCOUNT" used to appear for any
    # spike-flagged name that had no adjusted method at all.
    dcf_raw_base = _get_num(_method_outputs(valuations, "dcf"), "base")
    dcf_base = (
        published_dcf_base(pricing_zone_detail, dcf_raw_base)
        if _is_num(dcf_raw_base)
        else None
    )
    dcf_adjusted = _get_num(_method_outputs(valuations, "dcf_adjusted"), "base")
    epv_value = _get_num(_method_outputs(valuations, "epv"), "value_per_share")
    epv_adjusted = _get_num(_method_outputs(valuations, "epv_adjusted"), "value_per_share")
    graham_value = _get_num(_method_outputs(valuations, "graham"), "value_per_share")
    track = (
        scorecard.get("track_comparison")
        if isinstance(scorecard.get("track_comparison"), dict)
        else {}
    )
    tech_adjustment = _method_outputs(valuations, "tech_adjustment")
    rnd_adjustment = (
        tech_adjustment.get("rnd_adjustment")
        if isinstance(tech_adjustment.get("rnd_adjustment"), dict)
        else {}
    )
    category_payload = (
        tech_adjustment.get("category_classification")
        if isinstance(tech_adjustment.get("category_classification"), dict)
        else {}
    )
    tech_divergence = _get_num(tech_adjustment, "tech_valuation_divergence")

    if _is_num(market_implied_growth) and float(market_implied_growth) > 0.15:
        signals.append(
            SignalEvidence(
                source="VALUATION",
                signal_type="HIGH_MARKET_IMPLIED_GROWTH",
                direction="SUPPORTS_OVERVALUED",
                strength="HIGH" if float(market_implied_growth) > 0.20 else "MEDIUM",
                summary=f"Reverse DCF implies about {float(market_implied_growth):.1%} growth, which is a demanding market expectation.",
                derived_from=_valuation_path("reverse_dcf", "outputs", "outputs", "implied_growth"),
            )
        )

    if _is_num(market_price) and float(market_price) > 0:
        # An adjusted method is the same model re-run on adjusted inputs, so when it
        # lands on the same side of the price as its plain twin it is the same
        # evidence, not a second piece: emitting both doubled the HIGH-strength count
        # and the fair-growth adjustment. The adjusted signal is kept when its twin
        # produced no signal or the opposite one.
        emitted_kind: dict[str, str] = {}
        for method_name, family, value in (
            ("dcf", "dcf", dcf_base),
            ("dcf_adjusted", "dcf", dcf_adjusted),
            ("epv", "epv", epv_value),
            ("epv_adjusted", "epv", epv_adjusted),
            ("graham", "graham", graham_value),
        ):
            if not _is_num(value) or float(value) <= 0:
                continue
            upside = (float(value) - float(market_price)) / float(market_price)
            kind = "DISCOUNT" if upside > 0.30 else "PREMIUM" if upside < -0.15 else ""
            if kind and method_name.endswith("_adjusted") and emitted_kind.get(family) == kind:
                continue
            if kind:
                emitted_kind.setdefault(family, kind)
            if upside > 0.30:
                signals.append(
                    SignalEvidence(
                        source="VALUATION",
                        signal_type=f"{method_name.upper()}_DISCOUNT",
                        direction="SUPPORTS_UNDERVALUED",
                        strength="HIGH" if upside > 0.50 else "MEDIUM",
                        summary=f"{method_name.upper()} indicates about {upside:.0%} upside versus the current price anchor.",
                        derived_from=_valuation_path(
                            method_name,
                            "outputs",
                            "base" if method_name.startswith("dcf") else "value_per_share",
                        )
                        + _valuation_path("reverse_dcf", "inputs", "price"),
                    )
                )
            elif upside < -0.15:
                signals.append(
                    SignalEvidence(
                        source="VALUATION",
                        signal_type=f"{method_name.upper()}_PREMIUM",
                        direction="SUPPORTS_OVERVALUED",
                        strength="HIGH" if upside < -0.30 else "MEDIUM",
                        summary=f"{method_name.upper()} sits about {abs(upside):.0%} below the current price anchor.",
                        derived_from=_valuation_path(
                            method_name,
                            "outputs",
                            "base" if method_name.startswith("dcf") else "value_per_share",
                        )
                        + _valuation_path("reverse_dcf", "inputs", "price"),
                    )
                )

    # OVERVALUED lives in legacy_signal; "signal" is the zone action
    # (BUY/HOLD/PASS/...) and can never equal OVERVALUED (audit:
    # deploy-inside-growth-dependent-zone, fix 2).
    scorecard_signal = str(scorecard.get("legacy_signal") or "").upper()
    if scorecard_signal == "OVERVALUED":
        signals.append(
            SignalEvidence(
                source="VALUATION",
                signal_type="SCORECARD_OVERVALUED",
                direction="SUPPORTS_OVERVALUED",
                strength="MEDIUM",
                summary="The margin-of-safety scorecard flags the current price as overvalued against the available earnings-based methods.",
                derived_from=_valuation_path("scorecard", "outputs", "legacy_signal"),
            )
        )

    gaap_stance = str(track.get("gaap_stance") or "")
    adjusted_stance = str(track.get("adjusted_stance") or "")
    if gaap_stance == "AVOID" and adjusted_stance == "INVESTIGATE":
        signals.append(
            SignalEvidence(
                source="VALUATION",
                signal_type="GAAP_ACCOUNTING_MASKS_ECONOMICS",
                direction="SUPPORTS_UNDERVALUED",
                strength="MEDIUM",
                summary="GAAP-basis valuation says avoid, while the R&D-adjusted track says investigate, implying reported accounting may understate true economics.",
                derived_from=_dedupe_refs(
                    _valuation_path("scorecard", "outputs", "track_comparison", "gaap_stance")
                    + _valuation_path("scorecard", "outputs", "track_comparison", "adjusted_stance")
                    + [
                        str(ref)
                        for ref in (category_payload.get("derived_from") or [])
                        if str(ref).strip()
                    ]
                    + [
                        str(ref)
                        for ref in (rnd_adjustment.get("derived_from") or [])
                        if str(ref).strip()
                    ]
                ),
            )
        )

    if _is_num(tech_divergence) and float(tech_divergence) > 0.30:
        signals.append(
            SignalEvidence(
                source="VALUATION",
                signal_type="RND_ACCOUNTING_DIVERGENCE",
                direction="SUPPORTS_UNDERVALUED",
                strength="HIGH" if float(tech_divergence) > 0.50 else "MEDIUM",
                summary=(
                    f"R&D-adjusted intrinsic value is {float(tech_divergence):+.1%} above the GAAP anchor"
                    + (
                        f" for the {str(category_payload.get('category') or '').strip()} category."
                        if str(category_payload.get("category") or "").strip()
                        else "."
                    )
                ),
                derived_from=_dedupe_refs(
                    _valuation_path("tech_adjustment", "outputs", "tech_valuation_divergence")
                    + [
                        str(ref)
                        for ref in (category_payload.get("derived_from") or [])
                        if str(ref).strip()
                    ]
                    + [
                        str(ref)
                        for ref in (rnd_adjustment.get("derived_from") or [])
                        if str(ref).strip()
                    ]
                ),
            )
        )

    estimated_fair_growth = market_implied_growth
    return signals, market_implied_growth, estimated_fair_growth


def _extract_diff_signals(*, report: FilingDiffReport | None) -> list[SignalEvidence]:
    if not isinstance(report, FilingDiffReport):
        return []
    signals: list[SignalEvidence] = []
    for change in report.changes:
        summary = str(change.summary or "").strip()
        change_type = str(change.change_type or "")
        direction = "NEUTRAL"
        if change_type in {"NEW_DISCLOSURE", "RISK_SIGNAL", "COMPETITIVE_SIGNAL"}:
            direction = "SUPPORTS_OVERVALUED"
        elif change_type == "REMOVED_DISCLOSURE":
            direction = "SUPPORTS_UNDERVALUED"
        elif change_type == "STRATEGIC_SIGNAL":
            if _supports_over(summary):
                direction = "SUPPORTS_OVERVALUED"
            elif _supports_under(summary) or "completed" in summary.lower():
                direction = "SUPPORTS_UNDERVALUED"
        elif change_type == "LANGUAGE_SHIFT":
            if _supports_over(summary):
                direction = "SUPPORTS_OVERVALUED"
            elif _supports_under(summary):
                direction = "SUPPORTS_UNDERVALUED"
        if direction == "NEUTRAL":
            continue
        materiality = str(change.materiality or "LOW").upper()
        signals.append(
            SignalEvidence(
                source="FILING_DIFF",
                signal_type=change_type,
                direction=direction,
                strength=materiality if materiality in {"HIGH", "MEDIUM", "LOW"} else "LOW",
                summary=summary,
                derived_from=[
                    f"diff.{report.ticker}.{change.section}.{change.fiscal_year_from}-{change.fiscal_year_to}.{change.change_type}"
                ],
            )
        )
    return signals


def _pattern_direction(pattern_id: str, hit: Any, result: Any) -> str:
    details = str(getattr(hit, "outcome_details", "") or "").lower()
    outcome_value = getattr(hit, "outcome_value", None)
    if pattern_id in {
        "deferred_revenue_leading_indicator",
        "capex_to_depreciation_divergence",
        "rnd_intensity_inflection",
        "capital_return_inflection",
    }:
        return "SUPPORTS_UNDERVALUED"
    if pattern_id == "cash_conversion_quality_divergence":
        if (
            "weaker subsequent earnings" in details
            or "low cfo-to-net-income" in details
            or (_is_num(outcome_value) and float(outcome_value) < 0)
        ):
            return "SUPPORTS_OVERVALUED"
        return "SUPPORTS_UNDERVALUED"
    if pattern_id == "gross_margin_regime_change":
        if (
            "downward" in details
            or "deteriorated" in details
            or (_is_num(outcome_value) and float(outcome_value) < 0)
        ):
            return "SUPPORTS_OVERVALUED"
        return "SUPPORTS_UNDERVALUED"
    return "NEUTRAL"


def _extract_pattern_signals(
    *, report: PatternScanReport | None, ticker: str
) -> list[SignalEvidence]:
    if not isinstance(report, PatternScanReport):
        return []
    ticker_norm = ticker.upper()
    signals: list[SignalEvidence] = []
    for result in report.pattern_results:
        direction = "NEUTRAL"
        for hit in result.hits:
            if hit.ticker.upper() != ticker_norm:
                continue
            direction = _pattern_direction(result.pattern_id, hit, result)
            if direction == "NEUTRAL":
                continue
            hit_rate = float(result.hit_rate) if _is_num(result.hit_rate) else None
            if hit_rate is not None and hit_rate <= 0.5 and int(result.sample_size or 0) >= 3:
                continue
            strength_rank = float(hit.detection_strength)
            if hit_rate is not None:
                strength_rank = max(strength_rank, hit_rate)
            definition = get_pattern_definition(result.pattern_id)
            years_text = (
                ", ".join(str(year) for year in hit.years_detected)
                if hit.years_detected
                else "unknown years"
            )
            summary = (
                f"{definition.name if definition else result.pattern_id} was detected in {ticker_norm} during {years_text}"
                + (
                    f", with a {hit_rate:.0%} historical hit rate across {result.sample_size} checkable peers."
                    if hit_rate is not None and int(result.sample_size or 0) > 0
                    else "."
                )
            )
            signals.append(
                SignalEvidence(
                    source="PATTERN",
                    signal_type=result.pattern_id,
                    direction=direction,
                    strength=_signal_strength(strength_rank),
                    summary=summary,
                    derived_from=list(hit.derived_from),
                )
            )
    return signals


def _extract_intangible_signals(*, payload: dict[str, Any] | None) -> list[SignalEvidence]:
    if not isinstance(payload, dict) or not payload:
        return []
    signals: list[SignalEvidence] = []
    total = _get_num(payload, "intangible_economics_total")
    rnd_productivity = _get_num(payload, "rnd_productivity_score")
    owner_value_capture = _get_num(payload, "owner_value_capture_score")
    gross_margin_durability = _get_num(payload, "gross_margin_durability_score")
    reason_codes = [
        str(code)
        for code in (payload.get("intangible_economics_reason_codes") or [])
        if str(code).strip()
    ]

    if _is_num(total) and float(total) >= 12.0:
        signals.append(
            SignalEvidence(
                source="INTANGIBLE_ECONOMICS",
                signal_type="HIGH_INTANGIBLE_ECONOMICS_TOTAL",
                direction="SUPPORTS_UNDERVALUED",
                strength="HIGH" if float(total) >= 14.0 else "MEDIUM",
                summary=f"Intangible economics total is {float(total):.1f}, indicating unusually durable non-physical economics.",
                derived_from=[
                    str(ref) for ref in (payload.get("derived_from") or []) if str(ref).strip()
                ],
            )
        )
    elif _is_num(total) and float(total) <= 8.0:
        signals.append(
            SignalEvidence(
                source="INTANGIBLE_ECONOMICS",
                signal_type="WEAK_INTANGIBLE_ECONOMICS_TOTAL",
                direction="SUPPORTS_OVERVALUED",
                strength="HIGH" if float(total) <= 6.0 else "MEDIUM",
                summary=f"Intangible economics total is only {float(total):.1f}, which is weak support for a premium multiple.",
                derived_from=[
                    str(ref) for ref in (payload.get("derived_from") or []) if str(ref).strip()
                ],
            )
        )
    if _is_num(rnd_productivity) and float(rnd_productivity) < 1.0:
        signals.append(
            SignalEvidence(
                source="INTANGIBLE_ECONOMICS",
                signal_type="LOW_RND_PRODUCTIVITY",
                direction="SUPPORTS_OVERVALUED",
                strength="MEDIUM",
                summary="R&D productivity score is weak, so incremental reinvestment is not yet translating cleanly into owner value.",
                derived_from=[
                    str(ref) for ref in (payload.get("derived_from") or []) if str(ref).strip()
                ],
            )
        )
    if _is_num(owner_value_capture) and float(owner_value_capture) < 2.0:
        signals.append(
            SignalEvidence(
                source="INTANGIBLE_ECONOMICS",
                signal_type="WEAK_OWNER_VALUE_CAPTURE",
                direction="SUPPORTS_OVERVALUED",
                strength="MEDIUM",
                summary="Owner value capture is weak, suggesting dilution or weak per-share translation is limiting economic compounding.",
                derived_from=[
                    str(ref) for ref in (payload.get("derived_from") or []) if str(ref).strip()
                ],
            )
        )
    if (
        _is_num(gross_margin_durability)
        and float(gross_margin_durability) >= 4.0
        and _is_num(rnd_productivity)
        and float(rnd_productivity) >= 2.0
    ):
        signals.append(
            SignalEvidence(
                source="INTANGIBLE_ECONOMICS",
                signal_type="DURABLE_INTANGIBLE_ECONOMICS",
                direction="SUPPORTS_UNDERVALUED",
                strength="MEDIUM",
                summary="Gross-margin durability and R&D productivity both read as supportive, which strengthens the case that the business economics are more durable than GAAP headlines suggest.",
                derived_from=[
                    str(ref) for ref in (payload.get("derived_from") or []) if str(ref).strip()
                ],
            )
        )
    if any(
        code in {"REASON_EXCESS_DILUTION", "REASON_WEAK_PER_SHARE_CAPTURE"} for code in reason_codes
    ):
        signals.append(
            SignalEvidence(
                source="INTANGIBLE_ECONOMICS",
                signal_type="DILUTION_OR_PER_SHARE_CAPTURE_RISK",
                direction="SUPPORTS_OVERVALUED",
                strength="MEDIUM",
                summary="Intangible economics reason codes flag dilution or weak per-share capture, which weakens the case for market-implied growth.",
                derived_from=[
                    str(ref) for ref in (payload.get("derived_from") or []) if str(ref).strip()
                ],
            )
        )
    return signals


def _signals_by_direction(signals: list[SignalEvidence], direction: str) -> list[SignalEvidence]:
    return [signal for signal in signals if signal.direction == direction]


def _source_count(signals: list[SignalEvidence]) -> int:
    return len({signal.source for signal in signals})


def _summarize_signals(signals: list[SignalEvidence]) -> dict[str, Any]:
    by_source = {source: 0 for source in _ALL_SOURCES}
    by_direction = {"SUPPORTS_UNDERVALUED": 0, "SUPPORTS_OVERVALUED": 0, "NEUTRAL": 0}
    by_source_direction: dict[str, dict[str, int]] = {
        source: {"SUPPORTS_UNDERVALUED": 0, "SUPPORTS_OVERVALUED": 0, "NEUTRAL": 0}
        for source in _ALL_SOURCES
    }
    for signal in signals:
        by_source[signal.source] = by_source.get(signal.source, 0) + 1
        by_direction[signal.direction] = by_direction.get(signal.direction, 0) + 1
        bucket = by_source_direction.setdefault(
            signal.source, {"SUPPORTS_UNDERVALUED": 0, "SUPPORTS_OVERVALUED": 0, "NEUTRAL": 0}
        )
        bucket[signal.direction] = bucket.get(signal.direction, 0) + 1
    return {
        "total_signals": len(signals),
        "by_source": by_source,
        "by_direction": by_direction,
        "by_source_direction": by_source_direction,
    }


def _signal_sort_key(signal: SignalEvidence) -> tuple[int, str, str]:
    return (
        {"LOW": 0, "MEDIUM": 1, "HIGH": 2}.get(signal.strength, 0),
        signal.source,
        signal.signal_type,
    )


def _estimate_fair_growth(
    *,
    market_implied_growth: float | None,
    supporting: list[SignalEvidence],
    contradicting: list[SignalEvidence],
    direction: str,
) -> tuple[float | None, float | None]:
    if not _is_num(market_implied_growth):
        return None, None
    delta = 0.0
    strength_weight = {"HIGH": 0.03, "MEDIUM": 0.02, "LOW": 0.01}
    sign = 1.0 if direction == "UNDERVALUED" else -1.0
    for signal in supporting:
        delta += sign * strength_weight.get(signal.strength, 0.01)
    for signal in contradicting:
        delta -= sign * (strength_weight.get(signal.strength, 0.01) * 0.5)
    estimated = float(market_implied_growth) + max(-0.12, min(0.12, delta))
    gap = None
    if float(market_implied_growth) != 0.0:
        gap = (estimated - float(market_implied_growth)) / abs(float(market_implied_growth))
    else:
        gap = estimated - float(market_implied_growth)
    return estimated, gap


def _prediction_bundle(
    *,
    direction: str,
    supporting: list[SignalEvidence],
    market_implied_growth: float | None,
    estimated_fair_growth: float | None,
) -> tuple[str, str, str, str]:
    signal_types = {signal.signal_type for signal in supporting}
    if direction == "UNDERVALUED":
        if "deferred_revenue_leading_indicator" in signal_types:
            return (
                "Revenue growth should accelerate within the next 1-2 annual filings, with deferred revenue continuing to grow faster than recognized revenue.",
                "MEDIUM",
                "A follow-up earnings release or annual filing showing backlog conversion into faster recognized revenue growth.",
                "If deferred revenue growth falls back below recognized revenue growth and revenue does not accelerate, the thesis is weakened.",
            )
        if (
            "RND_ACCOUNTING_DIVERGENCE" in signal_types
            or "GAAP_ACCOUNTING_MASKS_ECONOMICS" in signal_types
        ):
            return (
                "Operating margin should expand by more than 2 percentage points within the next 2 annual filings as investment spend is harvested into reported earnings.",
                "MEDIUM",
                "The next 1-2 annual filings showing margin expansion or a narrower gap between GAAP and adjusted earnings power.",
                "If margins fail to inflect or adjusted economics do not translate into reported earnings, the accounting-mismatch thesis is wrong.",
            )
        return (
            "Within the next 1-2 annual filings, growth or margins should run ahead of the market's current expectations and force a rerating.",
            "MEDIUM",
            "A filing or earnings release that converts today’s operating or pattern signals into visibly stronger reported growth and margins.",
            "If reported growth and margins fail to improve over the next 1-2 annual filings, the undervaluation case weakens materially.",
        )
    if _is_num(market_implied_growth):
        threshold = max(0.0, float(market_implied_growth) - 0.03)
        prediction = f"Revenue growth should fall below roughly {threshold:.1%} or operating margin should contract within the next 1-2 annual filings."
    else:
        prediction = "Growth or margins should disappoint relative to the current market narrative within the next 1-2 annual filings."
    if "gross_margin_regime_change" in signal_types:
        prediction = "Gross or operating margin should deteriorate by more than 1 percentage point within the next 1-2 annual filings if the pressure signaled by the pattern is real."
    return (
        prediction,
        "SHORT" if "HIGH_MARKET_IMPLIED_GROWTH" in signal_types else "MEDIUM",
        "The next earnings release or annual filing that reveals slower growth, weaker cash conversion, or margin compression versus the current bar.",
        "If growth and margins continue to meet or exceed the implied bar, the overvaluation thesis is invalidated.",
    )


def _build_thesis(
    *,
    ticker: str,
    direction: str,
    market_implied_growth: float | None,
    estimated_fair_growth: float | None,
    supporting: list[SignalEvidence],
) -> str:
    top_support = supporting[:3]
    support_text = "; ".join(signal.summary.rstrip(".") for signal in top_support)
    if _is_num(market_implied_growth) and _is_num(estimated_fair_growth):
        return (
            f"The market is pricing {ticker.upper()} at about {float(market_implied_growth):.1%} implied growth, "
            f"but {support_text}, suggesting fair growth is closer to {float(estimated_fair_growth):.1%}."
            if direction == "UNDERVALUED"
            else f"The market is pricing {ticker.upper()} at about {float(market_implied_growth):.1%} implied growth, "
            f"but {support_text}, suggesting fair growth is materially lower than the market-implied bar."
        )
    return f"Available valuation and signal convergence suggests {ticker.upper()} is {direction.lower()} because {support_text}."


def _confidence_for_direction(
    *,
    direction: str,
    supporting: list[SignalEvidence],
    contradicting: list[SignalEvidence],
) -> str:
    source_count = _source_count(supporting)
    high_count = len([signal for signal in supporting if signal.strength == "HIGH"])
    signal_types = {signal.signal_type for signal in supporting}
    if direction == "UNDERVALUED":
        if source_count >= 3 and {"VALUATION", "PATTERN", "FILING_DIFF"}.issubset(
            {signal.source for signal in supporting}
        ):
            return "HIGH"
        if (
            "RND_ACCOUNTING_DIVERGENCE" in signal_types
            and "HIGH_INTANGIBLE_ECONOMICS_TOTAL" in signal_types
        ):
            return "MEDIUM"
        if "GAAP_ACCOUNTING_MASKS_ECONOMICS" in signal_types:
            return "MEDIUM"
    else:
        if (
            source_count >= 3
            and "HIGH_MARKET_IMPLIED_GROWTH" in signal_types
            and {"FILING_DIFF", "INTANGIBLE_ECONOMICS"}.issubset(
                {signal.source for signal in supporting}
            )
        ):
            return "HIGH"
        if (
            source_count >= 2
            and "HIGH_MARKET_IMPLIED_GROWTH" in signal_types
            and any(signal.source == "PATTERN" for signal in supporting)
        ):
            return "MEDIUM"
    if source_count >= 3 and high_count >= 2 and _source_count(contradicting) < source_count:
        return "HIGH"
    if source_count >= 2:
        return "MEDIUM"
    return "LOW"


def _build_perception(
    *,
    ticker: str,
    as_of_date: str,
    direction: str,
    market_implied_growth: float | None,
    supporting: list[SignalEvidence],
    contradicting: list[SignalEvidence],
    index: int,
) -> VariantPerception:
    estimated_fair_growth, gap_pct = _estimate_fair_growth(
        market_implied_growth=market_implied_growth,
        supporting=supporting,
        contradicting=contradicting,
        direction=direction,
    )
    prediction, time_horizon, catalyst, risk = _prediction_bundle(
        direction=direction,
        supporting=supporting,
        market_implied_growth=market_implied_growth,
        estimated_fair_growth=estimated_fair_growth,
    )
    thesis = _build_thesis(
        ticker=ticker,
        direction=direction,
        market_implied_growth=market_implied_growth,
        estimated_fair_growth=estimated_fair_growth,
        supporting=supporting,
    )
    return VariantPerception(
        perception_id=f"{ticker.upper()}_{as_of_date}_{direction.lower()}_{index}",
        ticker=ticker.upper(),
        as_of_date=as_of_date,
        thesis=thesis,
        direction=direction,
        confidence=_confidence_for_direction(
            direction=direction, supporting=supporting, contradicting=contradicting
        ),
        implied_vs_estimated={
            "market_implied_growth": float(market_implied_growth)
            if _is_num(market_implied_growth)
            else None,
            "estimated_fair_growth": float(estimated_fair_growth)
            if _is_num(estimated_fair_growth)
            else None,
            "gap_pct": float(gap_pct) if _is_num(gap_pct) else None,
        },
        supporting_signals=supporting,
        contradicting_signals=contradicting,
        testable_prediction=prediction,
        time_horizon=time_horizon,
        catalyst=catalyst,
        risk=risk,
        derived_from=_dedupe_refs(
            [ref for signal in supporting + contradicting for ref in signal.derived_from]
        ),
        generated_at=utc_now_iso(),
    )


def _select_perceptions(
    *,
    ticker: str,
    as_of_date: str,
    signals: list[SignalEvidence],
    market_implied_growth: float | None,
) -> list[VariantPerception]:
    undervalued = _signals_by_direction(signals, "SUPPORTS_UNDERVALUED")
    overvalued = _signals_by_direction(signals, "SUPPORTS_OVERVALUED")
    under_sources = _source_count(undervalued)
    over_sources = _source_count(overvalued)
    perceptions: list[VariantPerception] = []

    if under_sources >= 2 and under_sources > over_sources:
        perceptions.append(
            _build_perception(
                ticker=ticker,
                as_of_date=as_of_date,
                direction="UNDERVALUED",
                market_implied_growth=market_implied_growth,
                supporting=sorted(undervalued, key=_signal_sort_key, reverse=True),
                contradicting=sorted(overvalued, key=_signal_sort_key, reverse=True),
                index=1,
            )
        )
    if over_sources >= 2 and over_sources > under_sources:
        perceptions.append(
            _build_perception(
                ticker=ticker,
                as_of_date=as_of_date,
                direction="OVERVALUED",
                market_implied_growth=market_implied_growth,
                supporting=sorted(overvalued, key=_signal_sort_key, reverse=True),
                contradicting=sorted(undervalued, key=_signal_sort_key, reverse=True),
                index=len(perceptions) + 1,
            )
        )
    return perceptions


# A pattern's calibrated hit_rate is only actionable if it rests on enough
# DECISIVE resolutions. With no floor, 1 confirmed + N inconclusive reports a
# hit_rate of 1.0 and would wrongly upgrade confidence. sample_size in the
# calibration weights is the decisive denominator (confirmed + disconfirmed).
_MIN_DECISIVE_SAMPLE = 3


def _pattern_hit_rate(calibration_weights: dict[str, Any], pattern_id: str) -> float | None:
    node = calibration_weights.get("pattern_weights")
    if not isinstance(node, dict):
        return None
    payload = node.get(pattern_id)
    if isinstance(payload, dict):
        if int(payload.get("sample_size") or 0) < _MIN_DECISIVE_SAMPLE:
            return None
        value = payload.get("hit_rate")
        if _is_num(value):
            return float(value)
    return None


def _adjust_perception_confidence(
    perception: VariantPerception,
    calibration_weights: dict[str, Any] | None,
) -> VariantPerception:
    if (
        perception.confidence != "MEDIUM"
        or not isinstance(calibration_weights, dict)
        or not calibration_weights
    ):
        return perception
    rates = [
        _pattern_hit_rate(calibration_weights, signal.signal_type)
        for signal in perception.supporting_signals
        if signal.source == "PATTERN"
    ]
    usable = [rate for rate in rates if _is_num(rate)]
    if not usable:
        return perception
    has_high = any(float(rate) > 0.7 for rate in usable)
    has_low = any(float(rate) < 0.3 for rate in usable)
    if has_high and not has_low:
        return perception.model_copy(update={"confidence": "HIGH"})
    if has_low and not has_high:
        return perception.model_copy(update={"confidence": "LOW"})
    return perception


def build_variant_perceptions(
    *,
    ticker: str,
    as_of_date: str,
    run_id: str,
    cfg: AppConfig | None = None,
    valuation_payload: dict[str, Any] | None = None,
    diff_report: FilingDiffReport | dict[str, Any] | None = None,
    pattern_report: PatternScanReport | dict[str, Any] | None = None,
    intangible_payload: dict[str, Any] | None = None,
    category_payload: dict[str, Any] | None = None,
    calibration_weights: dict[str, Any] | None = None,
    persist: bool = True,
) -> VariantPerceptionReport:
    cfg = cfg or get_config()
    ticker_norm = str(ticker or "").strip().upper()
    valuations = (
        valuation_payload
        if isinstance(valuation_payload, dict)
        else _load_valuation_payload(ticker=ticker_norm, as_of_date=as_of_date)
    )
    diff = _coerce_diff_report(diff_report) or _load_diff_report(
        ticker=ticker_norm, run_id=run_id, cfg=cfg
    )
    patterns = _coerce_pattern_report(pattern_report) or load_pattern_scan_report(run_id, cfg=cfg)
    intangible = (
        intangible_payload
        if isinstance(intangible_payload, dict)
        else _load_intangible_payload(ticker=ticker_norm, run_id=run_id, cfg=cfg)
    )
    category = _category_payload(valuations, category_payload)

    valuation_signals, market_implied_growth, _estimated = _extract_valuation_signals(
        ticker=ticker_norm, valuations=valuations
    )
    diff_signals = _extract_diff_signals(report=diff)
    pattern_signals = _extract_pattern_signals(report=patterns, ticker=ticker_norm)
    intangible_signals = _extract_intangible_signals(payload=intangible)
    signals = valuation_signals + diff_signals + pattern_signals + intangible_signals

    data_quality = {
        "source_status": {
            "VALUATION": "AVAILABLE" if bool(valuations) else "MISSING",
            "FILING_DIFF": "AVAILABLE" if isinstance(diff, FilingDiffReport) else "MISSING",
            "PATTERN": "AVAILABLE" if isinstance(patterns, PatternScanReport) else "MISSING",
            "INTANGIBLE_ECONOMICS": "AVAILABLE"
            if isinstance(intangible, dict) and bool(intangible)
            else "MISSING",
        },
        "available_sources": [
            source
            for source, status in {
                "VALUATION": bool(valuations),
                "FILING_DIFF": isinstance(diff, FilingDiffReport),
                "PATTERN": isinstance(patterns, PatternScanReport),
                "INTANGIBLE_ECONOMICS": isinstance(intangible, dict) and bool(intangible),
            }.items()
            if status
        ],
        "missing_sources": [
            source
            for source, status in {
                "VALUATION": bool(valuations),
                "FILING_DIFF": isinstance(diff, FilingDiffReport),
                "PATTERN": isinstance(patterns, PatternScanReport),
                "INTANGIBLE_ECONOMICS": isinstance(intangible, dict) and bool(intangible),
            }.items()
            if not status
        ],
        "market_implied_growth_available": _is_num(market_implied_growth),
        "tech_category_available": bool(category),
        "valuation_method_count": len(valuations),
        "signal_count": len(signals),
        "too_many_unknowns": len(
            [
                source
                for source in _ALL_SOURCES
                if source
                not in {
                    item
                    for item in [
                        "VALUATION" if bool(valuations) else None,
                        "FILING_DIFF" if isinstance(diff, FilingDiffReport) else None,
                        "PATTERN" if isinstance(patterns, PatternScanReport) else None,
                        "INTANGIBLE_ECONOMICS"
                        if isinstance(intangible, dict) and bool(intangible)
                        else None,
                    ]
                    if item
                }
            ]
        )
        >= 3,
    }

    perceptions = (
        []
        if data_quality["too_many_unknowns"]
        else _select_perceptions(
            ticker=ticker_norm,
            as_of_date=as_of_date,
            signals=signals,
            market_implied_growth=market_implied_growth,
        )
    )
    perceptions = [
        _adjust_perception_confidence(perception, calibration_weights) for perception in perceptions
    ]

    report = VariantPerceptionReport(
        run_id=run_id,
        ticker=ticker_norm,
        as_of_date=as_of_date,
        perceptions=perceptions,
        signal_summary=_summarize_signals(signals),
        data_quality=data_quality,
    )

    if persist:
        path = variant_perception_report_path(ticker_norm, as_of_date, cfg=cfg)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report.model_dump(mode="json"), indent=2), encoding="utf-8")
    return report
