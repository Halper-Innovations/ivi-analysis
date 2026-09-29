"""
Per-Share Capital Allocation Discipline v1

Deterministic per-ticker assessment of how well management converts business-level
economics into per-share owner value. Distinct from raw quality scoring: this layer
asks whether share count discipline and per-share progress are coherent.

Doctrine:
- No technical indicators, no ML, no macro forecasting
- No narrative scoring, no aggressive optimism
- UNKNOWN remains UNKNOWN with explicit reason codes
- Capital allocation discipline does NOT override FAIL or MOS discipline
- High dilution with strong quality → MIXED, not friendly
- Evidence gaps block classification
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from app.config import AppConfig, get_config
from app.db import utc_now_iso


UNKNOWN = "UNKNOWN"
OK = "OK"

# ── Discipline classes ─────────────────────────────────────────────────────────

OWNER_FRIENDLY_DISCIPLINED = "OWNER_FRIENDLY_DISCIPLINED"
MIXED_CAPITAL_ALLOCATION = "MIXED_CAPITAL_ALLOCATION"
OWNER_DILUTIVE_OR_DESTRUCTIVE = "OWNER_DILUTIVE_OR_DESTRUCTIVE"
CAPITAL_ALLOCATION_UNKNOWN = "CAPITAL_ALLOCATION_UNKNOWN"

# ── Caution classes ────────────────────────────────────────────────────────────

CAUTION_SUPPORTIVE = "CAPITAL_ALLOCATION_SUPPORTIVE"
CAUTION_MIXED = "CAPITAL_ALLOCATION_MIXED"
CAUTION_HEADWIND = "CAPITAL_ALLOCATION_HEADWIND"
CAUTION_UNCLEAR = "CAPITAL_ALLOCATION_UNCLEAR"

# ── Support signal constants ───────────────────────────────────────────────────

SIG_LOW_DILUTION = "LOW_DILUTION"
SIG_REVENUE_PER_SHARE_GROWTH = "REVENUE_PER_SHARE_GROWTH_PRESENT"
SIG_OWNER_EARNINGS_PER_SHARE_GROWTH = "OWNER_EARNINGS_PER_SHARE_GROWTH_PRESENT"
SIG_FCF_PER_SHARE_GROWTH = "FCF_PER_SHARE_GROWTH_PRESENT"
SIG_SHARE_COUNT_DISCIPLINE = "SHARE_COUNT_DISCIPLINE_PRESENT"
SIG_OWNER_VALUE_CAPTURE = "OWNER_VALUE_CAPTURE_PRESENT"

# ── Headwind signal constants ──────────────────────────────────────────────────

SIG_HIGH_DILUTION = "HIGH_DILUTION"
SIG_REVENUE_GROWTH_WITHOUT_PER_SHARE = "REVENUE_GROWTH_WITHOUT_PER_SHARE_PROGRESS"
SIG_OE_GROWTH_WITHOUT_PER_SHARE = "OWNER_EARNINGS_GROWTH_WITHOUT_PER_SHARE_CAPTURE"
SIG_FCF_PER_SHARE_STAGNATION = "FCF_PER_SHARE_STAGNATION"
SIG_SHARE_COUNT_HEADWIND = "SHARE_COUNT_HEADWIND"
SIG_POSSIBLE_OWNER_VALUE_LEAKAGE = "POSSIBLE_OWNER_VALUE_LEAKAGE"

# ── Reason codes ───────────────────────────────────────────────────────────────

REASON_EVIDENCE_BLOCKED_MISSING_FACTS = "EVIDENCE_BLOCKED_MISSING_FACTS"
REASON_EVIDENCE_BLOCKED_MISSING_SHARES = "EVIDENCE_BLOCKED_MISSING_SHARES"
REASON_HIGH_DILUTION_DOMINANT = "HIGH_DILUTION_DOMINANT"
REASON_HIGH_DILUTION_OFFSET_BY_QUALITY = "HIGH_DILUTION_OFFSET_BY_QUALITY"
REASON_MODERATE_DILUTION_WITH_QUALITY = "MODERATE_DILUTION_WITH_QUALITY"
REASON_LOW_DILUTION_STRONG_QUALITY = "LOW_DILUTION_STRONG_QUALITY"
REASON_LOW_DILUTION_QUALITY_UNKNOWN = "LOW_DILUTION_QUALITY_UNKNOWN"
REASON_VERY_LOW_QUALITY_DESTRUCTIVE = "VERY_LOW_QUALITY_DESTRUCTIVE"
REASON_EVIDENCE_INSUFFICIENT_QUALITY = "EVIDENCE_INSUFFICIENT_FOR_QUALITY_JUDGMENT"
REASON_BUYBACKS_QUALITY_CONFIRMED = "BUYBACKS_WITH_QUALITY_CONFIRMED"
REASON_SHAREHOLDER_FRIENDLY_ALLOCATION = "SHAREHOLDER_FRIENDLY_ALLOCATION"

# ── Ordering: best → worst ─────────────────────────────────────────────────────

_DISCIPLINE_ORDER = {
    OWNER_FRIENDLY_DISCIPLINED: 0,
    MIXED_CAPITAL_ALLOCATION: 1,
    CAPITAL_ALLOCATION_UNKNOWN: 2,
    OWNER_DILUTIVE_OR_DESTRUCTIVE: 3,
}


def _is_num(value: Any) -> bool:
    # NaN and infinity are not measurements: a NaN dilution rate compares False against
    # every tier boundary and would otherwise fall into the most favourable tier, and
    # json.dumps would write it out as a bare NaN literal.
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _dedupe_refs(values: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        token = str(value or "").strip()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


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


def _collect_derived_from(*payloads: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        refs.extend([str(ref) for ref in (payload.get("derived_from") or []) if str(ref).strip()])
    return _dedupe_refs(refs)


def _build_value_capture_summary(
    *,
    discipline_class: str,
    high_dilution: bool,
    moderate_dilution: bool,
    low_dilution: bool,
    buybacks: bool,
    quality_present: bool,
    quality_high: bool,
    quality_unknown: bool,
    facts_ok: bool,
    shares_ok: bool,
) -> str:
    if not facts_ok:
        return "blocked by missing facts — cannot assess per-share capital allocation discipline"
    if not shares_ok:
        return "blocked by missing shares data — cannot assess share count discipline"
    if discipline_class == OWNER_FRIENDLY_DISCIPLINED:
        parts: list[str] = []
        if buybacks:
            parts.append("active buybacks (share count declining)")
        elif low_dilution:
            parts.append("low dilution (share count stable)")
        if quality_high:
            parts.append("strong owner economics quality")
        elif quality_present:
            parts.append("adequate owner economics quality")
        return "owner-friendly: " + " + ".join(parts) if parts else "owner-friendly: disciplined share count with quality support"
    if discipline_class == MIXED_CAPITAL_ALLOCATION:
        if high_dilution and quality_present:
            return "mixed: dilution present but partially offset by owner economics quality"
        if moderate_dilution and quality_present:
            return "mixed: moderate dilution with adequate quality — per-share capture is partial"
        if low_dilution and quality_unknown:
            return "mixed: low dilution but quality evidence insufficient to confirm per-share capture"
        return "mixed: capital allocation signals partially supportive, partially offsetting"
    if discipline_class == OWNER_DILUTIVE_OR_DESTRUCTIVE:
        if high_dilution:
            return "destructive: high dilution — owner value is being systematically leaked to new shareholders"
        return "destructive: very weak owner economics with no offsetting share count discipline"
    # UNKNOWN
    if not facts_ok or not shares_ok:
        return "unknown — evidence blocked"
    return "unknown — insufficient evidence to assess per-share capital allocation discipline"


def compute_capital_allocation_discipline(
    ticker: str,
    as_of_date: str,
    *,
    owner_quality_payload: dict[str, Any] | None = None,
    intrinsic_payload: dict[str, Any] | None = None,
    fcf_payload: dict[str, Any] | None = None,
    shares_payload: dict[str, Any] | None = None,
    evidence_sufficiency_payload: dict[str, Any] | None = None,
    valuation_confidence_payload: dict[str, Any] | None = None,
    price_status: Any = UNKNOWN,
    facts_status: Any = UNKNOWN,
    shares_status: Any = UNKNOWN,
    cfg: AppConfig | None = None,
) -> dict[str, Any]:
    """
    Compute per-share capital allocation discipline for a single ticker.

    Conservative underwriting: this layer asks whether management is converting
    business economics into per-share owner value, or leaking value through dilution.
    UNKNOWN remains UNKNOWN. Evidence gaps are not treated as disciplined.
    """
    cfg = cfg or get_config()
    del cfg

    ticker_norm = str(ticker or "").strip().upper()
    oe_quality = owner_quality_payload if isinstance(owner_quality_payload, dict) else {}
    intrinsic = intrinsic_payload if isinstance(intrinsic_payload, dict) else {}
    fcf = fcf_payload if isinstance(fcf_payload, dict) else {}
    shares = shares_payload if isinstance(shares_payload, dict) else {}
    evs = evidence_sufficiency_payload if isinstance(evidence_sufficiency_payload, dict) else {}
    conf = valuation_confidence_payload if isinstance(valuation_confidence_payload, dict) else {}

    # ── Evidence availability ──────────────────────────────────────────────────
    facts_ok = str(facts_status or "").upper() == "OK"
    shares_ok = str(shares_status or "").upper() == "OK"

    # ── Extract dilution data from owner_quality_payload ──────────────────────
    # The owner_earnings_quality module produces dilution_rate_shares_cagr
    # as a raw fraction (e.g., 0.03 = 3% CAGR). capital_allocation_reason_codes
    # and oe_quality_total are also available.
    dilution_rate_raw = oe_quality.get("dilution_rate_shares_cagr", UNKNOWN)
    oe_quality_total = oe_quality.get("oe_quality_total", UNKNOWN)
    capital_allocation_score = oe_quality.get("capital_allocation_score", UNKNOWN)
    owner_earnings_stability_score = oe_quality.get("owner_earnings_stability_score", UNKNOWN)
    capital_allocation_reason_codes = set(
        str(code)
        for code in (oe_quality.get("capital_allocation_reason_codes") or [])
        if str(code).strip()
    )
    oe_quality_reason_codes = set(
        str(code)
        for code in (oe_quality.get("oe_quality_reason_codes") or [])
        if str(code).strip()
    )

    # Convert dilution_rate from fraction to pct if numeric
    dilution_rate_pct: float | str = UNKNOWN
    if _is_num(dilution_rate_raw):
        dilution_rate_pct = float(dilution_rate_raw) * 100.0

    # Also check shares_payload for share_count_cagr
    share_count_cagr_raw = shares.get("share_count_cagr", shares.get("shares_cagr", UNKNOWN))
    share_count_cagr_pct: float | str = UNKNOWN
    if _is_num(share_count_cagr_raw):
        share_count_cagr_pct = float(share_count_cagr_raw) * 100.0
    elif _is_num(dilution_rate_pct):
        share_count_cagr_pct = float(dilution_rate_pct)

    # Determine effective dilution for classification
    effective_dilution_pct: float | str = UNKNOWN
    if _is_num(dilution_rate_pct):
        effective_dilution_pct = float(dilution_rate_pct)
    elif _is_num(share_count_cagr_pct):
        effective_dilution_pct = float(share_count_cagr_pct)

    # ── Dilution tier classification ───────────────────────────────────────────
    high_dilution = False
    moderate_dilution = False
    low_dilution = False
    buybacks = False  # share count declining (negative dilution)

    if _is_num(effective_dilution_pct):
        d = float(effective_dilution_pct)
        if d > 3.0:
            high_dilution = True
        elif d > 1.0:
            moderate_dilution = True
        else:
            low_dilution = True
            if d < 0.0:
                buybacks = True
    # If "EXCESS_DILUTION" in reason codes, treat as high dilution regardless
    if "EXCESS_DILUTION" in capital_allocation_reason_codes:
        high_dilution = True
        moderate_dilution = False
        low_dilution = False
    # If "SHAREHOLDER_FRIENDLY" in reason codes and no high dilution determined
    if "SHAREHOLDER_FRIENDLY" in capital_allocation_reason_codes and not high_dilution:
        low_dilution = True
        moderate_dilution = False

    dilution_known = _is_num(effective_dilution_pct) or bool(capital_allocation_reason_codes)

    # ── Quality assessment ─────────────────────────────────────────────────────
    quality_high = _is_num(oe_quality_total) and float(oe_quality_total) >= 8.0
    quality_present = _is_num(oe_quality_total) and float(oe_quality_total) >= 4.0
    quality_very_low = _is_num(oe_quality_total) and float(oe_quality_total) < 2.0
    quality_unknown = not _is_num(oe_quality_total)

    # Capital allocation score gives additional signal
    cap_alloc_score_high = _is_num(capital_allocation_score) and float(capital_allocation_score) >= 3.0
    cap_alloc_score_ok = _is_num(capital_allocation_score) and float(capital_allocation_score) >= 2.0
    cap_alloc_score_low = _is_num(capital_allocation_score) and float(capital_allocation_score) < 2.0

    # Stability adds signal
    stability_present = _is_num(owner_earnings_stability_score) and float(owner_earnings_stability_score) >= 3.0

    # ── Build support signals ──────────────────────────────────────────────────
    per_share_support_signals: list[str] = []
    if low_dilution:
        per_share_support_signals.append(SIG_LOW_DILUTION)
    if buybacks:
        per_share_support_signals.append(SIG_SHARE_COUNT_DISCIPLINE)
    if cap_alloc_score_high or "SHAREHOLDER_FRIENDLY" in capital_allocation_reason_codes:
        per_share_support_signals.append(SIG_OWNER_VALUE_CAPTURE)
    if quality_present and stability_present:
        per_share_support_signals.append(SIG_OWNER_EARNINGS_PER_SHARE_GROWTH)

    # ── Build headwind signals ─────────────────────────────────────────────────
    per_share_headwind_signals: list[str] = []
    if high_dilution:
        per_share_headwind_signals.append(SIG_HIGH_DILUTION)
        per_share_headwind_signals.append(SIG_SHARE_COUNT_HEADWIND)
    elif moderate_dilution:
        per_share_headwind_signals.append(SIG_SHARE_COUNT_HEADWIND)
    if high_dilution and quality_present:
        per_share_headwind_signals.append(SIG_POSSIBLE_OWNER_VALUE_LEAKAGE)
    if "EXCESS_DILUTION" in capital_allocation_reason_codes:
        per_share_headwind_signals.append(SIG_OE_GROWTH_WITHOUT_PER_SHARE)
    if cap_alloc_score_low:
        per_share_headwind_signals.append(SIG_FCF_PER_SHARE_STAGNATION)

    # ── Classification logic ───────────────────────────────────────────────────
    reason_codes: list[str] = []

    # Priority 1: Evidence blocked
    if not facts_ok:
        discipline_class = CAPITAL_ALLOCATION_UNKNOWN
        reason_codes.append(REASON_EVIDENCE_BLOCKED_MISSING_FACTS)

    elif not shares_ok and not dilution_known:
        discipline_class = CAPITAL_ALLOCATION_UNKNOWN
        reason_codes.append(REASON_EVIDENCE_BLOCKED_MISSING_SHARES)

    # Priority 2: Very low quality regardless of dilution → destructive
    elif quality_very_low and not quality_unknown:
        discipline_class = OWNER_DILUTIVE_OR_DESTRUCTIVE
        reason_codes.append(REASON_VERY_LOW_QUALITY_DESTRUCTIVE)
        if high_dilution:
            reason_codes.append(REASON_HIGH_DILUTION_DOMINANT)

    # Priority 3: High dilution + weak quality → destructive
    elif high_dilution and not quality_present and not quality_unknown:
        discipline_class = OWNER_DILUTIVE_OR_DESTRUCTIVE
        reason_codes.append(REASON_HIGH_DILUTION_DOMINANT)

    # Priority 4: High dilution even with quality → MIXED
    elif high_dilution and quality_present:
        discipline_class = MIXED_CAPITAL_ALLOCATION
        reason_codes.append(REASON_HIGH_DILUTION_OFFSET_BY_QUALITY)

    # Priority 5: High dilution with unknown quality → destructive lean
    elif high_dilution and quality_unknown:
        discipline_class = OWNER_DILUTIVE_OR_DESTRUCTIVE
        reason_codes.append(REASON_HIGH_DILUTION_DOMINANT)

    # Priority 6: Moderate dilution + quality present → MIXED
    elif moderate_dilution and quality_present:
        discipline_class = MIXED_CAPITAL_ALLOCATION
        reason_codes.append(REASON_MODERATE_DILUTION_WITH_QUALITY)

    # Priority 7: Moderate dilution without quality → MIXED (lean destructive)
    elif moderate_dilution:
        discipline_class = MIXED_CAPITAL_ALLOCATION
        reason_codes.append(REASON_MODERATE_DILUTION_WITH_QUALITY)
        if quality_unknown:
            reason_codes.append(REASON_EVIDENCE_INSUFFICIENT_QUALITY)

    # Priority 8: Low dilution (or buybacks) + strong quality → OWNER_FRIENDLY
    elif low_dilution and quality_high:
        discipline_class = OWNER_FRIENDLY_DISCIPLINED
        if buybacks:
            reason_codes.append(REASON_BUYBACKS_QUALITY_CONFIRMED)
        else:
            reason_codes.append(REASON_LOW_DILUTION_STRONG_QUALITY)
        if cap_alloc_score_high or "SHAREHOLDER_FRIENDLY" in capital_allocation_reason_codes:
            reason_codes.append(REASON_SHAREHOLDER_FRIENDLY_ALLOCATION)

    # Priority 9: Low dilution + quality present (not high) → OWNER_FRIENDLY
    elif low_dilution and quality_present:
        discipline_class = OWNER_FRIENDLY_DISCIPLINED
        reason_codes.append(REASON_LOW_DILUTION_STRONG_QUALITY)

    # Priority 10: Low dilution but quality unknown → MIXED (lean friendly)
    elif low_dilution and quality_unknown:
        discipline_class = MIXED_CAPITAL_ALLOCATION
        reason_codes.append(REASON_LOW_DILUTION_QUALITY_UNKNOWN)
        reason_codes.append(REASON_EVIDENCE_INSUFFICIENT_QUALITY)

    # Priority 11: No dilution info + quality unknown → UNKNOWN
    elif quality_unknown and not dilution_known:
        discipline_class = CAPITAL_ALLOCATION_UNKNOWN
        reason_codes.append(REASON_EVIDENCE_INSUFFICIENT_QUALITY)

    # Fallback
    else:
        discipline_class = CAPITAL_ALLOCATION_UNKNOWN
        reason_codes.append(REASON_EVIDENCE_INSUFFICIENT_QUALITY)

    # ── Primary capital allocation caution ────────────────────────────────────
    if discipline_class == OWNER_FRIENDLY_DISCIPLINED:
        primary_capital_allocation_caution = CAUTION_SUPPORTIVE
    elif discipline_class == MIXED_CAPITAL_ALLOCATION:
        primary_capital_allocation_caution = CAUTION_MIXED
    elif discipline_class == OWNER_DILUTIVE_OR_DESTRUCTIVE:
        primary_capital_allocation_caution = CAUTION_HEADWIND
    else:
        primary_capital_allocation_caution = CAUTION_UNCLEAR

    # ── Value capture summary ─────────────────────────────────────────────────
    per_share_value_capture_summary = _build_value_capture_summary(
        discipline_class=discipline_class,
        high_dilution=high_dilution,
        moderate_dilution=moderate_dilution,
        low_dilution=low_dilution,
        buybacks=buybacks,
        quality_present=quality_present,
        quality_high=quality_high,
        quality_unknown=quality_unknown,
        facts_ok=facts_ok,
        shares_ok=shares_ok,
    )

    derived_from = _collect_derived_from(oe_quality, intrinsic, fcf, shares, evs, conf)

    return {
        "ticker": ticker_norm,
        "as_of_date": as_of_date,
        "capital_allocation_discipline_class": discipline_class,
        "capital_allocation_discipline_reason_codes": _dedupe_refs(reason_codes),
        "per_share_support_signals": _dedupe_refs(per_share_support_signals),
        "per_share_headwind_signals": _dedupe_refs(per_share_headwind_signals),
        "primary_capital_allocation_caution": primary_capital_allocation_caution,
        "per_share_value_capture_summary": per_share_value_capture_summary,
        "dilution_rate_annual_pct": dilution_rate_pct,
        "effective_dilution_pct": effective_dilution_pct,
        "oe_quality_total": oe_quality_total if _is_num(oe_quality_total) else UNKNOWN,
        "capital_allocation_score": capital_allocation_score
        if _is_num(capital_allocation_score)
        else UNKNOWN,
        "derived_from": derived_from,
        "generated_at": utc_now_iso(),
    }


def write_capital_allocation_discipline_for_run(
    *,
    run_id: str,
    as_of_date: str,
    tickers: list[str],
    output_path: Path,
    scoreboard_rows: list[dict[str, Any]] | None = None,
    cfg: AppConfig | None = None,
) -> dict[str, Any]:
    """Write capital_allocation_discipline.json artifact for a universe or sector run."""
    cfg = cfg or get_config()
    del cfg

    detail_by_ticker = {
        str(row.get("ticker") or "").upper(): row.get("capital_allocation_discipline_detail")
        for row in (scoreboard_rows or [])
        if isinstance(row, dict) and isinstance(row.get("capital_allocation_discipline_detail"), dict)
    }

    rows: list[dict[str, Any]] = []
    counts_by_class: dict[str, int] = {}
    counts_by_caution: dict[str, int] = {}
    reason_counts: dict[str, int] = {}

    for ticker in sorted({str(t or "").strip().upper() for t in tickers if str(t or "").strip()}):
        detail = detail_by_ticker.get(ticker)
        if not isinstance(detail, dict):
            # Compute on the fly from scoreboard row
            score_row = next(
                (
                    r
                    for r in (scoreboard_rows or [])
                    if isinstance(r, dict) and str(r.get("ticker") or "").strip().upper() == ticker
                ),
                {},
            )
            _oe_detail = score_row.get("owner_earnings_quality_detail")
            _intrinsic_detail = score_row.get("intrinsic_discipline_detail")
            _evs_detail = score_row.get("evidence_sufficiency_detail")
            _conf_detail = score_row.get("valuation_confidence_detail")
            detail = compute_capital_allocation_discipline(
                ticker=ticker,
                as_of_date=as_of_date,
                owner_quality_payload=_oe_detail if isinstance(_oe_detail, dict) else score_row,
                intrinsic_payload=_intrinsic_detail if isinstance(_intrinsic_detail, dict) else score_row,
                evidence_sufficiency_payload=_evs_detail if isinstance(_evs_detail, dict) else score_row,
                valuation_confidence_payload=_conf_detail if isinstance(_conf_detail, dict) else score_row,
                price_status=score_row.get("price_status", UNKNOWN),
                facts_status=score_row.get("facts_status", UNKNOWN),
                shares_status=score_row.get("shares_status", UNKNOWN),
            )

        discipline_class = str(detail.get("capital_allocation_discipline_class") or CAPITAL_ALLOCATION_UNKNOWN)
        caution = str(detail.get("primary_capital_allocation_caution") or CAUTION_UNCLEAR)
        counts_by_class[discipline_class] = counts_by_class.get(discipline_class, 0) + 1
        counts_by_caution[caution] = counts_by_caution.get(caution, 0) + 1
        for code in (detail.get("capital_allocation_discipline_reason_codes") or []):
            token = str(code or "").strip()
            if token:
                reason_counts[token] = reason_counts.get(token, 0) + 1

        rows.append({
            "ticker": ticker,
            "capital_allocation_discipline_class": discipline_class,
            "capital_allocation_discipline_reason_codes": list(detail.get("capital_allocation_discipline_reason_codes") or []),
            "per_share_support_signals": list(detail.get("per_share_support_signals") or []),
            "per_share_headwind_signals": list(detail.get("per_share_headwind_signals") or []),
            "primary_capital_allocation_caution": caution,
            "per_share_value_capture_summary": str(detail.get("per_share_value_capture_summary") or ""),
            "dilution_rate_annual_pct": detail.get("dilution_rate_annual_pct", UNKNOWN),
            "oe_quality_total": detail.get("oe_quality_total", UNKNOWN),
            "derived_from": list(detail.get("derived_from") or []),
        })

    def _class_rows(cls: str) -> list[dict[str, Any]]:
        return [r for r in rows if str(r.get("capital_allocation_discipline_class") or "") == cls]

    def _caution_rows(caution_val: str) -> list[dict[str, Any]]:
        return [r for r in rows if str(r.get("primary_capital_allocation_caution") or "") == caution_val]

    payload = {
        "run_id": run_id,
        "as_of_date": as_of_date,
        "ticker_count": len(rows),
        "counts_by_capital_allocation_discipline_class": dict(
            sorted(counts_by_class.items(), key=lambda kv: (_DISCIPLINE_ORDER.get(kv[0], 99), kv[0]))
        ),
        "counts_by_primary_capital_allocation_caution": dict(
            sorted(counts_by_caution.items(), key=lambda kv: (-kv[1], kv[0]))
        ),
        "top_10_owner_friendly_disciplined": [
            {
                "ticker": str(r.get("ticker") or ""),
                "reason_codes": list(r.get("capital_allocation_discipline_reason_codes") or []),
            }
            for r in _class_rows(OWNER_FRIENDLY_DISCIPLINED)[:10]
        ],
        "top_10_owner_dilutive_or_destructive": [
            {
                "ticker": str(r.get("ticker") or ""),
                "reason_codes": list(r.get("capital_allocation_discipline_reason_codes") or []),
            }
            for r in _class_rows(OWNER_DILUTIVE_OR_DESTRUCTIVE)[:10]
        ],
        "top_10_capital_allocation_headwind": [
            {
                "ticker": str(r.get("ticker") or ""),
                "caution": str(r.get("primary_capital_allocation_caution") or ""),
            }
            for r in _caution_rows(CAUTION_HEADWIND)
        ][:10],
        "most_common_reason_codes": [
            {"reason_code": str(k), "count": int(v)}
            for k, v in sorted(reason_counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
        ][:15],
        "rows": rows,
        "generated_at": utc_now_iso(),
    }
    _json_write(output_path, payload)
    return payload


def _capital_allocation_discipline_path(run_id: str) -> Path | None:
    cfg = get_config()
    candidates = [
        cfg.outputs_dir / "universe" / run_id / "capital_allocation_discipline.json",
        cfg.sectors_dir / run_id / "capital_allocation_discipline.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def open_capital_allocation_discipline(*, run_id: str, top_n: int = 10) -> dict[str, Any]:
    """Open and summarize the capital_allocation_discipline.json artifact for a run."""
    path = _capital_allocation_discipline_path(run_id)
    if path is None or not path.exists():
        return {
            "status": "MISSING",
            "run_id": run_id,
            "capital_allocation_discipline_path": str(path) if path is not None else "",
        }
    payload = _safe_json(path)
    rows = [r for r in (payload.get("rows") or []) if isinstance(r, dict)]

    def _class_rows(cls: str) -> list[dict[str, Any]]:
        return [r for r in rows if str(r.get("capital_allocation_discipline_class") or "") == cls]

    def _caution_rows(caution_val: str) -> list[dict[str, Any]]:
        return [r for r in rows if str(r.get("primary_capital_allocation_caution") or "") == caution_val]

    reason_data = payload.get("most_common_reason_codes") or []

    return {
        "status": "OK",
        "run_id": run_id,
        "ticker_count": int(payload.get("ticker_count") or 0),
        "counts_by_capital_allocation_discipline_class": (
            payload.get("counts_by_capital_allocation_discipline_class")
            if isinstance(payload.get("counts_by_capital_allocation_discipline_class"), dict)
            else {}
        ),
        "counts_by_primary_capital_allocation_caution": (
            payload.get("counts_by_primary_capital_allocation_caution")
            if isinstance(payload.get("counts_by_primary_capital_allocation_caution"), dict)
            else {}
        ),
        "top_owner_friendly_disciplined": [
            {
                "ticker": str(r.get("ticker") or ""),
                "reason_codes": list(r.get("capital_allocation_discipline_reason_codes") or []),
            }
            for r in _class_rows(OWNER_FRIENDLY_DISCIPLINED)[: max(1, int(top_n))]
        ],
        "top_owner_dilutive_or_destructive": [
            {
                "ticker": str(r.get("ticker") or ""),
                "reason_codes": list(r.get("capital_allocation_discipline_reason_codes") or []),
            }
            for r in _class_rows(OWNER_DILUTIVE_OR_DESTRUCTIVE)[: max(1, int(top_n))]
        ],
        "top_capital_allocation_headwind": [
            {
                "ticker": str(r.get("ticker") or ""),
                "caution": str(r.get("primary_capital_allocation_caution") or ""),
            }
            for r in _caution_rows(CAUTION_HEADWIND)[: max(1, int(top_n))]
        ],
        "most_common_reason_codes": [
            {"reason_code": str(item.get("reason_code") or ""), "count": int(item.get("count") or 0)}
            for item in reason_data
            if isinstance(item, dict)
        ][: max(1, int(top_n))],
        "capital_allocation_discipline_path": str(path),
    }
