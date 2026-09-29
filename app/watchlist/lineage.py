"""Exact emitted-decision authorization for mutable watchlist rows.

Run/ticker membership and even a matching grade are insufficient provenance
for a mutable watchlist row.  The source artifact also emits the confidence,
narrative, valuation anchor/target, and other decision claims that the row
later presents.  This module deterministically reconstructs those claims from
the exact authorized artifact bytes and requires the current row to match.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from app.autonomous.artifact_financial_audit import PASS, authorized_artifact_bytes
from app.outcomes.lineage import (
    authorized_emitted_decision_binding,
    emitted_decision_claim,
)


_GRADE_ALIASES = {
    "WATCH": "WATCHLIST_ONLY",
    "WATCHLIST": "WATCHLIST_ONLY",
}

WATCHLIST_SOURCE_DECISION_FIELDS = (
    "ticker",
    "conviction_grade",
    "confidence",
    "conviction_source",
    "scan_family",
    "valuation_anchor_method",
    "valuation_anchor_value",
    "buy_price_target",
    "current_price_at_addition",
    "thesis_text",
    "key_risks",
    "falsifiers",
    "open_questions",
    "source_run_id",
    "source_sector",
    "market_cap_mm",
    "cap_source",
    "cap_band",
    "cap_asof",
    "pipeline_version",
    "candidate_disposition",
    "decision_basis",
    "selection_validation_status",
)


def _mapping(value: Mapping[str, Any] | Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    keys = getattr(value, "keys", None)
    if callable(keys):
        return {str(key): value[key] for key in keys()}
    return {}


def _optional_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _row_json_list(values: Mapping[str, Any], field: str) -> list[str] | None:
    direct = values.get(field)
    if isinstance(direct, list):
        return [str(item) for item in direct]
    encoded = values.get(f"{field}_json")
    if not isinstance(encoded, str):
        return None
    try:
        parsed = json.loads(encoded)
    except (TypeError, ValueError):
        return None
    return [str(item) for item in parsed] if isinstance(parsed, list) else None


def watchlist_source_decision_state(
    payload: Mapping[str, Any] | Any,
    ticker: str | None,
) -> dict[str, Any] | None:
    """Reconstruct the exact source-derived watchlist state for one ticker."""

    values = _mapping(payload)
    normalized_ticker = str(ticker or "").strip().upper()
    claim = emitted_decision_claim(values, normalized_ticker)
    if not values or not normalized_ticker or claim is None:
        return None

    try:
        from app.autonomous.sector_contract import AutonomousSectorFinancialRunArtifact
        from app.watchlist import store
        from app.watchlist.margin_of_safety import DISCOUNT_CONFIG, compute_buy_target

        artifact = AutonomousSectorFinancialRunArtifact.from_dict(values)
    except (KeyError, TypeError, ValueError):
        return None

    packet_matches = [
        packet
        for packet in artifact.company_packets
        if str(packet.ticker or "").strip().upper() == normalized_ticker
    ]
    ranking_matches = [
        row
        for row in artifact.relative_ranking
        if isinstance(row, dict)
        and str(row.get("ticker") or "").strip().upper() == normalized_ticker
    ]
    if len(packet_matches) != 1 or len(ranking_matches) > 1:
        return None
    packet = packet_matches[0]
    ranking = ranking_matches[0] if ranking_matches else None
    grade = store._candidate_verdict(artifact, normalized_ticker, ranking)
    if grade != claim["expected_grade"]:
        return None
    confidence = store._candidate_confidence(artifact, normalized_ticker, ranking)
    conviction_source = store._candidate_conviction_source(
        artifact,
        normalized_ticker,
        ranking,
    )

    valuation = packet.valuation if isinstance(packet.valuation, dict) else {}
    explicit_target: float | None = None
    for key in ("buy_below_price", "buy_price_target", "buy_below"):
        candidate = _optional_float(valuation.get(key))
        if candidate is not None and candidate > 0:
            explicit_target = candidate
            break
    anchor = store._valuation_anchor(packet)
    if explicit_target is not None:
        buy_price_target = explicit_target
    elif anchor is None:
        buy_price_target = None
    elif DISCOUNT_CONFIG.w_vol != 0.0:
        # Volatility is external mutable state.  If it ever becomes an active
        # input, the source artifact must explicitly serialize that input
        # before a later reader can authorize the derived target.
        return None
    else:
        buy_price_target = compute_buy_target(
            anchor,
            conviction_grade=grade,
            confidence=confidence,
            intrinsic_low=store._intrinsic_bound(packet, "intrinsic_range_low"),
            intrinsic_high=store._intrinsic_bound(packet, "intrinsic_range_high"),
            realized_volatility=None,
        )

    memo_payload = store._memo_candidate_payload(artifact, normalized_ticker)
    thesis_text = str(memo_payload.get("thesis") or "").strip()
    if not thesis_text and ranking:
        thesis_text = str(ranking.get("positioning_summary") or "").strip()
    if (
        not thesis_text
        and artifact.final_decision is not None
        and str(artifact.selected_ticker or "").strip().upper() == normalized_ticker
    ):
        thesis_text = str(artifact.final_decision.thesis or "").strip()
    open_questions = store._strings_from_payload(memo_payload, "open_questions")
    if grade == "DATA_INCOMPLETE" and artifact.final_decision is not None:
        for code in artifact.final_decision.data_resolution_needed:
            normalized_code = str(code).strip()
            if normalized_code and normalized_code not in open_questions:
                open_questions.append(normalized_code)

    pipeline_version = store._artifact_pipeline_version(artifact)
    semantics = None
    if pipeline_version == "v2":
        disposition = store._candidate_disposition_map(artifact).get(normalized_ticker)
        semantics = store._v2_candidate_semantics(
            artifact,
            normalized_ticker,
            disposition,
        )
        if semantics is None or semantics.grade != grade:
            return None
        for code in list(store._object_value(disposition, "reason_codes", []) or []):
            normalized_code = str(code).strip()
            if normalized_code and normalized_code not in open_questions:
                open_questions.append(normalized_code)

    packet_cap_source = str(getattr(packet, "market_cap_source", "") or "").strip()
    if packet_cap_source:
        market_cap_mm = _optional_float(getattr(packet, "market_cap_mm", None))
        cap_source = packet_cap_source.lower()
        raw_cap_band = str(getattr(packet, "market_cap_category", "") or "").strip()
        cap_band = raw_cap_band.lower() or None
        cap_asof = str(artifact.as_of_date or "").strip() or None
    else:
        # The legacy fallback consults mutable database state.  Such values are
        # not source-derived and therefore cannot be authorized by this run.
        market_cap_mm = None
        cap_source = None
        cap_band = None
        cap_asof = None

    return {
        "ticker": normalized_ticker,
        "conviction_grade": grade,
        "confidence": str(confidence).upper() if confidence else None,
        "conviction_source": str(conviction_source).lower() if conviction_source else None,
        "scan_family": str(artifact.scan_family or "normal").lower(),
        "valuation_anchor_method": store._anchor_method(packet),
        "valuation_anchor_value": anchor,
        "buy_price_target": buy_price_target,
        "current_price_at_addition": _optional_float(packet.current_price),
        "thesis_text": thesis_text or None,
        "key_risks": store._strings_from_payload(memo_payload, "key_risks"),
        "falsifiers": store._strings_from_payload(memo_payload, "falsifiers"),
        "open_questions": open_questions,
        "source_run_id": str(artifact.run_id),
        "source_sector": artifact.sector,
        "market_cap_mm": market_cap_mm,
        "cap_source": cap_source,
        "cap_band": cap_band,
        "cap_asof": cap_asof,
        "pipeline_version": "v2" if pipeline_version == "v2" else None,
        "candidate_disposition": semantics.terminal_state if semantics is not None else None,
        "decision_basis": semantics.decision_basis if semantics is not None else None,
        "selection_validation_status": (
            semantics.selection_validation_status if semantics is not None else None
        ),
    }


def watchlist_row_source_decision_state(
    row: Mapping[str, Any] | Any,
    *,
    source_run_field: str = "source_run_id",
    grade_field: str = "conviction_grade",
    ticker: str | None = None,
) -> dict[str, Any] | None:
    """Project one database row onto the exact source-derived field contract."""

    values = _mapping(row)
    key_risks = _row_json_list(values, "key_risks")
    falsifiers = _row_json_list(values, "falsifiers")
    open_questions = _row_json_list(values, "open_questions")
    if key_risks is None or falsifiers is None or open_questions is None:
        return None
    normalized_grade = str(values.get(grade_field) or "").strip().upper()
    normalized_grade = _GRADE_ALIASES.get(normalized_grade, normalized_grade)
    projection = {
        "ticker": str(ticker or values.get("ticker") or "").strip().upper(),
        "conviction_grade": normalized_grade or None,
        "confidence": str(values.get("confidence") or "").strip().upper() or None,
        "conviction_source": (str(values.get("conviction_source") or "").strip().lower() or None),
        "scan_family": str(values.get("scan_family") or "normal").strip().lower(),
        "valuation_anchor_method": values.get("valuation_anchor_method"),
        "valuation_anchor_value": _optional_float(values.get("valuation_anchor_value")),
        "buy_price_target": _optional_float(values.get("buy_price_target")),
        "current_price_at_addition": _optional_float(values.get("current_price_at_addition")),
        "thesis_text": values.get("thesis_text"),
        "key_risks": key_risks,
        "falsifiers": falsifiers,
        "open_questions": open_questions,
        "source_run_id": str(values.get(source_run_field) or "").strip(),
        "source_sector": values.get("source_sector"),
        "market_cap_mm": _optional_float(values.get("market_cap_mm")),
        "cap_source": str(values.get("cap_source") or "").strip().lower() or None,
        "cap_band": str(values.get("cap_band") or "").strip().lower() or None,
        "cap_asof": values.get("cap_asof"),
        "pipeline_version": (str(values.get("pipeline_version") or "").strip().lower() or None),
        "candidate_disposition": (
            str(values.get("candidate_disposition") or "").strip().upper() or None
        ),
        "decision_basis": (str(values.get("decision_basis") or "").strip().upper() or None),
        "selection_validation_status": (
            str(values.get("selection_validation_status") or "").strip().upper() or None
        ),
    }
    if not projection["ticker"] or not projection["source_run_id"]:
        return None
    return projection


def watchlist_row_matches_source_decision(
    row: Mapping[str, Any] | Any,
    payload: Mapping[str, Any] | Any,
    *,
    source_run_field: str = "source_run_id",
    grade_field: str = "conviction_grade",
    ticker: str | None = None,
) -> bool:
    expected = watchlist_source_decision_state(payload, ticker or _mapping(row).get("ticker"))
    observed = watchlist_row_source_decision_state(
        row,
        source_run_field=source_run_field,
        grade_field=grade_field,
        ticker=ticker,
    )
    return expected is not None and observed == expected


def authorized_watchlist_decision_binding(
    row: Mapping[str, Any] | Any,
    manifest_path: str | Path | None = None,
    *,
    source_run_field: str = "source_run_id",
    grade_field: str = "conviction_grade",
    ticker: str | None = None,
) -> dict[str, Any] | None:
    """Authorize a row only when it matches the exact emitted decision state.

    Merely finding the ticker in a packet, loaded set, or candidate list is not
    decision provenance.  The source artifact must contain one unambiguous
    selected/ranking/disposition decision for the ticker, and every
    source-derived field on the mutable row must equal that exact source.
    """

    values = _mapping(row)
    normalized_ticker = str(ticker or values.get("ticker") or "").strip().upper()
    normalized_grade = str(values.get(grade_field) or "").strip().upper()
    normalized_grade = _GRADE_ALIASES.get(normalized_grade, normalized_grade)
    if not normalized_ticker or not normalized_grade:
        return None
    binding = authorized_emitted_decision_binding(
        values.get(source_run_field),
        normalized_ticker,
        manifest_path,
    )
    if binding is None or binding.get("expected_grade") != normalized_grade:
        return None
    status, artifact_bytes = authorized_artifact_bytes(
        binding["source_artifact_path"],
        manifest_path,
    )
    if (
        status != PASS
        or artifact_bytes is None
        or hashlib.sha256(artifact_bytes).hexdigest() != binding["source_artifact_sha256"]
    ):
        return None
    try:
        payload = json.loads(artifact_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping) or not watchlist_row_matches_source_decision(
        values,
        payload,
        source_run_field=source_run_field,
        grade_field=grade_field,
        ticker=normalized_ticker,
    ):
        return None
    return binding


def watchlist_row_is_decision_eligible(
    row: Mapping[str, Any] | Any,
    manifest_path: str | Path | None = None,
    *,
    source_run_field: str = "source_run_id",
    grade_field: str = "conviction_grade",
    ticker: str | None = None,
) -> bool:
    return (
        authorized_watchlist_decision_binding(
            row,
            manifest_path,
            source_run_field=source_run_field,
            grade_field=grade_field,
            ticker=ticker,
        )
        is not None
    )


__all__ = [
    "WATCHLIST_SOURCE_DECISION_FIELDS",
    "authorized_watchlist_decision_binding",
    "watchlist_row_matches_source_decision",
    "watchlist_row_is_decision_eligible",
    "watchlist_row_source_decision_state",
    "watchlist_source_decision_state",
]
