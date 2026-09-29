"""Exact source and mutable-row authorization for ``ticker_outcomes``.

An authorized run containing a ticker is not proof that it emitted a decision
for that ticker, and it is not proof that a mutable database row still carries
the decision and outcome values originally written.  This module binds both:

* the exact authorized autonomous-sector artifact bytes and the exact
  decision-bearing subrecords for one ticker; and
* every decision/outcome field on the database row through a canonical
  fingerprint refreshed only by controlled writers.

Legacy rows remain inspectable with NULL lineage, but cannot resolve or enter a
current aggregate.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Mapping

from app.autonomous.artifact_financial_audit import (
    PASS,
    authorized_artifact_bytes,
    authorized_run_artifact_binding,
)

SOURCE_DECISION_BINDING_SCHEMA_VERSION = "ticker_outcome_source_decision_v1"
OUTCOME_INTEGRITY_SCHEMA_VERSION = "ticker_outcome_integrity_v1"
OUTCOME_REPORT_BINDING_SCHEMA_VERSION = "ticker_outcome_report_binding_v1"

_GRADE_TO_DECISION = {
    "ACTIONABLE": "BUY",
    "AVOID": "PASS",
    "DATA_INCOMPLETE": "WATCH",
    "WATCHLIST_ONLY": "WATCH",
}
_DECISION_GRADE_ALIASES = {
    "WATCHLIST": "WATCHLIST_ONLY",
    "WATCH": "WATCHLIST_ONLY",
}

# Every mutable decision/outcome field plus the exact source binding.  The
# stored fingerprint itself is intentionally excluded.
OUTCOME_FINGERPRINT_FIELDS = (
    "id",
    "ticker",
    "as_of_date",
    "run_id",
    "discovery_run_id",
    "deep_run_id",
    "decision",
    "conviction",
    "horizon_days",
    "thesis_tags_json",
    "notes",
    "outcome_status",
    "close_date",
    "realized_return_pct",
    "max_drawdown_pct",
    "entry_price",
    "entry_price_source",
    "entry_date",
    "grade",
    "status",
    "benchmark_symbol",
    "cap_category",
    "pipeline_version",
    "candidate_disposition",
    "decision_basis",
    "selection_validation_status",
    "source_sector",
    "benchmark_return_pct",
    "excess_return_pct",
    "buy_price_target",
    "reached_buy_target",
    "source_artifact_path",
    "source_artifact_sha256",
    "source_decision_fingerprint",
    "created_at",
    "updated_at",
)


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    keys = getattr(value, "keys", None)
    if callable(keys):
        return {str(key): value[key] for key in keys()}
    return {}


def _normalized_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _normalized_ticker(value: Any) -> str | None:
    text = _normalized_text(value)
    return text.upper() if text is not None else None


def _canonical_sha256(value: Any) -> str | None:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(encoded).hexdigest()


def _normalized_grade(value: Any) -> str | None:
    grade = str(value or "").strip().upper()
    grade = _DECISION_GRADE_ALIASES.get(grade, grade)
    return grade if grade in _GRADE_TO_DECISION else None


def _matching_rows(value: Any, ticker: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [
        dict(item)
        for item in value
        if isinstance(item, Mapping) and _normalized_ticker(item.get("ticker")) == ticker
    ]


def _v2_emitted_grade(
    payload: Mapping[str, Any],
    ticker: str,
    disposition: Mapping[str, Any],
) -> str | None:
    terminal_state = str(disposition.get("terminal_state") or "").strip().upper()
    raw_verdict = str(disposition.get("underwriting_verdict") or "").strip().upper()
    if terminal_state in {"OUT_OF_SCOPE", "DEFERRED_BY_BOUND"}:
        return None
    if terminal_state == "SCREENED_OUT":
        return "AVOID"
    if terminal_state == "NEEDS_DATA":
        return "DATA_INCOMPLETE"
    if terminal_state == "READY_FOR_UNDERWRITING":
        return "WATCHLIST_ONLY"
    if terminal_state != "UNDERWRITTEN":
        return None
    grade = _normalized_grade(raw_verdict)
    if grade is None:
        return None
    if grade != "ACTIONABLE":
        return grade
    validation = payload.get("selection_validation")
    validation_row = dict(validation) if isinstance(validation, Mapping) else {}
    validated_selection = (
        _normalized_ticker(payload.get("selected_ticker")) == ticker
        and str(payload.get("final_verdict") or "").strip().upper() == "SELECTED"
        and str(payload.get("decision_status") or "").strip().upper() == "COMPLETE"
        and _normalized_ticker(validation_row.get("selected_ticker")) == ticker
        and str(validation_row.get("status") or "").strip().upper() == "VALIDATED"
    )
    return "ACTIONABLE" if validated_selection else "WATCHLIST_ONLY"


def _v1_emitted_grade(
    payload: Mapping[str, Any],
    ticker: str,
    rankings: list[dict[str, Any]],
) -> str | None:
    # V1 does not have the v2 contract's constructor-level uniqueness checks.
    # An authorized artifact with duplicate rows must not let list order choose
    # which financial decision enters the outcome ledger.
    if len(rankings) > 1:
        return None

    selected_grade: str | None = None
    if _normalized_ticker(payload.get("selected_ticker")) == ticker:
        selected_grade = {
            "SELECTED": "ACTIONABLE",
            "WATCHLIST": "WATCHLIST_ONLY",
            "DATA_INCOMPLETE": "DATA_INCOMPLETE",
        }.get(str(payload.get("final_verdict") or "").strip().upper())
    ranking_grade: str | None = None
    if rankings:
        observed_grades = {
            grade
            for key in ("company_autonomy_verdict", "final_verdict", "verdict")
            if (grade := _normalized_grade(rankings[0].get(key))) is not None
        }
        if len(observed_grades) > 1:
            return None
        ranking_grade = next(iter(observed_grades), None)

    if selected_grade is not None and ranking_grade is not None and selected_grade != ranking_grade:
        return None
    return selected_grade or ranking_grade


def _emitted_decision_payload(
    payload: Mapping[str, Any],
    ticker: str,
) -> tuple[dict[str, Any], str, str] | None:
    """Return exact source subrecords plus expected grade/decision.

    Mere packet, loaded-set, or survivor membership is intentionally
    insufficient.  At least one decision-bearing selected/ranking/disposition
    record must emit a recognized grade for this exact ticker.
    """

    rankings = _matching_rows(payload.get("relative_ranking"), ticker)
    dispositions = _matching_rows(payload.get("candidate_dispositions"), ticker)
    pipeline_version = str(payload.get("pipeline_version") or "v1").strip().lower()
    if pipeline_version == "v2":
        grade = (
            _v2_emitted_grade(payload, ticker, dispositions[0]) if len(dispositions) == 1 else None
        )
    else:
        grade = _v1_emitted_grade(payload, ticker, rankings)
    if grade is None:
        return None
    decision = _GRADE_TO_DECISION[grade]

    selected_ticker = _normalized_ticker(payload.get("selected_ticker"))
    selected = selected_ticker == ticker
    selection_validation = payload.get("selection_validation")
    validation_row = (
        dict(selection_validation)
        if isinstance(selection_validation, Mapping)
        and _normalized_ticker(selection_validation.get("selected_ticker")) == ticker
        else None
    )
    final_decision = payload.get("final_decision")
    final_decision_row = (
        dict(final_decision) if selected and isinstance(final_decision, Mapping) else None
    )
    memo_body = payload.get("memo_body")
    memo_candidates = (
        memo_body.get("candidates")
        if isinstance(memo_body, Mapping) and isinstance(memo_body.get("candidates"), Mapping)
        else {}
    )
    memo_candidate = memo_candidates.get(ticker) or memo_candidates.get(ticker.lower())
    company_packets = _matching_rows(payload.get("company_packets"), ticker)
    source_payload = {
        "schema_version": SOURCE_DECISION_BINDING_SCHEMA_VERSION,
        "run_id": _normalized_text(payload.get("run_id")),
        "ticker": ticker,
        "pipeline_version": pipeline_version,
        "expected_grade": grade,
        "expected_decision": decision,
        "selected_decision": (
            {
                "selected_ticker": selected_ticker,
                "final_verdict": payload.get("final_verdict"),
                "decision_status": payload.get("decision_status"),
                "confidence": payload.get("confidence"),
                "final_decision": final_decision_row,
                "selection_validation": validation_row,
            }
            if selected
            else None
        ),
        "relative_ranking_rows": rankings,
        "candidate_dispositions": dispositions,
        "candidate_selection": (
            dict(payload["candidate_selection"])
            if isinstance(payload.get("candidate_selection"), Mapping)
            else {}
        ),
        "memo_candidate": dict(memo_candidate) if isinstance(memo_candidate, Mapping) else None,
        "company_packets": company_packets,
    }
    return source_payload, grade, decision


def emitted_decision_claim(
    payload: Mapping[str, Any] | Any,
    ticker: str | None,
) -> dict[str, str] | None:
    """Return the exact decision claim emitted for ``ticker`` by ``payload``."""

    normalized_ticker = _normalized_ticker(ticker)
    values = _mapping(payload)
    if normalized_ticker is None or not values:
        return None
    emitted = _emitted_decision_payload(values, normalized_ticker)
    if emitted is None:
        return None
    source_payload, expected_grade, expected_decision = emitted
    source_decision_fingerprint = _canonical_sha256(source_payload)
    if source_decision_fingerprint is None:
        return None
    return {
        "ticker": normalized_ticker,
        "source_decision_fingerprint": source_decision_fingerprint,
        "expected_grade": expected_grade,
        "expected_decision": expected_decision,
    }


def authorized_emitted_decision_binding(
    run_id: str | None,
    ticker: str | None,
    manifest_path: str | Path | None = None,
) -> dict[str, str] | None:
    """Bind one exact emitted ticker decision to current authorized bytes."""

    normalized_ticker = _normalized_ticker(ticker)
    binding = authorized_run_artifact_binding(run_id, manifest_path)
    if normalized_ticker is None or binding is None:
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
    if not isinstance(payload, Mapping):
        return None
    if _normalized_text(payload.get("run_id")) != binding["source_run_id"]:
        return None
    claim = emitted_decision_claim(payload, normalized_ticker)
    if claim is None:
        return None
    return {
        **binding,
        **claim,
    }


def outcome_integrity_fingerprint(row: Mapping[str, Any] | Any) -> str | None:
    values = _mapping(row)
    if any(field not in values for field in OUTCOME_FINGERPRINT_FIELDS):
        return None
    return _canonical_sha256(
        {
            "schema_version": OUTCOME_INTEGRITY_SCHEMA_VERSION,
            "row": {field: values.get(field) for field in OUTCOME_FINGERPRINT_FIELDS},
        }
    )


def clear_outcome_lineage(conn: sqlite3.Connection, outcome_id: int) -> None:
    conn.execute(
        """
        UPDATE ticker_outcomes
        SET source_artifact_path = NULL,
            source_artifact_sha256 = NULL,
            source_decision_fingerprint = NULL,
            financial_integrity_fingerprint = NULL
        WHERE id = ?
        """,
        (int(outcome_id),),
    )


def bind_outcome_row(conn: sqlite3.Connection, outcome_id: int) -> bool:
    """Bind a freshly selected/written row to its exact emitted source."""

    row = conn.execute("SELECT * FROM ticker_outcomes WHERE id = ?", (int(outcome_id),)).fetchone()
    if row is None:
        return False
    values = _mapping(row)
    binding = authorized_emitted_decision_binding(values.get("run_id"), values.get("ticker"))
    if (
        binding is None
        or _normalized_grade(values.get("grade")) != binding["expected_grade"]
        or str(values.get("decision") or "").strip().upper() != binding["expected_decision"]
    ):
        return False
    values.update(
        {
            "source_artifact_path": binding["source_artifact_path"],
            "source_artifact_sha256": binding["source_artifact_sha256"],
            "source_decision_fingerprint": binding["source_decision_fingerprint"],
        }
    )
    fingerprint = outcome_integrity_fingerprint(values)
    if fingerprint is None:
        return False
    conn.execute(
        """
        UPDATE ticker_outcomes
        SET source_artifact_path = ?,
            source_artifact_sha256 = ?,
            source_decision_fingerprint = ?,
            financial_integrity_fingerprint = ?
        WHERE id = ?
        """,
        (
            binding["source_artifact_path"],
            binding["source_artifact_sha256"],
            binding["source_decision_fingerprint"],
            fingerprint,
            int(outcome_id),
        ),
    )
    return True


def refresh_outcome_integrity_fingerprint(conn: sqlite3.Connection, outcome_id: int) -> bool:
    """Refresh a controlled mutation only while its exact source remains valid."""

    row = conn.execute("SELECT * FROM ticker_outcomes WHERE id = ?", (int(outcome_id),)).fetchone()
    if row is None:
        return False
    values = _mapping(row)
    binding = authorized_emitted_decision_binding(values.get("run_id"), values.get("ticker"))
    if (
        binding is None
        or values.get("source_artifact_path") != binding["source_artifact_path"]
        or values.get("source_artifact_sha256") != binding["source_artifact_sha256"]
        or values.get("source_decision_fingerprint") != binding["source_decision_fingerprint"]
        or _normalized_grade(values.get("grade")) != binding["expected_grade"]
        or str(values.get("decision") or "").strip().upper() != binding["expected_decision"]
    ):
        return False
    fingerprint = outcome_integrity_fingerprint(values)
    if fingerprint is None:
        return False
    conn.execute(
        "UPDATE ticker_outcomes SET financial_integrity_fingerprint = ? WHERE id = ?",
        (fingerprint, int(outcome_id)),
    )
    return True


def outcome_row_is_decision_eligible(
    row: Mapping[str, Any] | Any,
    manifest_path: str | Path | None = None,
) -> bool:
    values = _mapping(row)
    stored_fingerprint = _normalized_text(values.get("financial_integrity_fingerprint"))
    source_decision_fingerprint = _normalized_text(values.get("source_decision_fingerprint"))
    source_path = _normalized_text(values.get("source_artifact_path"))
    source_sha256 = _normalized_text(values.get("source_artifact_sha256"))
    if None in {
        stored_fingerprint,
        source_decision_fingerprint,
        source_path,
        source_sha256,
    }:
        return False
    binding = authorized_emitted_decision_binding(
        values.get("run_id"),
        values.get("ticker"),
        manifest_path,
    )
    if (
        binding is None
        or source_path != binding["source_artifact_path"]
        or source_sha256 != binding["source_artifact_sha256"]
        or source_decision_fingerprint != binding["source_decision_fingerprint"]
        or _normalized_grade(values.get("grade")) != binding["expected_grade"]
        or str(values.get("decision") or "").strip().upper() != binding["expected_decision"]
    ):
        return False
    return outcome_integrity_fingerprint(values) == stored_fingerprint


def serialize_outcome_binding(row: Mapping[str, Any] | Any) -> dict[str, Any] | None:
    values = _mapping(row)
    if not outcome_row_is_decision_eligible(values):
        return None
    return {
        "schema_version": OUTCOME_REPORT_BINDING_SCHEMA_VERSION,
        "outcome_state": {field: values.get(field) for field in OUTCOME_FINGERPRINT_FIELDS},
        "financial_integrity_fingerprint": values["financial_integrity_fingerprint"],
    }


def serialized_outcome_binding_is_decision_eligible(
    payload: Mapping[str, Any] | Any,
    manifest_path: str | Path | None = None,
) -> bool:
    values = _mapping(payload)
    state = values.get("outcome_state")
    if values.get("schema_version") != OUTCOME_REPORT_BINDING_SCHEMA_VERSION or not isinstance(
        state, Mapping
    ):
        return False
    row = dict(state)
    row["financial_integrity_fingerprint"] = values.get("financial_integrity_fingerprint")
    return outcome_row_is_decision_eligible(row, manifest_path)


__all__ = [
    "OUTCOME_FINGERPRINT_FIELDS",
    "OUTCOME_INTEGRITY_SCHEMA_VERSION",
    "OUTCOME_REPORT_BINDING_SCHEMA_VERSION",
    "SOURCE_DECISION_BINDING_SCHEMA_VERSION",
    "authorized_emitted_decision_binding",
    "bind_outcome_row",
    "clear_outcome_lineage",
    "emitted_decision_claim",
    "outcome_integrity_fingerprint",
    "outcome_row_is_decision_eligible",
    "refresh_outcome_integrity_fingerprint",
    "serialize_outcome_binding",
    "serialized_outcome_binding_is_decision_eligible",
]
