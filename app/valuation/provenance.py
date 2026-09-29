"""Point-in-time provenance validation for v2 valuation scorecards."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, is_dataclass
from datetime import date
from typing import Any, Sequence

from app.util.financial_data_access import normalize_cik
from app.valuation.lineage import latest_decision_eligible_valuation_row
from app.valuation.valuation_writer import valuation_facts_fingerprint


V2_VALUATION_STALENESS_DAYS = 45


def _as_plain_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if is_dataclass(value) and not isinstance(value, type):
        payload = asdict(value)
        return dict(payload) if isinstance(payload, dict) else {}
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        return dict(payload) if isinstance(payload, dict) else {}
    return {}


def scorecard_is_stale(
    scorecard_as_of_date: str | None,
    as_of_date: str,
    *,
    max_age_days: int = V2_VALUATION_STALENESS_DAYS,
) -> bool:
    """Return whether a scorecard falls outside the v2 freshness window."""

    if not scorecard_as_of_date:
        return True
    try:
        run_date = date.fromisoformat(str(as_of_date)[:10])
        scorecard_date = date.fromisoformat(str(scorecard_as_of_date)[:10])
    except ValueError:
        return True
    return (run_date - scorecard_date).days > max_age_days


def validate_v2_valuation_method_provenance(
    conn: Any,
    ticker: str,
    *,
    method: str,
    as_of_date: str,
    issuer_cik: str | None,
    issuer_aliases: Sequence[str] = (),
    price_snapshot: Any | None,
) -> dict[str, Any]:
    """Load and validate one exact v2 valuation method at ``as_of_date``.

    The returned outputs are intentionally empty unless every identity,
    point-in-time facts, and price binding matches.  Callers therefore cannot
    accidentally rank or prompt on a legacy/stale scorecard while separately
    reporting the candidate as data-incomplete.
    """

    normalized_method = str(method or "").strip().lower()
    if not normalized_method:
        raise ValueError("valuation method is required")
    candidate_params = (
        str(ticker).upper(),
        normalized_method,
        str(as_of_date)[:10],
    )
    try:
        candidate = conn.execute(
            """
            SELECT id, as_of_date
            FROM valuations
            WHERE ticker = ? AND method = ? AND as_of_date <= ?
            ORDER BY as_of_date DESC, created_at DESC, id DESC
            LIMIT 1
            """,
            candidate_params,
        ).fetchone()
    except sqlite3.DatabaseError:
        # Readiness probes may inspect a legacy schema without ``created_at``.
        # That schema cannot authorize a row, but an empty table is still the
        # ordinary MISSING state rather than an exception.
        candidate = conn.execute(
            """
            SELECT id, as_of_date
            FROM valuations
            WHERE ticker = ? AND method = ? AND as_of_date <= ?
            ORDER BY as_of_date DESC, id DESC
            LIMIT 1
            """,
            candidate_params,
        ).fetchone()
    if candidate is None:
        return {
            "row_id": None,
            "raw_asof": None,
            "validated_asof": None,
            "mismatch_reasons": [f"V2_{normalized_method.upper()}_MISSING"],
            "inputs": {},
            "outputs": {},
        }

    raw_asof = str(candidate["as_of_date"])
    try:
        row = latest_decision_eligible_valuation_row(
            conn,
            ticker=str(ticker).upper(),
            method=normalized_method,
            as_of_date=str(as_of_date)[:10],
        )
    except Exception:  # noqa: BLE001 - legacy/malformed schemas fail closed
        row = None
    if row is None or int(row["id"]) != int(candidate["id"]):
        return {
            "row_id": int(candidate["id"]),
            "raw_asof": raw_asof,
            "validated_asof": None,
            "mismatch_reasons": ["VALUATION_EXACT_SOURCE_UNAUTHORIZED"],
            "inputs": {},
            "outputs": {},
        }

    try:
        inputs = json.loads(str(row["inputs_json"] or "{}"))
    except (TypeError, ValueError, json.JSONDecodeError):
        inputs = {}
    if not isinstance(inputs, dict):
        inputs = {}
    try:
        raw_outputs = json.loads(str(row["outputs_json"] or "{}"))
    except (TypeError, ValueError, json.JSONDecodeError):
        raw_outputs = {}
    if not isinstance(raw_outputs, dict):
        raw_outputs = {}

    reasons: list[str] = []
    if str(inputs.get("pipeline_version") or "").lower() != "v2":
        reasons.append("VALUATION_PIPELINE_VERSION_MISMATCH")
    if not bool(inputs.get("require_filed_asof")):
        reasons.append("VALUATION_FACTS_NOT_FILED_ASOF")

    expected_cik = normalize_cik(issuer_cik)
    if expected_cik is None or normalize_cik(inputs.get("issuer_cik")) != expected_cik:
        reasons.append("VALUATION_ISSUER_MISMATCH")

    snapshot = _as_plain_dict(price_snapshot)
    expected_price = snapshot.get("price")
    stored_price = inputs.get("market_price")
    if (
        isinstance(expected_price, bool)
        or not isinstance(expected_price, (int, float))
        or isinstance(stored_price, bool)
        or not isinstance(stored_price, (int, float))
        or abs(float(stored_price) - float(expected_price)) > 1e-9
    ):
        reasons.append("VALUATION_PRICE_MISMATCH")
    if str(inputs.get("price_currency") or "").upper() != "USD":
        reasons.append("VALUATION_PRICE_CURRENCY_MISMATCH")
    if str(snapshot.get("currency") or "").upper() != "USD":
        reasons.append("VALUATION_PACKET_PRICE_CURRENCY_MISMATCH")
    if str(inputs.get("price_as_of_date") or "") != str(snapshot.get("as_of_date") or ""):
        reasons.append("VALUATION_PRICE_ASOF_MISMATCH")

    try:
        expected_facts_fingerprint = valuation_facts_fingerprint(
            ticker,
            conn,
            as_of_date=str(as_of_date)[:10],
            issuer_cik=expected_cik,
            issuer_aliases=tuple(str(item) for item in issuer_aliases),
        )
    except Exception as exc:  # noqa: BLE001 - becomes an explicit mismatch
        expected_facts_fingerprint = None
        reasons.append(f"VALUATION_FACTS_FINGERPRINT_ERROR:{type(exc).__name__}")
    if (
        expected_facts_fingerprint is None
        or str(inputs.get("facts_fingerprint") or "") != expected_facts_fingerprint
    ):
        reasons.append("VALUATION_FACTS_REVISION_MISMATCH")

    if scorecard_is_stale(raw_asof, str(as_of_date)[:10]):
        reasons.append("VALUATION_SCORECARD_STALE")
    mismatches = list(dict.fromkeys(reasons))
    return {
        "row_id": int(row["id"]),
        "raw_asof": raw_asof,
        "validated_asof": None if mismatches else raw_asof,
        "mismatch_reasons": mismatches,
        "inputs": inputs,
        "outputs": {} if mismatches else raw_outputs,
    }


def validate_v2_scorecard_provenance(
    conn: Any,
    ticker: str,
    *,
    as_of_date: str,
    issuer_cik: str | None,
    issuer_aliases: Sequence[str] = (),
    price_snapshot: Any | None,
) -> dict[str, Any]:
    """Validate the v2 scorecard row used for anchors and quality context."""

    return validate_v2_valuation_method_provenance(
        conn,
        ticker,
        method="scorecard",
        as_of_date=as_of_date,
        issuer_cik=issuer_cik,
        issuer_aliases=issuer_aliases,
        price_snapshot=price_snapshot,
    )


__all__ = [
    "V2_VALUATION_STALENESS_DAYS",
    "scorecard_is_stale",
    "validate_v2_valuation_method_provenance",
    "validate_v2_scorecard_provenance",
]
