"""Exact-source authorization for decision-bearing valuation rows.

The mutable ``valuations`` tables are not themselves an authorization
boundary.  A row is current-product eligible only when it names one audited
source artifact, the artifact's current bytes match the stored SHA-256, the
artifact identifies the same run and ticker, and the row fingerprint still
matches every decision-bearing database field.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.autonomous.artifact_financial_audit import (
    PASS,
    authorized_artifact_bytes,
    authorized_run_artifact_binding,
)
from app.util.financial_data_access import normalize_cik

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

LINEAGE_COLUMNS = (
    "source_run_id",
    "source_artifact_path",
    "source_artifact_sha256",
    "financial_integrity_fingerprint",
)
VALUATION_SOURCE_RECORD_SCHEMA_VERSION = "valuation_source_record_v1"

_FINGERPRINT_FIELDS = (
    "ticker",
    "as_of_date",
    "method",
    "inputs_json",
    "outputs_json",
    "warnings_json",
    "created_at",
    "valuation_writer_version",
    "quality_gate_verdict",
    "confidence_class",
    "gate_reason_codes",
    "valuation_headwinds",
    "valuation_supports",
    "source_run_id",
    "source_artifact_path",
    "source_artifact_sha256",
)
_VALUATION_SOURCE_RECORD_FIELDS = tuple(
    field
    for field in _FINGERPRINT_FIELDS
    if field not in {"source_artifact_path", "source_artifact_sha256"}
)
_WRITER_RECEIPT_LOCK = threading.RLock()
_WRITER_RECEIPTS_BY_RUN: dict[str, dict[str, dict[str, Any]]] = {}


# Every selector in this module reads the complete row fingerprint contract.
# Selecting only ``outputs_json`` plus the four obvious lineage fields is not
# sufficient: an attacker (or stale writer) could change a decision-bearing
# field that participates in ``valuation_integrity_fingerprint`` while leaving
# the visible payload and source pointer untouched.
DECISION_ELIGIBLE_VALUATION_COLUMNS = (
    "id",
    *_FINGERPRINT_FIELDS,
    "financial_integrity_fingerprint",
)


def _row_mapping(row: Mapping[str, Any] | Any) -> dict[str, Any]:
    if isinstance(row, Mapping):
        return dict(row)
    keys = getattr(row, "keys", None)
    if callable(keys):
        return {str(key): row[key] for key in keys()}
    return {}


def _normalized_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valuation_integrity_fingerprint(row: Mapping[str, Any] | Any) -> str:
    """Bind one valuation row's decision fields to its exact source identity."""

    values = _row_mapping(row)
    payload = {field: values.get(field) for field in _FINGERPRINT_FIELDS}
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def valuation_source_record(row: Mapping[str, Any] | Any) -> dict[str, Any] | None:
    """Serialize one exact pre-publication method/input/output claim."""

    values = _row_mapping(row)
    if any(field not in values for field in _VALUATION_SOURCE_RECORD_FIELDS):
        return None
    record = {
        "schema_version": VALUATION_SOURCE_RECORD_SCHEMA_VERSION,
        "row": {field: values.get(field) for field in _VALUATION_SOURCE_RECORD_FIELDS},
    }
    try:
        json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError):
        return None
    return record


def _source_record_key(record: Mapping[str, Any]) -> str:
    return json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def register_valuation_writer_records(
    run_id: str | None,
    records: Sequence[Mapping[str, Any]],
) -> None:
    """Register exact in-process records emitted by the deterministic writer.

    Product publication occurs in the same controlled run invocation. This
    receipt prevents a caller from manufacturing a matching pending DB row and
    then self-authorizing it by merely copying that row into the run artifact.
    """

    normalized_run_id = _normalized_text(run_id)
    if normalized_run_id is None or not records:
        return
    normalized: list[dict[str, Any]] = []
    for raw_record in records:
        if (
            not isinstance(raw_record, Mapping)
            or set(raw_record) != {"schema_version", "row"}
            or not isinstance(raw_record.get("row"), Mapping)
        ):
            raise RuntimeError("valuation writer emitted a malformed source record")
        record = valuation_source_record(raw_record["row"])
        if record is None or record != dict(raw_record):
            raise RuntimeError("valuation writer emitted a noncanonical source record")
        if _normalized_text(record["row"].get("source_run_id")) != normalized_run_id:
            raise RuntimeError("valuation writer receipt run identity mismatch")
        normalized.append(record)
    with _WRITER_RECEIPT_LOCK:
        receipts = _WRITER_RECEIPTS_BY_RUN.setdefault(normalized_run_id, {})
        for record in normalized:
            receipts[_source_record_key(record)] = record


def valuation_records_have_writer_receipts(
    run_id: str | None,
    records: Sequence[Mapping[str, Any]],
) -> bool:
    """Whether every exact record was emitted by this process's writer."""

    normalized_run_id = _normalized_text(run_id)
    if normalized_run_id is None:
        return not records
    try:
        keys = {_source_record_key(record) for record in records}
    except (TypeError, ValueError):
        return False
    with _WRITER_RECEIPT_LOCK:
        receipts = _WRITER_RECEIPTS_BY_RUN.get(normalized_run_id, {})
        return bool(keys) and keys <= set(receipts)


def _artifact_valuation_source_records(payload: Any) -> list[dict[str, Any]] | None:
    if not isinstance(payload, Mapping):
        return None
    raw_records = payload.get("valuation_source_records")
    if not isinstance(raw_records, list):
        return None
    records: list[dict[str, Any]] = []
    identities: list[tuple[str, str, str, str]] = []
    for raw_record in raw_records:
        if not isinstance(raw_record, Mapping) or set(raw_record) != {"schema_version", "row"}:
            return None
        row = raw_record.get("row")
        if (
            raw_record.get("schema_version") != VALUATION_SOURCE_RECORD_SCHEMA_VERSION
            or not isinstance(row, Mapping)
            or set(row) != set(_VALUATION_SOURCE_RECORD_FIELDS)
        ):
            return None
        normalized = {
            "schema_version": VALUATION_SOURCE_RECORD_SCHEMA_VERSION,
            "row": {field: row.get(field) for field in _VALUATION_SOURCE_RECORD_FIELDS},
        }
        try:
            json.dumps(
                normalized,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        except (TypeError, ValueError):
            return None
        identity = (
            str(row.get("ticker") or "").strip().upper(),
            str(row.get("as_of_date") or "").strip(),
            str(row.get("method") or "").strip(),
            str(row.get("created_at") or "").strip(),
        )
        if not all(identity):
            return None
        identities.append(identity)
        records.append(normalized)
    if identities != sorted(identities) or len(identities) != len(set(identities)):
        return None
    return records


def artifact_contains_exact_valuation_source_record(
    payload: Any,
    row: Mapping[str, Any] | Any,
) -> bool:
    records = _artifact_valuation_source_records(payload)
    expected = valuation_source_record(row)
    return records is not None and expected is not None and records.count(expected) == 1


def valuation_source_lineage(run_id: str | None) -> dict[str, str | None]:
    """Resolve a run to its current exact artifact binding, if authorized.

    A run id is still persisted when its artifact is not yet published or
    authorized.  That pending row remains ineligible, but the post-write
    binder can later identify precisely which rows were produced for the run
    without blessing unrelated same-day valuations.
    """

    normalized_run_id = _normalized_text(run_id)
    if normalized_run_id is None:
        return {
            "source_run_id": None,
            "source_artifact_path": None,
            "source_artifact_sha256": None,
        }
    binding = authorized_run_artifact_binding(normalized_run_id)
    if binding is not None:
        return {
            "source_run_id": binding["source_run_id"],
            "source_artifact_path": binding["source_artifact_path"],
            "source_artifact_sha256": binding["source_artifact_sha256"],
        }
    return {
        "source_run_id": normalized_run_id,
        "source_artifact_path": None,
        "source_artifact_sha256": None,
    }


def _artifact_authorized_valuation_packets(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    packets = payload.get("company_packets")
    if isinstance(packets, list):
        return [
            packet
            for packet in packets
            if isinstance(packet, dict)
            and str(packet.get("ticker") or "").strip()
            and str(packet.get("financial_integrity_status") or "").strip().upper() == PASS
        ]
    ticker = _normalized_text(payload.get("ticker"))
    if ticker is None or str(payload.get("status") or "").strip().upper() != "OK":
        return []
    return [payload]


def _artifact_authorized_valuation_tickers(payload: Any) -> set[str]:
    return {
        str(packet.get("ticker")).strip().upper()
        for packet in _artifact_authorized_valuation_packets(payload)
    }


def _artifact_proves_exact_issuer_binding(
    payload: Any,
    *,
    ticker: str,
    expected_issuer_cik: str | None,
    expected_issuer_aliases: Sequence[str],
) -> bool:
    """Require one authorized packet to bind the row to the expected issuer."""

    normalized_cik = normalize_cik(expected_issuer_cik)
    if normalized_cik is None:
        return False
    normalized_ticker = str(ticker or "").strip().upper()
    aliases = {
        str(item).strip().upper() for item in expected_issuer_aliases if str(item or "").strip()
    }
    if aliases and normalized_ticker not in aliases:
        return False
    matching_packets = [
        packet
        for packet in _artifact_authorized_valuation_packets(payload)
        if str(packet.get("ticker") or "").strip().upper() == normalized_ticker
    ]
    if len(matching_packets) != 1:
        return False
    packet = matching_packets[0]
    packet_cik = normalize_cik(packet.get("issuer_cik") or packet.get("cik"))
    return packet_cik == normalized_cik


def valuation_row_is_decision_eligible(
    row: Mapping[str, Any] | Any,
    *,
    require_exact_source_payload: bool = False,
    expected_issuer_cik: str | None = None,
    expected_issuer_aliases: Sequence[str] = (),
    require_exact_issuer_binding: bool = False,
) -> bool:
    """Return whether the selected row is authorized for a current surface.

    Callers must perform their normal newest-row selection before invoking this
    function.  Filtering candidate rows first would resurrect an older value.
    Surfaces that publish a row's complete opaque output payload can require it
    to equal the exact source artifact rather than merely bind to a ticker
    packet inside a broader sector-run artifact.
    """

    values = _row_mapping(row)
    source_run_id = _normalized_text(values.get("source_run_id"))
    source_path_text = _normalized_text(values.get("source_artifact_path"))
    source_sha256 = _normalized_text(values.get("source_artifact_sha256"))
    fingerprint = _normalized_text(values.get("financial_integrity_fingerprint"))
    ticker = _normalized_text(values.get("ticker"))
    if None in (source_run_id, source_path_text, source_sha256, fingerprint, ticker):
        return False
    assert source_path_text is not None
    assert source_sha256 is not None
    assert fingerprint is not None
    assert source_run_id is not None
    assert ticker is not None
    if _SHA256_RE.fullmatch(source_sha256) is None:
        return False

    source_path = Path(source_path_text).expanduser()
    if not source_path.is_absolute():
        return False
    try:
        resolved_path = source_path.resolve(strict=True)
    except OSError:
        return False
    if str(resolved_path) != source_path_text:
        return False
    try:
        integrity_status, source_bytes = authorized_artifact_bytes(resolved_path)
        if integrity_status != PASS or source_bytes is None:
            return False
        if hashlib.sha256(source_bytes).hexdigest() != source_sha256:
            return False
        artifact = json.loads(source_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(artifact, dict):
        return False
    if _normalized_text(artifact.get("run_id")) != source_run_id:
        return False
    if ticker.upper() not in _artifact_authorized_valuation_tickers(artifact):
        return False
    if (
        require_exact_issuer_binding
        or expected_issuer_cik is not None
        or bool(expected_issuer_aliases)
    ) and not _artifact_proves_exact_issuer_binding(
        artifact,
        ticker=ticker,
        expected_issuer_cik=expected_issuer_cik,
        expected_issuer_aliases=expected_issuer_aliases,
    ):
        return False
    if isinstance(artifact.get("company_packets"), list):
        if not artifact_contains_exact_valuation_source_record(artifact, values):
            return False
    else:
        try:
            outputs = json.loads(str(values.get("outputs_json") or ""))
        except json.JSONDecodeError:
            return False
        if outputs != artifact:
            return False
    try:
        return valuation_integrity_fingerprint(values) == fingerprint
    except (TypeError, ValueError):
        return False


def _valuation_selection_where(
    *,
    ticker: str,
    method: str | None,
    methods: tuple[str, ...] | None,
    as_of_date: str | None,
    before_as_of_date: str | None,
    exact_as_of_date: bool,
) -> tuple[str, tuple[Any, ...]]:
    normalized_ticker = str(ticker or "").strip().upper()
    normalized_as_of = _normalized_text(as_of_date)
    normalized_before = _normalized_text(before_as_of_date)
    if not normalized_ticker:
        raise ValueError("ticker is required for valuation selection")
    if exact_as_of_date and normalized_as_of is None:
        raise ValueError("exact_as_of_date requires as_of_date")
    if normalized_as_of is not None and normalized_before is not None:
        raise ValueError("as_of_date and before_as_of_date are mutually exclusive")
    if method is not None and methods is not None:
        raise ValueError("method and methods are mutually exclusive")

    clauses = ["ticker = ?"]
    params: list[Any] = [normalized_ticker]
    if method is not None:
        normalized_method = str(method).strip()
        if not normalized_method:
            raise ValueError("method must be nonempty")
        clauses.append("method = ?")
        params.append(normalized_method)
    elif methods is not None:
        if not methods:
            return "0 = 1", ()
        placeholders = ",".join("?" for _ in methods)
        clauses.append(f"method IN ({placeholders})")
        params.extend(methods)

    if normalized_as_of is not None:
        clauses.append(f"as_of_date {'=' if exact_as_of_date else '<='} ?")
        params.append(normalized_as_of)
    if normalized_before is not None:
        clauses.append("as_of_date < ?")
        params.append(normalized_before)
    return " AND ".join(clauses), tuple(params)


def latest_decision_eligible_valuation_row(
    conn: Any,
    *,
    ticker: str,
    method: str | None = None,
    as_of_date: str | None = None,
    before_as_of_date: str | None = None,
    exact_as_of_date: bool = False,
    require_exact_source_payload: bool = False,
    expected_issuer_cik: str | None = None,
    expected_issuer_aliases: Sequence[str] = (),
    require_exact_issuer_binding: bool = False,
) -> Any | None:
    """Select the newest candidate row, then authorize that exact row.

    Authorization is intentionally *not* part of the SQL predicate. If the
    newest candidate is stale, tampered, or unaudited, the result is ``None``;
    an older authorized row is never resurrected as current evidence.
    """

    where_sql, params = _valuation_selection_where(
        ticker=ticker,
        method=method,
        methods=None,
        as_of_date=as_of_date,
        before_as_of_date=before_as_of_date,
        exact_as_of_date=exact_as_of_date,
    )
    columns_sql = ", ".join(DECISION_ELIGIBLE_VALUATION_COLUMNS)
    row = conn.execute(
        f"""
        SELECT {columns_sql}
        FROM valuations
        WHERE {where_sql}
        ORDER BY as_of_date DESC, created_at DESC, id DESC
        LIMIT 1
        """,
        params,
    ).fetchone()
    if row is None or not valuation_row_is_decision_eligible(
        row,
        require_exact_source_payload=require_exact_source_payload,
        expected_issuer_cik=expected_issuer_cik,
        expected_issuer_aliases=expected_issuer_aliases,
        require_exact_issuer_binding=require_exact_issuer_binding,
    ):
        return None
    return row


def latest_decision_eligible_valuation_rows(
    conn: Any,
    *,
    ticker: str,
    methods: tuple[str, ...] | list[str] | set[str] | None = None,
    as_of_date: str | None = None,
    before_as_of_date: str | None = None,
    exact_as_of_date: bool = False,
    require_exact_source_payload: bool = False,
    expected_issuer_cik: str | None = None,
    expected_issuer_aliases: Sequence[str] = (),
    require_exact_issuer_binding: bool = False,
) -> list[Any]:
    """Return at most one authorized newest row per valuation method.

    Candidate selection happens for every method before any authorization
    check. An invalid newest row suppresses only that method and never causes a
    search for an older PASS row.
    """

    normalized_methods: tuple[str, ...] | None = None
    if methods is not None:
        normalized_methods = tuple(
            dict.fromkeys(str(method).strip() for method in methods if str(method).strip())
        )
        if not normalized_methods:
            return []
    where_sql, params = _valuation_selection_where(
        ticker=ticker,
        method=None,
        methods=normalized_methods,
        as_of_date=as_of_date,
        before_as_of_date=before_as_of_date,
        exact_as_of_date=exact_as_of_date,
    )
    columns_sql = ", ".join(DECISION_ELIGIBLE_VALUATION_COLUMNS)
    rows = conn.execute(
        f"""
        SELECT {columns_sql}
        FROM valuations
        WHERE {where_sql}
        ORDER BY method ASC, as_of_date DESC, created_at DESC, id DESC
        """,
        params,
    ).fetchall()
    newest_by_method: dict[str, Any] = {}
    for row in rows:
        newest_by_method.setdefault(str(row["method"]), row)
    return [
        row
        for row in newest_by_method.values()
        if valuation_row_is_decision_eligible(
            row,
            require_exact_source_payload=require_exact_source_payload,
            expected_issuer_cik=expected_issuer_cik,
            expected_issuer_aliases=expected_issuer_aliases,
            require_exact_issuer_binding=require_exact_issuer_binding,
        )
    ]


def bind_authorized_valuation_rows(
    *,
    run_id: str,
    tickers: list[str] | tuple[str, ...] | set[str],
    as_of_date: str,
    cfg: Any | None = None,
) -> int:
    """Bind pending live valuation rows to one newly authorized run artifact.

    Only rows already stamped with this ``run_id`` by the valuation writer are
    eligible.  This prevents a new sector run from retroactively authorizing
    unrelated valuations that merely share a ticker and date.  Pre-binding
    rows are copied to history, and the exact artifact bytes are rechecked
    before the transaction is allowed to commit.
    """

    normalized_run_id = _normalized_text(run_id)
    normalized_as_of = _normalized_text(as_of_date)
    normalized_tickers = sorted(
        {str(ticker).strip().upper() for ticker in tickers if str(ticker).strip()}
    )
    if normalized_run_id is None or normalized_as_of is None:
        raise ValueError("run_id and as_of_date are required for valuation lineage")
    if not normalized_tickers:
        return 0

    binding = authorized_run_artifact_binding(normalized_run_id)
    if binding is None:
        raise RuntimeError(f"Cannot bind valuations for unaudited run {normalized_run_id}")
    source_path = Path(binding["source_artifact_path"])
    source_sha256 = binding["source_artifact_sha256"]
    try:
        integrity_status, source_bytes = authorized_artifact_bytes(source_path)
        if integrity_status != PASS or source_bytes is None:
            raise RuntimeError("Valuation source is no longer authorized")
        artifact = json.loads(source_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Authorized valuation source is unreadable") from exc
    if hashlib.sha256(source_bytes).hexdigest() != source_sha256:
        raise RuntimeError("Authorized valuation source changed before binding")
    if not isinstance(artifact, dict):
        raise RuntimeError("Authorized valuation source must be a JSON object")
    artifact_tickers = _artifact_authorized_valuation_tickers(artifact)
    missing_tickers = sorted(set(normalized_tickers) - artifact_tickers)
    if missing_tickers:
        raise RuntimeError(
            "Authorized valuation source omits requested tickers: " + ", ".join(missing_tickers)
        )
    if _artifact_valuation_source_records(artifact) is None:
        raise RuntimeError("Authorized valuation source omits exact valuation source records")

    from app.db import get_db, utc_now_iso

    placeholders = ",".join("?" for _ in normalized_tickers)
    updated = 0
    with get_db(cfg=cfg) as conn:
        rows = conn.execute(
            f"""
            SELECT *
            FROM valuations
            WHERE source_run_id = ?
              AND as_of_date = ?
              AND ticker IN ({placeholders})
            ORDER BY ticker, method, id
            """,
            (normalized_run_id, normalized_as_of, *normalized_tickers),
        ).fetchall()
        archived_at = utc_now_iso()
        for row in rows:
            values = _row_mapping(row)
            if not artifact_contains_exact_valuation_source_record(artifact, values):
                # A pending row is not an authority. Only a record supplied by
                # the controlled writer/packet path and sealed into the exact
                # artifact may be bound; unrelated same-run rows stay pending.
                continue
            values.update(binding)
            fingerprint = valuation_integrity_fingerprint(values)
            if (
                values.get("source_artifact_path") == row["source_artifact_path"]
                and values.get("source_artifact_sha256") == row["source_artifact_sha256"]
                and fingerprint == row["financial_integrity_fingerprint"]
            ):
                continue
            conn.execute(
                """
                INSERT INTO valuations_history(
                    source_id, ticker, as_of_date, method, inputs_json,
                    outputs_json, warnings_json, created_at,
                    valuation_writer_version, quality_gate_verdict,
                    confidence_class, gate_reason_codes, valuation_headwinds,
                    valuation_supports, source_run_id, source_artifact_path,
                    source_artifact_sha256, financial_integrity_fingerprint,
                    archived_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["id"],
                    row["ticker"],
                    row["as_of_date"],
                    row["method"],
                    row["inputs_json"],
                    row["outputs_json"],
                    row["warnings_json"],
                    row["created_at"],
                    row["valuation_writer_version"],
                    row["quality_gate_verdict"],
                    row["confidence_class"],
                    row["gate_reason_codes"],
                    row["valuation_headwinds"],
                    row["valuation_supports"],
                    row["source_run_id"],
                    row["source_artifact_path"],
                    row["source_artifact_sha256"],
                    row["financial_integrity_fingerprint"],
                    archived_at,
                ),
            )
            conn.execute(
                """
                UPDATE valuations
                SET source_artifact_path = ?,
                    source_artifact_sha256 = ?,
                    financial_integrity_fingerprint = ?
                WHERE id = ?
                """,
                (
                    binding["source_artifact_path"],
                    binding["source_artifact_sha256"],
                    fingerprint,
                    row["id"],
                ),
            )
            updated += 1
        final_status, final_bytes = authorized_artifact_bytes(source_path)
        if final_status != PASS or final_bytes is None or final_bytes != source_bytes:
            raise RuntimeError("Authorized valuation source changed during binding")
    return updated
