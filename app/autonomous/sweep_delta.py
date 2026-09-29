"""Delta-sweep support: persisted terminal coverage + unswept computation.

A name is "unswept" for (band, sector) if it never appeared in any prior
completed sector-run terminal set for that band. Coverage is read from the
sector_run_loaded_sets table, seeded from auditable persisted run-artifact
states and written by every completed sector run going forward — never from
the raw loader count or log parsing.

Explicit-ticker probe runs record their rows but never count as band
coverage (a one-name probe is not sweep coverage). Names that already carry
an autonomous review verdict in the ticker_outcomes ledger are "carried":
they stay out of the fresh-LLM review set and their existing verdict rides
along in the delta report.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import stat
from collections.abc import Mapping
from datetime import date, datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.autonomous.artifact_financial_audit import (
    PASS as FINANCIAL_INTEGRITY_PASS,
    active_financial_integrity_manifest_path,
    authorized_artifact_bytes,
    authorized_run_artifact_binding,
    financial_integrity_manifest_is_usable,
)
from app.autonomous.candidate_review import (
    candidate_memo_is_substantive,
    candidate_memo_schema_name,
    provider_usage_attestation,
)
from app.outcomes.lineage import outcome_row_is_decision_eligible
from app.valuation.lineage import latest_decision_eligible_valuation_row

# Explicit production sweep contract. Keep this list independent from the
# census taxonomy so exact-set tests fail whenever either registry changes
# without the other.
CANONICAL_SWEEP_SECTORS = (
    "aerospace_defense",
    "automotive",
    "biotech",
    "building_products",
    "business_services",
    "capital_markets",
    "chemicals",
    "construction_machinery",
    "construction_services",
    "consumer_discretionary",
    "consumer_services",
    "consumer_staples",
    "diversified_industrials",
    "education_services",
    "energy",
    "enterprise_software",
    "healthcare_pharma",
    "healthcare_services",
    "hospitality_gaming",
    "industrial_tech",
    "insurance",
    "internet_services",
    "large_cap_financials",
    "media_entertainment",
    "medical_devices",
    "metals_mining",
    "payments_fintech",
    "reits",
    "restaurants_food_service",
    "retail",
    "semiconductors",
    "telecom",
    "transportation_logistics",
    "utilities",
)

# A classic full-universe pass must enumerate this exact disjoint grid.  The
# composite smid_cap and large_and_mega aliases are intentionally absent: a
# closure check over overlapping bands can double-count names and hide an
# omitted atomic cell.
V1_ATOMIC_BANDS = (
    "micro_cap",
    "small_cap",
    "mid_cap",
    "large_cap",
    "mega_cap",
)

_REVIEW_GRADES = {"ACTIONABLE", "WATCHLIST_ONLY", "DATA_INCOMPLETE", "AVOID"}
V1_SECTOR_CONTEXT_LIMIT = 25
_V1_COVERAGE_COMPLETE_DISPOSITIONS = {
    "LLM_CANDIDATE_REVIEW_COMPLETED",
    "STRUCTURAL_SCREENED",
    "CARRIED_VERDICT",
    "NEEDS_DATA_SPARSE_HISTORY",
    "NEEDS_DATA_FRAMEWORK_EVIDENCE",
}
_V1_CARRIED_SUPPRESSING_DISPOSITIONS = {
    "LLM_CANDIDATE_REVIEW_FAILED",
    "LLM_CANDIDATE_REVIEW_UNAUDITABLE",
}
_V2_COVERAGE_COMPLETE_DISPOSITIONS = {
    "UNDERWRITTEN",
    "SCREENED_OUT",
    "OUT_OF_SCOPE",
}
_COVERAGE_RECEIPT_SCHEMA_VERSION = "classic_coverage_receipt_v1"
_ZERO_COST_RECEIPT = "ZERO_COST"
_UNKNOWN_CAP_PROJECTION_RECEIPT = "UNKNOWN_CAP_PROJECTION"
_COVERAGE_RECEIPT_FILENAME = "coverage_receipt.json"
_SAFE_RUN_COMPONENT = re.compile(r"[A-Za-z0-9_.-]+")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_table(conn: sqlite3.Connection) -> None:
    from app.db import _migrate_sector_run_loaded_sets

    _migrate_sector_run_loaded_sets(conn)


def _canonical_coverage_receipt_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
            separators=(",", ": "),
        )
        + "\n"
    ).encode("utf-8")


def _coverage_receipt_path(run_id: str) -> Path:
    from app.config import get_config

    normalized = str(run_id or "").strip()
    if (
        not normalized
        or _SAFE_RUN_COMPONENT.fullmatch(normalized) is None
        or normalized in {".", ".."}
    ):
        raise ValueError("coverage receipt run_id is not a safe canonical path component")
    root = (Path(get_config().runs_dir).resolve() / "autonomous_sector_coverage").resolve()
    return root / normalized / _COVERAGE_RECEIPT_FILENAME


def _run_id_requires_coverage_receipt(run_id: str) -> bool:
    """Reserve synthetic coverage identities for receipt-only authority."""

    normalized = str(run_id or "").strip()
    if normalized.startswith("coverage_only_v1_"):
        return True
    source_run_id, marker, band = normalized.rpartition("__unknown_cap__")
    return bool(marker and source_run_id and band in V1_ATOMIC_BANDS)


def _write_coverage_receipt(payload: Mapping[str, Any]) -> dict[str, str]:
    """Create one immutable canonical receipt and return its exact byte binding."""

    run_id = str(payload.get("run_id") or "").strip()
    kind = str(payload.get("kind") or "").strip().upper()
    if payload.get("schema_version") != _COVERAGE_RECEIPT_SCHEMA_VERSION or kind not in {
        _ZERO_COST_RECEIPT,
        _UNKNOWN_CAP_PROJECTION_RECEIPT,
    }:
        raise ValueError("coverage receipt payload has an unsupported schema or kind")
    path = _coverage_receipt_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = _canonical_coverage_receipt_bytes(payload)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(expected)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # A hard-link publishes without replacing a receipt created by a
            # concurrent invocation. Coverage receipts are immutable.
            os.link(temporary, path)
        except FileExistsError:
            try:
                current = path.read_bytes()
            except OSError as exc:
                raise RuntimeError("existing coverage receipt is unreadable") from exc
            if current != expected:
                raise RuntimeError("refusing to replace an existing coverage receipt") from None
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return {
        "kind": kind,
        "path": str(path),
        "sha256": sha256(expected).hexdigest(),
    }


def record_loaded_set(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    sector: str,
    market_cap_focus: str,
    source: str,
    tickers: list[str],
    loaded_at: str | None = None,
    pipeline_version: str | None = None,
    candidate_dispositions: dict[str, str] | None = None,
    coverage_campaign_id: str | None = None,
    coverage_evidence_by_ticker: dict[str, Any] | None = None,
    coverage_authority_kind: str | None = None,
    coverage_authority_path: str | None = None,
    coverage_authority_sha256: str | None = None,
    coverage_source_run_id: str | None = None,
) -> int:
    """Persist auditable per-ticker run states. Idempotent on (run_id, ticker).

    A v1 row is coverage-complete only when it carries one of the explicit
    terminal dispositions above. Legacy loader-only rows remain in the table
    for audit history but no longer suppress future review.
    """
    _ensure_table(conn)
    stamp = loaded_at or _now()
    normalized_pipeline = str(pipeline_version or "").strip().lower() or None
    normalized_campaign = str(coverage_campaign_id or "").strip() or None
    normalized_authority_kind = str(coverage_authority_kind or "").strip().upper() or None
    normalized_authority_path = str(coverage_authority_path or "").strip() or None
    normalized_authority_sha256 = str(coverage_authority_sha256 or "").strip().lower() or None
    normalized_source_run_id = str(coverage_source_run_id or "").strip() or None
    authority_values = (
        normalized_authority_kind,
        normalized_authority_path,
        normalized_authority_sha256,
    )
    if any(authority_values) and (
        not all(authority_values)
        or normalized_authority_kind not in {_ZERO_COST_RECEIPT, _UNKNOWN_CAP_PROJECTION_RECEIPT}
        or re.fullmatch(r"[0-9a-f]{64}", normalized_authority_sha256 or "") is None
    ):
        raise ValueError("coverage authority requires one complete canonical receipt binding")
    if (
        normalized_authority_kind == _UNKNOWN_CAP_PROJECTION_RECEIPT
        and normalized_source_run_id is None
    ):
        raise ValueError("unknown-cap projection authority requires its exact source run")
    if normalized_authority_kind == _ZERO_COST_RECEIPT and normalized_source_run_id is not None:
        raise ValueError("zero-cost coverage authority cannot name a source run")
    dispositions = {
        str(ticker).strip().upper(): str(state or "").strip().upper()
        for ticker, state in (candidate_dispositions or {}).items()
        if str(ticker).strip()
    }
    evidence = {
        str(ticker).strip().upper(): value
        for ticker, value in (coverage_evidence_by_ticker or {}).items()
        if str(ticker).strip() and isinstance(value, dict)
    }
    inserted = 0
    for ticker in dict.fromkeys(str(t).strip().upper() for t in tickers if str(t).strip()):
        state = dispositions.get(ticker)
        coverage_complete = (
            state in _V2_COVERAGE_COMPLETE_DISPOSITIONS
            if normalized_pipeline == "v2"
            else state in _V1_COVERAGE_COMPLETE_DISPOSITIONS
        )
        evidence_json = (
            json.dumps(evidence[ticker], sort_keys=True, separators=(",", ":"))
            if ticker in evidence
            else None
        )
        evidence_sha256 = (
            sha256(evidence_json.encode("utf-8")).hexdigest() if evidence_json is not None else None
        )
        cursor = conn.execute(
            """
            INSERT INTO sector_run_loaded_sets(
                run_id, sector, market_cap_focus, source, ticker, loaded_at,
                pipeline_version, candidate_disposition, coverage_campaign_id,
                coverage_evidence_json, coverage_evidence_sha256,
                coverage_authority_kind, coverage_authority_path,
                coverage_authority_sha256, coverage_source_run_id,
                coverage_complete)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, ticker) DO NOTHING
            """,
            (
                run_id,
                sector,
                str(market_cap_focus).strip().lower(),
                source,
                ticker,
                stamp,
                normalized_pipeline,
                state,
                normalized_campaign,
                evidence_json,
                evidence_sha256,
                normalized_authority_kind,
                normalized_authority_path,
                normalized_authority_sha256,
                normalized_source_run_id,
                1 if coverage_complete else 0,
            ),
        )
        inserted += cursor.rowcount
        if cursor.rowcount == 0 and state:
            # Artifact backfills can encounter rows inserted by the old
            # loader-only writer. Upgrade exact terminal states in place; the
            # WHERE clause keeps repeat backfills idempotent.
            updated = conn.execute(
                """
                UPDATE sector_run_loaded_sets
                SET pipeline_version = ?, candidate_disposition = ?,
                    coverage_campaign_id = COALESCE(?, coverage_campaign_id),
                    coverage_evidence_json = COALESCE(?, coverage_evidence_json),
                    coverage_evidence_sha256 = COALESCE(?, coverage_evidence_sha256),
                    coverage_authority_kind = COALESCE(?, coverage_authority_kind),
                    coverage_authority_path = COALESCE(?, coverage_authority_path),
                    coverage_authority_sha256 = COALESCE(?, coverage_authority_sha256),
                    coverage_source_run_id = COALESCE(?, coverage_source_run_id),
                    coverage_complete = ?
                WHERE run_id = ? AND ticker = ?
                  AND (
                    COALESCE(pipeline_version, '') != COALESCE(?, '')
                    OR COALESCE(candidate_disposition, '') != ?
                    OR (
                        ? IS NOT NULL
                        AND COALESCE(coverage_campaign_id, '') != ?
                    )
                    OR (
                        ? IS NOT NULL
                        AND COALESCE(coverage_evidence_sha256, '') != ?
                    )
                    OR (
                        ? IS NOT NULL
                        AND COALESCE(coverage_authority_kind, '') != ?
                    )
                    OR (
                        ? IS NOT NULL
                        AND COALESCE(coverage_authority_path, '') != ?
                    )
                    OR (
                        ? IS NOT NULL
                        AND COALESCE(coverage_authority_sha256, '') != ?
                    )
                    OR (
                        ? IS NOT NULL
                        AND COALESCE(coverage_source_run_id, '') != ?
                    )
                    OR coverage_complete != ?
                  )
                """,
                (
                    normalized_pipeline,
                    state,
                    normalized_campaign,
                    evidence_json,
                    evidence_sha256,
                    normalized_authority_kind,
                    normalized_authority_path,
                    normalized_authority_sha256,
                    normalized_source_run_id,
                    1 if coverage_complete else 0,
                    run_id,
                    ticker,
                    normalized_pipeline,
                    state,
                    normalized_campaign,
                    normalized_campaign,
                    evidence_sha256,
                    evidence_sha256,
                    normalized_authority_kind,
                    normalized_authority_kind,
                    normalized_authority_path,
                    normalized_authority_path,
                    normalized_authority_sha256,
                    normalized_authority_sha256,
                    normalized_source_run_id,
                    normalized_source_run_id,
                    1 if coverage_complete else 0,
                ),
            )
            inserted += updated.rowcount
    return inserted


def record_unknown_cap_cross_band_coverage(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    sector: str,
    source: str,
    primary_band: str,
    unknown_cap_tickers: list[str],
    candidate_dispositions: dict[str, str],
    loaded_at: str | None = None,
    coverage_campaign_id: str | None = None,
    coverage_evidence_by_ticker: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Project one cap-independent review across every atomic unknown-cap cell.

    An unresolved market cap is deliberately admitted by every atomic band so
    it cannot disappear from coverage. The underlying company review is still
    identical, however, and paying for it five times adds no evidence. Exact
    terminal or retry states from the first normal-loader run are therefore
    copied to the other four band cells with explicit projection provenance.
    """
    normalized_dispositions = {
        str(ticker).strip().upper(): str(state or "").strip().upper()
        for ticker, state in candidate_dispositions.items()
        if str(ticker).strip()
    }
    normalized_unknown = sorted(
        {
            str(ticker).strip().upper()
            for ticker in unknown_cap_tickers
            if str(ticker).strip() and str(ticker).strip().upper() in normalized_dispositions
        }
    )
    normalized_primary = str(primary_band).strip().lower()
    if not normalized_unknown or normalized_primary not in V1_ATOMIC_BANDS:
        return {"tickers": [], "bands": [], "rows_inserted_or_upgraded": 0}
    normalized_campaign = str(coverage_campaign_id or "").strip() or None
    normalized_evidence = {
        str(ticker).strip().upper(): value
        for ticker, value in (coverage_evidence_by_ticker or {}).items()
        if str(ticker).strip().upper() in normalized_unknown and isinstance(value, dict)
    }

    rows = 0
    bands: list[str] = []
    for band in V1_ATOMIC_BANDS:
        if band == normalized_primary:
            continue
        projection_run_id = f"{run_id}__unknown_cap__{band}"
        receipt = _write_coverage_receipt(
            {
                "schema_version": _COVERAGE_RECEIPT_SCHEMA_VERSION,
                "kind": _UNKNOWN_CAP_PROJECTION_RECEIPT,
                "run_id": projection_run_id,
                "pipeline_version": "v1",
                "sector": str(sector).strip().lower(),
                "market_cap_focus": band,
                "source": f"{source}:unknown_cap_cross_band_projection",
                "coverage_campaign_id": normalized_campaign,
                "source_run_id": str(run_id).strip(),
                "source_market_cap_focus": normalized_primary,
                "rows": [
                    {
                        "ticker": ticker,
                        "candidate_disposition": normalized_dispositions[ticker],
                        "coverage_complete": (
                            normalized_dispositions[ticker] in _V1_COVERAGE_COMPLETE_DISPOSITIONS
                        ),
                        "coverage_evidence": normalized_evidence.get(ticker),
                    }
                    for ticker in normalized_unknown
                ],
            }
        )
        rows += record_loaded_set(
            conn,
            run_id=projection_run_id,
            sector=sector,
            market_cap_focus=band,
            source=f"{source}:unknown_cap_cross_band_projection",
            tickers=normalized_unknown,
            loaded_at=loaded_at,
            pipeline_version="v1",
            candidate_dispositions={
                ticker: normalized_dispositions[ticker] for ticker in normalized_unknown
            },
            coverage_campaign_id=coverage_campaign_id,
            coverage_evidence_by_ticker=normalized_evidence,
            coverage_authority_kind=receipt["kind"],
            coverage_authority_path=receipt["path"],
            coverage_authority_sha256=receipt["sha256"],
            coverage_source_run_id=str(run_id).strip(),
        )
        bands.append(band)
    return {
        "tickers": normalized_unknown,
        "bands": bands,
        "rows_inserted_or_upgraded": rows,
    }


def _normalized_values(values: Any) -> list[str]:
    if not isinstance(values, (list, tuple, set)):
        return []
    return list(dict.fromkeys(str(value).strip().upper() for value in values if str(value).strip()))


def _normalized_ticker_mapping(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    normalized: dict[str, Any] = {}
    for raw_ticker, item in value.items():
        ticker = str(raw_ticker).strip().upper()
        if not ticker or ticker in normalized:
            return None
        normalized[ticker] = item
    return normalized


def unknown_cap_tickers_from_selection(selection: Any) -> list[str]:
    """Return loaded tickers whose persisted cap classification is unresolved."""
    if not isinstance(selection, dict):
        return []
    loaded = set(_normalized_values(selection.get("loaded_tickers")))
    classifications = selection.get("cap_classifications")
    if not isinstance(classifications, dict):
        return []
    return sorted(
        ticker
        for raw_ticker, raw_classification in classifications.items()
        if (ticker := str(raw_ticker).strip().upper()) in loaded
        and isinstance(raw_classification, dict)
        and str(raw_classification.get("cap_source") or "").strip().lower() == "unknown"
    )


def v1_terminal_coverage_from_artifact(
    artifact: Any,
    *,
    fallback_candidate_selection: dict[str, Any] | None = None,
    fallback_delta_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Project one completed v1 artifact into exact terminal coverage states.

    Raw ``loaded_tickers`` are intentionally absent. Only structural screens,
    carried prior verdicts, explicit runtime NEEDS_DATA states, and final
    packet tickers with a successful, source-bound per-candidate LLM memo call
    close coverage. Deterministic memo fallbacks remain open for retry.
    """
    payload = artifact if isinstance(artifact, dict) else artifact.to_dict()
    artifact_selection = (
        payload.get("candidate_selection")
        if isinstance(payload.get("candidate_selection"), dict)
        else {}
    )
    fallback_selection = (
        fallback_candidate_selection if isinstance(fallback_candidate_selection, dict) else {}
    )
    selection = {**fallback_selection, **artifact_selection}
    if "structural_gate_results" not in artifact_selection:
        selection["structural_gate_results"] = fallback_selection.get("structural_gate_results", {})
    loaded_tickers = set(_normalized_values(selection.get("loaded_tickers")))

    delta_audit = (
        artifact_selection.get("delta_audit")
        if isinstance(artifact_selection.get("delta_audit"), dict)
        else fallback_delta_audit
        if isinstance(fallback_delta_audit, dict)
        else {}
    )
    categories: dict[str, list[str]] = {
        "structural_screened": [],
        "carried_verdict": [],
        "needs_data_sparse_history": [],
        "needs_data_framework_evidence": [],
        "llm_candidate_review_completed": [],
        "llm_candidate_review_failed": [],
    }

    gate_results = selection.get("structural_gate_results")
    if isinstance(gate_results, dict):
        categories["structural_screened"] = sorted(
            str(ticker).strip().upper()
            for ticker, result in gate_results.items()
            if str(ticker).strip()
            and str(ticker).strip().upper() in loaded_tickers
            and isinstance(result, dict)
            and bool(result.get("quarantined"))
            and not bool(result.get("excluded_error"))
        )

    carried = delta_audit.get("carried_verdicts")
    if isinstance(carried, dict):
        categories["carried_verdict"] = sorted(
            str(ticker).strip().upper()
            for ticker in carried
            if str(ticker).strip() and str(ticker).strip().upper() in loaded_tickers
        )

    history_filter = selection.get("financial_history_filter")
    if (
        isinstance(history_filter, dict)
        and str(history_filter.get("status") or "").upper() == "FILTERED_SPARSE_FINANCIAL_HISTORY"
    ):
        categories["needs_data_sparse_history"] = sorted(
            set(_normalized_values(history_filter.get("excluded_tickers"))) & loaded_tickers
        )

    framework_filter = selection.get("framework_evidence_filter")
    if (
        isinstance(framework_filter, dict)
        and str(framework_filter.get("status") or "").upper() == "FILTERED_ZERO_PACKET_SUPPORT"
    ):
        categories["needs_data_framework_evidence"] = sorted(
            set(_normalized_values(framework_filter.get("excluded_tickers"))) & loaded_tickers
        )

    packet_tickers = {
        str(item.get("ticker") or "").strip().upper()
        for item in payload.get("company_packets") or []
        if isinstance(item, dict)
        and str(item.get("ticker") or "").strip()
        and str(item.get("ticker") or "").strip().upper() in loaded_tickers
    }
    memo_body = payload.get("memo_body")
    memo_candidates = (
        memo_body.get("candidates")
        if isinstance(memo_body, dict) and isinstance(memo_body.get("candidates"), dict)
        else {}
    )
    expected_provider = str(selection.get("coverage_expected_provider") or "").strip().lower()
    expected_model = str(selection.get("coverage_expected_model") or "").strip()
    usage_attestation = provider_usage_attestation(payload)
    attested_bindings = {
        (str(row.get("provider") or ""), str(row.get("model") or ""))
        for row in usage_attestation["provider_models"]
    }
    expected_binding_is_clean = not expected_provider or (
        usage_attestation["valid"] and attested_bindings == {(expected_provider, expected_model)}
    )
    successful_provider_schemas = {
        str(row.get("schema_name") or "")
        for row in payload.get("provider_usage") or []
        if isinstance(row, dict)
        and str(row.get("status") or "").upper() == "OK"
        and not str(row.get("fallback_from_provider") or "").strip()
        and (
            not expected_provider
            or str(row.get("provider") or "").strip().lower() == expected_provider
        )
        and (not expected_model or str(row.get("model") or "").strip() == expected_model)
        and expected_binding_is_clean
    }
    completed_reviews: list[str] = []
    for ticker in sorted(packet_tickers):
        candidate = memo_candidates.get(ticker)
        payload_ticker = (
            str(candidate.get("ticker") or "").strip().upper()
            if isinstance(candidate, dict)
            else ""
        )
        if (
            isinstance(candidate, dict)
            and str(candidate.get("source") or "").lower() == "llm"
            and str(candidate.get("status") or "").upper() == "OK"
            and (not payload_ticker or payload_ticker == ticker)
            and candidate_memo_is_substantive(candidate)
            and candidate_memo_schema_name(ticker) in successful_provider_schemas
        ):
            completed_reviews.append(ticker)
    categories["llm_candidate_review_completed"] = completed_reviews
    categories["llm_candidate_review_failed"] = sorted(packet_tickers - set(completed_reviews))

    disposition_by_category = {
        "structural_screened": "STRUCTURAL_SCREENED",
        "carried_verdict": "CARRIED_VERDICT",
        "needs_data_sparse_history": "NEEDS_DATA_SPARSE_HISTORY",
        "needs_data_framework_evidence": "NEEDS_DATA_FRAMEWORK_EVIDENCE",
        "llm_candidate_review_completed": "LLM_CANDIDATE_REVIEW_COMPLETED",
    }
    dispositions: dict[str, str] = {}
    for category, state in disposition_by_category.items():
        for ticker in categories[category]:
            dispositions[ticker] = state
    terminal_dispositions = dict(dispositions)
    for ticker in categories["llm_candidate_review_failed"]:
        dispositions[ticker] = "LLM_CANDIDATE_REVIEW_FAILED"
    return {
        **categories,
        "terminal_tickers": sorted(terminal_dispositions),
        "terminal_candidate_dispositions": terminal_dispositions,
        "disposition_tickers": sorted(dispositions),
        "candidate_dispositions": dispositions,
    }


def backfill_loaded_sets_from_artifacts(runs_dir: str | Path | None = None) -> dict[str, int]:
    """Seed exact terminal coverage from persisted sector-run artifacts."""
    from app.config import get_config
    from app.db import get_db

    base = Path(runs_dir) if runs_dir is not None else get_config().outputs_dir / "runs"
    artifacts = sorted(base.glob("autonomous_sector/*/autonomous_sector_run.json"))
    runs = rows = skipped = 0
    active_manifest = active_financial_integrity_manifest_path()
    if not financial_integrity_manifest_is_usable(active_manifest):
        return {
            "artifacts": len(artifacts),
            "runs_recorded": 0,
            "rows_inserted": 0,
            "skipped": len(artifacts),
        }
    with get_db() as conn:
        for path in artifacts:
            integrity_status, artifact_bytes = authorized_artifact_bytes(
                path,
                active_manifest,
            )
            if integrity_status != FINANCIAL_INTEGRITY_PASS or artifact_bytes is None:
                skipped += 1
                continue
            try:
                artifact = json.loads(artifact_bytes.decode("utf-8"))
                if not isinstance(artifact, dict):
                    raise ValueError("run artifact root must be an object")
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                skipped += 1
                continue
            status = str(artifact.get("status") or "COMPLETED")
            if status != "COMPLETED":
                skipped += 1
                continue
            selection = artifact.get("candidate_selection") or {}
            loaded = selection.get("loaded_tickers") or []
            run_id = str(artifact.get("run_id") or path.parent.name)
            sector = str(selection.get("sector") or artifact.get("sector") or "")
            focus = str(selection.get("market_cap_focus") or artifact.get("market_cap_focus") or "")
            source = str(selection.get("source") or "unknown")
            pipeline_version = str(artifact.get("pipeline_version") or "").lower() or None
            coverage_campaign_id = str(selection.get("coverage_campaign_id") or "").strip() or None
            if not loaded or not sector or not focus:
                skipped += 1
                continue
            if pipeline_version == "v2":
                disposition_rows = artifact.get("candidate_dispositions") or []
                candidate_dispositions = {
                    str(item.get("ticker") or "").upper(): str(
                        item.get("terminal_state") or ""
                    ).upper()
                    for item in disposition_rows
                    if isinstance(item, dict) and str(item.get("ticker") or "").strip()
                }
                terminal_tickers = list(loaded)
            else:
                pipeline_version = "v1"
                terminal = v1_terminal_coverage_from_artifact(artifact)
                candidate_dispositions = terminal["candidate_dispositions"]
                loaded_set = set(_normalized_values(loaded))
                terminal_tickers = [
                    ticker for ticker in terminal["disposition_tickers"] if ticker in loaded_set
                ]
            rows += record_loaded_set(
                conn,
                run_id=run_id,
                sector=sector,
                market_cap_focus=focus,
                source=source,
                tickers=terminal_tickers,
                loaded_at=str(artifact.get("created_at") or _now()),
                pipeline_version=pipeline_version,
                candidate_dispositions=candidate_dispositions,
                coverage_campaign_id=coverage_campaign_id,
            )
            if pipeline_version == "v1":
                projection = record_unknown_cap_cross_band_coverage(
                    conn,
                    run_id=run_id,
                    sector=sector,
                    source=source,
                    primary_band=focus,
                    unknown_cap_tickers=unknown_cap_tickers_from_selection(selection),
                    candidate_dispositions=candidate_dispositions,
                    loaded_at=str(artifact.get("created_at") or _now()),
                    coverage_campaign_id=coverage_campaign_id,
                )
                rows += int(projection["rows_inserted_or_upgraded"])
            runs += 1
    return {
        "artifacts": len(artifacts),
        "runs_recorded": runs,
        "rows_inserted": rows,
        "skipped": skipped,
    }


def _authorized_artifact_coverage_claim(
    run_id: str,
    *,
    pipeline_version: str,
    conn: sqlite3.Connection | None = None,
) -> dict[str, Any] | None:
    """Re-derive one normal run's exact coverage claim from authorized bytes."""

    binding = authorized_run_artifact_binding(run_id)
    if binding is None:
        return None
    status, artifact_bytes = authorized_artifact_bytes(binding["source_artifact_path"])
    if (
        status != FINANCIAL_INTEGRITY_PASS
        or artifact_bytes is None
        or sha256(artifact_bytes).hexdigest() != binding["source_artifact_sha256"]
    ):
        return None
    try:
        artifact = json.loads(artifact_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(artifact, dict) or str(artifact.get("run_id") or "") != str(run_id):
        return None
    selection = artifact.get("candidate_selection")
    selection = selection if isinstance(selection, dict) else {}
    artifact_sector = str(artifact.get("sector") or "").strip().lower()
    artifact_focus = str(artifact.get("market_cap_focus") or "").strip().lower()
    selection_sector = str(selection.get("sector") or "").strip().lower()
    selection_focus = str(selection.get("market_cap_focus") or "").strip().lower()
    sector = selection_sector or artifact_sector
    market_cap_focus = selection_focus or artifact_focus
    source = str(selection.get("source") or "").strip()
    campaign = str(selection.get("coverage_campaign_id") or "").strip() or None
    if not sector or not market_cap_focus:
        return None
    if pipeline_version == "v1":
        effective_as_of = str(artifact.get("as_of_date") or "").strip()
        if (
            not artifact_sector
            or artifact_focus not in V1_ATOMIC_BANDS
            or (selection_sector and selection_sector != artifact_sector)
            or (selection_focus and selection_focus != artifact_focus)
            or not source
            or source.lower().startswith("explicit_tickers")
        ):
            return None
        sector = artifact_sector
        market_cap_focus = artifact_focus
        terminal = v1_terminal_coverage_from_artifact(artifact)
        gate_results = _normalized_ticker_mapping(selection.get("structural_gate_results"))
        cap_classifications = _normalized_ticker_mapping(selection.get("cap_classifications"))
        delta_audit = (
            selection.get("delta_audit") if isinstance(selection.get("delta_audit"), dict) else {}
        )
        carried_verdict_rows = _normalized_ticker_mapping(delta_audit.get("carried_verdicts"))
        history_filter = selection.get("financial_history_filter")
        dispositions: dict[str, str] = {}
        authorized_classifications: dict[str, dict[str, Any]] = {}
        for raw_ticker, raw_disposition in terminal["terminal_candidate_dispositions"].items():
            ticker = str(raw_ticker).strip().upper()
            disposition = str(raw_disposition).strip().upper()
            if disposition == "STRUCTURAL_SCREENED":
                resolved = _authorized_structural_coverage_evidence(
                    ticker=ticker,
                    sector=sector,
                    market_cap_focus=market_cap_focus,
                    evidence={
                        "disposition": disposition,
                        "effective_as_of": effective_as_of,
                        "sector": sector,
                        "market_cap_focus": market_cap_focus,
                        "cap_classification": (
                            cap_classifications.get(ticker)
                            if cap_classifications is not None
                            else None
                        ),
                        "structural_gate_result": (
                            gate_results.get(ticker) if gate_results is not None else None
                        ),
                    },
                )
                if resolved is None:
                    continue
                authorized_classifications[ticker] = resolved
            elif disposition == "CARRIED_VERDICT":
                carried = (
                    carried_verdict_rows.get(ticker) if carried_verdict_rows is not None else None
                )
                if conn is None or not _authorized_carried_verdict_evidence(
                    conn,
                    ticker=ticker,
                    carried=carried,
                ):
                    continue
            elif disposition == "NEEDS_DATA_SPARSE_HISTORY":
                if not _authorized_sparse_history_evidence(
                    ticker=ticker,
                    effective_as_of=effective_as_of,
                    history_filter=history_filter,
                ):
                    continue
            elif disposition == "NEEDS_DATA_FRAMEWORK_EVIDENCE":
                # The v1 runtime removes zero-support packets before product
                # publication. The artifact therefore has no independently
                # replayable packet input for this filter; its own metadata
                # cannot authorize itself as terminal sweep coverage.
                continue
            dispositions[ticker] = disposition

        unknown_cap_tickers: set[str] = set()
        for ticker in unknown_cap_tickers_from_selection(selection):
            if ticker not in dispositions or cap_classifications is None:
                continue
            resolved = authorized_classifications.get(ticker)
            if resolved is None:
                resolved = _independently_authorized_cap_classification(
                    ticker=ticker,
                    effective_as_of=effective_as_of,
                    cap_classification=cap_classifications.get(ticker),
                )
            if resolved is not None and str(resolved.get("cap_source") or "").lower() == "unknown":
                unknown_cap_tickers.add(ticker)
    else:
        dispositions = {}
        unknown_cap_tickers = set(unknown_cap_tickers_from_selection(selection))
        for raw_row in artifact.get("candidate_dispositions") or []:
            if not isinstance(raw_row, dict):
                return None
            ticker = str(raw_row.get("ticker") or "").strip().upper()
            disposition = str(raw_row.get("terminal_state") or "").strip().upper()
            if not ticker or not disposition or ticker in dispositions:
                return None
            dispositions[ticker] = disposition
    return {
        "kind": "RUN_ARTIFACT",
        "run_id": str(run_id),
        "pipeline_version": pipeline_version,
        "sector": sector,
        "market_cap_focus": market_cap_focus,
        "source": source,
        "coverage_campaign_id": campaign,
        "dispositions": dispositions,
        "unknown_cap_tickers": unknown_cap_tickers,
    }


def _bound_coverage_receipt(
    conn: sqlite3.Connection,
    run_id: str,
) -> tuple[dict[str, Any], list[sqlite3.Row]] | None:
    """Read one exact immutable receipt and bind every ledger row to its bytes."""

    _ensure_table(conn)
    rows = conn.execute(
        "SELECT * FROM sector_run_loaded_sets WHERE run_id = ? ORDER BY ticker",
        (str(run_id),),
    ).fetchall()
    if not rows:
        return None
    kinds = {str(row["coverage_authority_kind"] or "").strip().upper() for row in rows}
    paths = {str(row["coverage_authority_path"] or "").strip() for row in rows}
    hashes = {str(row["coverage_authority_sha256"] or "").strip().lower() for row in rows}
    source_run_ids = {str(row["coverage_source_run_id"] or "").strip() or None for row in rows}
    if len(kinds) != 1 or len(paths) != 1 or len(hashes) != 1 or len(source_run_ids) != 1:
        return None
    kind = next(iter(kinds))
    stored_path = next(iter(paths))
    stored_sha256 = next(iter(hashes))
    if (
        kind not in {_ZERO_COST_RECEIPT, _UNKNOWN_CAP_PROJECTION_RECEIPT}
        or re.fullmatch(r"[0-9a-f]{64}", stored_sha256) is None
    ):
        return None
    try:
        expected_path = _coverage_receipt_path(run_id)
    except ValueError:
        return None
    if stored_path != str(expected_path) or expected_path.is_symlink():
        return None
    try:
        descriptor = os.open(
            expected_path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        with os.fdopen(descriptor, "rb") as handle:
            file_stat = os.fstat(handle.fileno())
            if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
                return None
            receipt_bytes = handle.read()
    except OSError:
        return None
    if sha256(receipt_bytes).hexdigest() != stored_sha256:
        return None
    try:
        payload = json.loads(receipt_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    try:
        canonical_bytes = _canonical_coverage_receipt_bytes(payload)
    except (TypeError, ValueError):
        return None
    required_keys = {
        "schema_version",
        "kind",
        "run_id",
        "pipeline_version",
        "sector",
        "market_cap_focus",
        "source",
        "coverage_campaign_id",
        "source_run_id",
        "source_market_cap_focus",
        "rows",
    }
    if (
        receipt_bytes != canonical_bytes
        or set(payload) != required_keys
        or payload.get("schema_version") != _COVERAGE_RECEIPT_SCHEMA_VERSION
        or str(payload.get("kind") or "").strip().upper() != kind
        or str(payload.get("run_id") or "") != str(run_id)
        or str(payload.get("pipeline_version") or "").strip().lower() != "v1"
    ):
        return None
    raw_receipt_rows = payload.get("rows")
    if (
        not isinstance(raw_receipt_rows, list)
        or len(raw_receipt_rows) != len(rows)
        or any(
            not isinstance(raw, dict)
            or set(raw)
            != {
                "ticker",
                "candidate_disposition",
                "coverage_complete",
                "coverage_evidence",
            }
            for raw in raw_receipt_rows
        )
    ):
        return None
    receipt_by_ticker: dict[str, dict[str, Any]] = {}
    for raw in raw_receipt_rows:
        ticker = str(raw.get("ticker") or "").strip().upper()
        if not ticker or ticker in receipt_by_ticker:
            return None
        receipt_by_ticker[ticker] = raw
    if list(receipt_by_ticker) != sorted(receipt_by_ticker):
        return None
    normalized_campaign = str(payload.get("coverage_campaign_id") or "").strip() or None
    normalized_source_run_id = str(payload.get("source_run_id") or "").strip() or None
    if source_run_ids != {normalized_source_run_id}:
        return None
    for row in rows:
        ticker = str(row["ticker"] or "").strip().upper()
        receipt_row = receipt_by_ticker.get(ticker)
        if receipt_row is None:
            return None
        evidence = receipt_row.get("coverage_evidence")
        if evidence is not None and not isinstance(evidence, dict):
            return None
        try:
            evidence_json = (
                json.dumps(evidence, sort_keys=True, separators=(",", ":"), allow_nan=False)
                if evidence is not None
                else None
            )
        except (TypeError, ValueError):
            return None
        evidence_sha256 = (
            sha256(evidence_json.encode("utf-8")).hexdigest() if evidence_json is not None else None
        )
        if (
            str(row["sector"] or "").strip().lower()
            != str(payload.get("sector") or "").strip().lower()
            or str(row["market_cap_focus"] or "").strip().lower()
            != str(payload.get("market_cap_focus") or "").strip().lower()
            or str(row["source"] or "").strip() != str(payload.get("source") or "").strip()
            or (str(row["pipeline_version"] or "").strip().lower() or "v1") != "v1"
            or (str(row["coverage_campaign_id"] or "").strip() or None) != normalized_campaign
            or str(row["candidate_disposition"] or "").strip().upper()
            != str(receipt_row.get("candidate_disposition") or "").strip().upper()
            or bool(row["coverage_complete"]) is not bool(receipt_row.get("coverage_complete"))
            or (row["coverage_evidence_json"] or None) != evidence_json
            or (row["coverage_evidence_sha256"] or None) != evidence_sha256
            or str(row["coverage_authority_kind"] or "").strip().upper() != kind
            or str(row["coverage_authority_path"] or "").strip() != stored_path
            or str(row["coverage_authority_sha256"] or "").strip().lower() != stored_sha256
        ):
            return None
    return payload, rows


def _independently_authorized_cap_classification(
    *,
    ticker: str,
    effective_as_of: str,
    cap_classification: Any,
) -> dict[str, Any] | None:
    """Re-resolve one exact PIT classification without live market data."""

    if not isinstance(cap_classification, dict):
        return None
    try:
        if date.fromisoformat(effective_as_of).isoformat() != effective_as_of:
            return None
        from app.config import get_config
        from app.sector.scan import classify_tickers_for_market_cap

        cfg = get_config()
        independently_resolved = classify_tickers_for_market_cap(
            tickers=[ticker],
            as_of_date=effective_as_of,
            db_path=cfg.db_path,
            pipeline_version="v1",
            allow_live_market_data=False,
            cfg=cfg,
        )[ticker].to_dict()
        if _canonical_coverage_receipt_bytes(independently_resolved) != (
            _canonical_coverage_receipt_bytes(cap_classification)
        ):
            return None
    except Exception:  # noqa: BLE001 - an unreadable cap source reopens coverage
        return None
    return independently_resolved


def _independently_resolved_financial_history_years(
    *,
    ticker: str,
    effective_as_of: str,
) -> int | None:
    """Replay the exact v1 sparse-history query against the local PIT source."""

    try:
        if date.fromisoformat(effective_as_of).isoformat() != effective_as_of:
            return None
        from app.autonomous.sector_runtime import (
            REPORTABLE_FINANCIAL_HISTORY_LINE_ITEMS,
        )
        from app.config import get_config
        from app.util.financial_data_access import companyfacts_rows

        db_path = Path(get_config().db_path)
        if not db_path.is_file():
            return None
        database_uri = f"{db_path.resolve().as_uri()}?mode=ro"
        with sqlite3.connect(database_uri, uri=True) as source_conn:
            required_columns = {
                "ticker",
                "fiscal_year",
                "period_type",
                "period_end",
                "filed_date",
                "line_item",
                "value",
                "accession",
                "source_url",
            }
            columns = {
                str(row[1])
                for row in source_conn.execute("PRAGMA table_info(companyfacts_facts)").fetchall()
            }
            if required_columns - columns:
                return None
            rows = companyfacts_rows(
                source_conn,
                ticker,
                columns=("fiscal_year", "period_end", "filed_date"),
                period_types=("FY",),
                line_items=REPORTABLE_FINANCIAL_HISTORY_LINE_ITEMS,
                as_of_date=effective_as_of,
                require_filed_asof=True,
                value_not_null=True,
                order_by="fiscal_year DESC",
            )
    except (OSError, sqlite3.Error, TypeError, ValueError):
        return None
    return len(
        {
            row[0]
            for row in rows
            if isinstance(row[0], int) and str(row[1] or "") <= str(row[2] or "")
        }
    )


def _authorized_sparse_history_evidence(
    *,
    ticker: str,
    effective_as_of: str,
    history_filter: Any,
) -> bool:
    """Require a v1 sparse terminal state to replay from PIT CompanyFacts."""

    if not isinstance(history_filter, dict):
        return False
    from app.autonomous.sector_runtime import REPORTABLE_FINANCIAL_HISTORY_MIN_ROWS

    minimum_rows = history_filter.get("minimum_rows")
    year_counts = _normalized_ticker_mapping(history_filter.get("year_counts"))
    claimed_years = year_counts.get(ticker) if year_counts is not None else None
    if (
        str(history_filter.get("status") or "").strip().upper()
        != "FILTERED_SPARSE_FINANCIAL_HISTORY"
        or type(minimum_rows) is not int
        or minimum_rows != REPORTABLE_FINANCIAL_HISTORY_MIN_ROWS
        or type(claimed_years) is not int
        or claimed_years < 0
        or ticker not in _normalized_values(history_filter.get("excluded_tickers"))
    ):
        return False
    independently_resolved = _independently_resolved_financial_history_years(
        ticker=ticker,
        effective_as_of=effective_as_of,
    )
    return (
        independently_resolved is not None
        and independently_resolved == claimed_years
        and independently_resolved < minimum_rows
    )


def _authorized_carried_verdict_evidence(
    conn: sqlite3.Connection,
    *,
    ticker: str,
    carried: Any,
) -> bool:
    """Bind one carried terminal state to the exact current outcome row."""

    if not isinstance(carried, dict):
        return False
    current = carried_verdicts(
        conn,
        [ticker],
        pipeline_version="v1",
    ).get(ticker)
    return current is not None and dict(carried) == current


def _authorized_structural_coverage_evidence(
    *,
    ticker: str,
    sector: str,
    market_cap_focus: str,
    evidence: Any,
) -> dict[str, Any] | None:
    """Validate and replay one structural-screen coverage proof."""

    if not isinstance(evidence, dict):
        return None
    effective_as_of = str(evidence.get("effective_as_of") or "").strip()
    if (
        str(evidence.get("disposition") or "").strip().upper() != "STRUCTURAL_SCREENED"
        or str(evidence.get("sector") or "").strip().lower() != sector
        or str(evidence.get("market_cap_focus") or "").strip().lower() != market_cap_focus
        or not effective_as_of
        or not isinstance(evidence.get("cap_classification"), dict)
    ):
        return None
    from app.autonomous.structural_gate import (
        QUARANTINE_STRUCTURAL_PREFIX,
        STRUCTURAL_CODES,
        evaluate_structural_gate,
    )
    from app.config import get_config

    result = evidence.get("structural_gate_result")
    if not isinstance(result, dict):
        return None
    triggered = result.get("triggered_codes")
    reasons = result.get("reasons")
    details = result.get("details")
    if (
        str(result.get("ticker") or "").strip().upper() != ticker
        or str(result.get("as_of_date") or "").strip() != effective_as_of
        or result.get("quarantined") is not True
        or result.get("excluded_error") is not False
        or not isinstance(triggered, list)
        or not triggered
        or len(triggered) != len(set(triggered))
        or any(code not in STRUCTURAL_CODES for code in triggered)
        or result.get("degraded_codes") not in ([], None)
        or reasons != [f"{QUARANTINE_STRUCTURAL_PREFIX}:{code}" for code in triggered]
        or not isinstance(details, dict)
        or set(details) != set(triggered)
    ):
        return None
    independently_resolved = _independently_authorized_cap_classification(
        ticker=ticker,
        effective_as_of=effective_as_of,
        cap_classification=evidence["cap_classification"],
    )
    if independently_resolved is None:
        return None
    try:
        cfg = get_config()
        rederived = evaluate_structural_gate(
            ticker,
            as_of_date=effective_as_of,
            price=independently_resolved.get("price_used"),
            market_cap_mm=independently_resolved.get("market_cap_mm"),
            db_path=cfg.db_path,
        ).to_dict()
        if _canonical_coverage_receipt_bytes(rederived) != (
            _canonical_coverage_receipt_bytes(result)
        ):
            return None
    except Exception:  # noqa: BLE001 - an unreadable gate source reopens coverage
        return None
    return independently_resolved


def _zero_cost_evidence_is_authorized(
    conn: sqlite3.Connection,
    *,
    ticker: str,
    disposition: str,
    sector: str,
    market_cap_focus: str,
    evidence: Any,
) -> bool:
    if disposition == "STRUCTURAL_SCREENED":
        return (
            _authorized_structural_coverage_evidence(
                ticker=ticker,
                sector=sector,
                market_cap_focus=market_cap_focus,
                evidence=evidence,
            )
            is not None
        )
    if not isinstance(evidence, dict):
        return False
    effective_as_of = str(evidence.get("effective_as_of") or "").strip()
    if (
        str(evidence.get("disposition") or "").strip().upper() != disposition
        or str(evidence.get("sector") or "").strip().lower() != sector
        or str(evidence.get("market_cap_focus") or "").strip().lower() != market_cap_focus
        or not effective_as_of
        or not isinstance(evidence.get("cap_classification"), dict)
    ):
        return False
    try:
        datetime.fromisoformat(effective_as_of[:10])
    except ValueError:
        return False
    if disposition != "CARRIED_VERDICT":
        return False
    carried = evidence.get("carried_verdict")
    return _authorized_carried_verdict_evidence(
        conn,
        ticker=ticker,
        carried=carried,
    )


def _authorized_coverage_claim(
    run_id: str,
    *,
    pipeline_version: str,
    conn: sqlite3.Connection | None,
    resolving: frozenset[str] = frozenset(),
) -> dict[str, Any] | None:
    normalized_run_id = str(run_id or "").strip()
    if not normalized_run_id or normalized_run_id in resolving:
        return None
    bound: tuple[dict[str, Any], list[sqlite3.Row]] | None = None
    receipt_required = _run_id_requires_coverage_receipt(normalized_run_id)
    if receipt_required and (pipeline_version != "v1" or conn is None):
        return None
    if pipeline_version == "v1" and conn is not None:
        _ensure_table(conn)
        receipt_required = receipt_required or (
            conn.execute(
                """
                SELECT 1
                FROM sector_run_loaded_sets
                WHERE run_id = ?
                  AND (
                    coverage_authority_kind IS NOT NULL
                    OR coverage_authority_path IS NOT NULL
                    OR coverage_authority_sha256 IS NOT NULL
                    OR coverage_source_run_id IS NOT NULL
                  )
                LIMIT 1
                """,
                (normalized_run_id,),
            ).fetchone()
            is not None
        )
        if receipt_required:
            bound = _bound_coverage_receipt(conn, normalized_run_id)
            if bound is None:
                # Reserved synthetic namespaces and any ledger row that ever
                # claims receipt authority can never fall back to a same-id
                # product artifact.
                return None
    if not receipt_required:
        return _authorized_artifact_coverage_claim(
            normalized_run_id,
            pipeline_version=pipeline_version,
            conn=conn,
        )
    assert bound is not None
    payload, _ledger_rows = bound
    kind = str(payload["kind"]).upper()
    sector = str(payload["sector"]).strip().lower()
    band = str(payload["market_cap_focus"]).strip().lower()
    campaign = str(payload.get("coverage_campaign_id") or "").strip() or None
    dispositions = {
        str(row["ticker"]).strip().upper(): str(row["candidate_disposition"]).strip().upper()
        for row in payload["rows"]
    }
    if kind == _ZERO_COST_RECEIPT:
        if (
            payload.get("source_run_id") is not None
            or payload.get("source_market_cap_focus") is not None
            or band not in V1_ATOMIC_BANDS
            or any(
                not _zero_cost_evidence_is_authorized(
                    conn,
                    ticker=str(row["ticker"]).strip().upper(),
                    disposition=str(row["candidate_disposition"]).strip().upper(),
                    sector=sector,
                    market_cap_focus=band,
                    evidence=row.get("coverage_evidence"),
                )
                for row in payload["rows"]
            )
        ):
            return None
        unknown: set[str] = set()
        for row in payload["rows"]:
            ticker = str(row["ticker"]).strip().upper()
            disposition = str(row["candidate_disposition"]).strip().upper()
            evidence = row.get("coverage_evidence") or {}
            cap_classification = evidence.get("cap_classification")
            if disposition == "STRUCTURAL_SCREENED":
                # The structural validator above already proved this exact
                # classification against an independent offline resolution.
                resolved = cap_classification
            else:
                resolved = _independently_authorized_cap_classification(
                    ticker=ticker,
                    effective_as_of=str(evidence.get("effective_as_of") or "").strip(),
                    cap_classification=cap_classification,
                )
            if resolved is not None and str(resolved.get("cap_source") or "").lower() == "unknown":
                unknown.add(ticker)
        return {
            "kind": kind,
            "run_id": normalized_run_id,
            "pipeline_version": "v1",
            "sector": sector,
            "market_cap_focus": band,
            "source": str(payload.get("source") or "").strip(),
            "coverage_campaign_id": campaign,
            "dispositions": dispositions,
            "unknown_cap_tickers": unknown,
        }
    source_run_id = str(payload.get("source_run_id") or "").strip()
    source_band = str(payload.get("source_market_cap_focus") or "").strip().lower()
    if (
        not source_run_id
        or source_band not in V1_ATOMIC_BANDS
        or band not in V1_ATOMIC_BANDS
        or band == source_band
        or normalized_run_id != f"{source_run_id}__unknown_cap__{band}"
    ):
        return None
    source_claim = _authorized_coverage_claim(
        source_run_id,
        pipeline_version="v1",
        conn=conn,
        resolving=resolving | {normalized_run_id},
    )
    if (
        source_claim is None
        or source_claim["kind"] == _UNKNOWN_CAP_PROJECTION_RECEIPT
        or source_claim["sector"] != sector
        or source_claim["market_cap_focus"] != source_band
        or source_claim["coverage_campaign_id"] != campaign
        or any(
            ticker not in source_claim["unknown_cap_tickers"]
            or source_claim["dispositions"].get(ticker) != disposition
            for ticker, disposition in dispositions.items()
        )
    ):
        return None
    return {
        "kind": kind,
        "run_id": normalized_run_id,
        "pipeline_version": "v1",
        "sector": sector,
        "market_cap_focus": band,
        "source": str(payload.get("source") or "").strip(),
        "coverage_campaign_id": campaign,
        "dispositions": dispositions,
        "unknown_cap_tickers": set(dispositions),
    }


def _authorized_terminal_dispositions(
    run_id: str,
    *,
    pipeline_version: str,
    conn: sqlite3.Connection | None = None,
) -> dict[str, str] | None:
    """Re-derive exact terminal states from a run artifact or coverage receipt."""

    claim = _authorized_coverage_claim(
        run_id,
        pipeline_version=pipeline_version,
        conn=conn,
    )
    if claim is None:
        return None
    if conn is not None and pipeline_version == "v1":
        _ensure_table(conn)
        rows = conn.execute(
            """
            SELECT sector, market_cap_focus, source, coverage_campaign_id, pipeline_version
            FROM sector_run_loaded_sets
            WHERE run_id = ?
            """,
            (str(run_id),),
        ).fetchall()
        if any(
            str(row["sector"] or "").strip().lower() != claim["sector"]
            or str(row["market_cap_focus"] or "").strip().lower() != claim["market_cap_focus"]
            or str(row["source"] or "").strip() != claim["source"]
            or (str(row["coverage_campaign_id"] or "").strip() or None)
            != claim["coverage_campaign_id"]
            or (str(row["pipeline_version"] or "").strip().lower() or "v1")
            != claim["pipeline_version"]
            for row in rows
        ):
            return None
    return dict(claim["dispositions"])


def swept_tickers_for_band(
    conn: sqlite3.Connection,
    band: str,
    sector: str | None = None,
    *,
    pipeline_version: str = "v1",
    coverage_campaign_id: str | None = None,
) -> set[str]:
    """Tickers whose latest sweep-sourced state is complete for the band/sector.

    Explicit-ticker probe runs are excluded: probing one name is not sweep
    coverage of it. Legacy v1 rows without a terminal disposition are also
    excluded because loader presence is not proof of review. A later v2
    NEEDS_DATA or READY_FOR_UNDERWRITING state reopens the name for
    ``--only-unswept`` even if an older run loaded it.
    """
    normalized_pipeline = str(pipeline_version or "v1").strip().lower()
    normalized_campaign = str(coverage_campaign_id or "").strip() or None
    if normalized_pipeline not in {"v1", "v2"}:
        raise ValueError("pipeline_version must be v1 or v2")
    if not financial_integrity_manifest_is_usable():
        return set()
    _ensure_table(conn)
    if normalized_pipeline == "v1":
        complete_dispositions = set(_V1_COVERAGE_COMPLETE_DISPOSITIONS)
        if normalized_campaign:
            # A registered campaign is a fresh-review namespace. A manually
            # inserted/carried historical verdict can never close it.
            complete_dispositions.discard("CARRIED_VERDICT")
        marks = ",".join("?" for _ in complete_dispositions)
        clauses = [
            "market_cap_focus = ?",
            "LOWER(source) NOT LIKE 'explicit_tickers%'",
            "LOWER(COALESCE(pipeline_version, 'v1')) != 'v2'",
        ]
        params: list[Any] = [str(band).strip().lower()]
        if sector:
            clauses.append("LOWER(sector) = ?")
            params.append(str(sector).strip().lower())
        if normalized_campaign:
            clauses.append("coverage_campaign_id = ?")
            params.append(normalized_campaign)
        params.extend(sorted(complete_dispositions))
        rows = conn.execute(
            f"""
            SELECT ticker, run_id, candidate_disposition
            FROM (
                SELECT ticker, run_id, coverage_complete, candidate_disposition,
                       ROW_NUMBER() OVER (
                           PARTITION BY ticker
                           ORDER BY loaded_at DESC, id DESC
                       ) AS coverage_rank
                FROM sector_run_loaded_sets
                WHERE {" AND ".join(clauses)}
            )
            WHERE coverage_rank = 1
              AND coverage_complete = 1
              AND candidate_disposition IN ({marks})
            """,
            params,
        ).fetchall()
        run_dispositions: dict[str, dict[str, str] | None] = {}
        swept: set[str] = set()
        for row in rows:
            run_id = str(row["run_id"])
            if run_id not in run_dispositions:
                run_dispositions[run_id] = _authorized_terminal_dispositions(
                    run_id,
                    pipeline_version="v1",
                    conn=conn,
                )
            exact = run_dispositions[run_id]
            ticker = str(row["ticker"]).strip().upper()
            disposition = str(row["candidate_disposition"] or "").strip().upper()
            if exact is not None and exact.get(ticker) == disposition:
                swept.add(ticker)
        return swept

    clauses = [
        "market_cap_focus = ?",
        "LOWER(source) NOT LIKE 'explicit_tickers%'",
        "pipeline_version = 'v2'",
    ]
    params: list[Any] = [str(band).strip().lower()]
    if sector:
        clauses.append("LOWER(sector) = ?")
        params.append(str(sector).strip().lower())
    if normalized_campaign:
        clauses.append("coverage_campaign_id = ?")
        params.append(normalized_campaign)
    rows = conn.execute(
        """
        SELECT ticker, run_id, candidate_disposition
        FROM (
            SELECT ticker, run_id, coverage_complete, candidate_disposition,
                   ROW_NUMBER() OVER (
                       PARTITION BY ticker
                       ORDER BY loaded_at DESC, id DESC
                   ) AS coverage_rank
            FROM sector_run_loaded_sets
            WHERE """
        + " AND ".join(clauses)
        + """
        )
        WHERE coverage_rank = 1 AND coverage_complete = 1
        """,
        params,
    ).fetchall()
    run_dispositions = {}
    swept = set()
    for row in rows:
        run_id = str(row["run_id"])
        if run_id not in run_dispositions:
            run_dispositions[run_id] = _authorized_terminal_dispositions(
                run_id,
                pipeline_version="v2",
                conn=conn,
            )
        exact = run_dispositions[run_id]
        ticker = str(row["ticker"]).strip().upper()
        disposition = str(row["candidate_disposition"] or "").strip().upper()
        if (
            exact is not None
            and exact.get(ticker) == disposition
            and disposition in _V2_COVERAGE_COMPLETE_DISPOSITIONS
        ):
            swept.add(ticker)
    return swept


def carried_verdicts(
    conn: sqlite3.Connection,
    tickers: list[str],
    *,
    pipeline_version: str = "v1",
    sector: str | None = None,
) -> dict[str, dict[str, Any]]:
    """ticker -> latest autonomous review verdict from the ticker_outcomes ledger.

    Only real review grades from autonomous runs count — backtest rows
    (h1_* / backtest_*) never do.
    """
    if not tickers or not financial_integrity_manifest_is_usable():
        return {}
    normalized_pipeline = str(pipeline_version or "v1").strip().lower()
    if normalized_pipeline not in {"v1", "v2"}:
        raise ValueError("pipeline_version must be v1 or v2")
    marks = ",".join("?" for _ in tickers)
    provenance_clause = (
        "AND LOWER(COALESCE(pipeline_version, 'v1')) != 'v2'"
        if normalized_pipeline == "v1"
        else "AND pipeline_version = 'v2' AND LOWER(COALESCE(source_sector, '')) = ?"
    )
    params: list[Any] = [str(t).upper() for t in tickers]
    if normalized_pipeline == "v2":
        params.append(str(sector or "").strip().lower())
    try:
        rows = conn.execute(
            f"""
        SELECT *
        FROM ticker_outcomes
        WHERE ticker IN ({marks})
          AND run_id LIKE 'autonomous_%'
          AND grade IS NOT NULL
          {provenance_clause}
        ORDER BY updated_at, id
        """,
            params,
        ).fetchall()
    except sqlite3.OperationalError:
        return {}
    carried: dict[str, dict[str, Any]] = {}
    for row in rows:  # later rows overwrite -> latest wins
        grade = str(row["grade"] or "").upper()
        ticker = str(row["ticker"])
        if not outcome_row_is_decision_eligible(row):
            # Latest literal provenance still wins: an unaudited/invalid newer
            # row clears an older verdict instead of resurrecting it.
            carried.pop(ticker, None)
            continue
        if grade not in _REVIEW_GRADES:
            carried.pop(ticker, None)
            continue
        row_pipeline_version = str(row["pipeline_version"] or "").lower()
        decision_basis = str(row["decision_basis"] or "").upper()
        disposition = str(row["candidate_disposition"] or "").upper()
        if row_pipeline_version == "v2" and (
            decision_basis not in {"UNDERWRITING", "VALIDATED_UNDERWRITING"}
            or disposition != "UNDERWRITTEN"
            or grade == "DATA_INCOMPLETE"
        ):
            # A newer screen-only v2 state reopens the name even when an older
            # autonomous run underwrote it. Latest literal provenance wins.
            carried.pop(ticker, None)
            continue
        carried[ticker] = {
            "outcome_id": row["id"],
            "verdict": grade,
            "run_id": row["run_id"],
            "as_of_date": row["as_of_date"],
            "decision_basis": row["decision_basis"],
            "candidate_disposition": row["candidate_disposition"],
            "source_artifact_path": row["source_artifact_path"],
            "source_artifact_sha256": row["source_artifact_sha256"],
            "source_decision_fingerprint": row["source_decision_fingerprint"],
            "financial_integrity_fingerprint": row["financial_integrity_fingerprint"],
        }
    return carried


def _latest_v1_failed_review_tickers(
    conn: sqlite3.Connection,
    *,
    band: str,
    sector: str | None,
    tickers: list[str],
    coverage_campaign_id: str | None = None,
) -> set[str]:
    """Return same-cell names whose latest v1 review state requires retry.

    Watchlist population can snapshot a ticker outcome after a memo fallback.
    That later outcome must not turn the failed review into a carried verdict;
    the explicit same-sector, same-band coverage ledger remains authoritative.
    """
    normalized = _normalized_values(tickers)
    if not sector or not normalized:
        return set()
    _ensure_table(conn)
    ticker_marks = ",".join("?" for _ in normalized)
    state_marks = ",".join("?" for _ in _V1_CARRIED_SUPPRESSING_DISPOSITIONS)
    campaign_clause = ""
    campaign_params: list[str] = []
    normalized_campaign = str(coverage_campaign_id or "").strip()
    if normalized_campaign:
        campaign_clause = "AND coverage_campaign_id = ?"
        campaign_params.append(normalized_campaign)
    rows = conn.execute(
        f"""
        SELECT ticker
        FROM (
            SELECT ticker, coverage_complete, candidate_disposition,
                   ROW_NUMBER() OVER (
                       PARTITION BY ticker
                       ORDER BY loaded_at DESC, id DESC
                   ) AS coverage_rank
            FROM sector_run_loaded_sets
            WHERE market_cap_focus = ?
              AND LOWER(sector) = ?
              AND LOWER(source) NOT LIKE 'explicit_tickers%'
              AND LOWER(COALESCE(pipeline_version, 'v1')) != 'v2'
              AND ticker IN ({ticker_marks})
              {campaign_clause}
        )
        WHERE coverage_rank = 1
          AND coverage_complete = 0
          AND candidate_disposition IN ({state_marks})
        """,
        [
            str(band).strip().lower(),
            str(sector).strip().lower(),
            *normalized,
            *campaign_params,
            *sorted(_V1_CARRIED_SUPPRESSING_DISPOSITIONS),
        ],
    ).fetchall()
    return {str(row["ticker"]).strip().upper() for row in rows}


def split_unswept(
    conn: sqlite3.Connection,
    *,
    band: str,
    sector: str | None = None,
    selected_tickers: list[str],
    pipeline_version: str = "v1",
    coverage_campaign_id: str | None = None,
    allow_carried_verdicts: bool = True,
) -> dict[str, Any]:
    """Split a freshly resolved selection into swept / carried / to_review."""
    swept_union = swept_tickers_for_band(
        conn,
        band,
        sector,
        pipeline_version=pipeline_version,
        coverage_campaign_id=coverage_campaign_id,
    )
    selected = [str(t).upper() for t in selected_tickers]
    swept = [t for t in selected if t in swept_union]
    unswept = [t for t in selected if t not in swept_union]
    carried = (
        carried_verdicts(
            conn,
            unswept,
            pipeline_version=pipeline_version,
            sector=sector,
        )
        if allow_carried_verdicts
        else {}
    )
    if str(pipeline_version or "v1").strip().lower() == "v1":
        retry_required = _latest_v1_failed_review_tickers(
            conn,
            band=band,
            sector=sector,
            tickers=unswept,
            coverage_campaign_id=coverage_campaign_id,
        )
        carried = {
            ticker: verdict for ticker, verdict in carried.items() if ticker not in retry_required
        }
    to_review = [t for t in unswept if t not in carried]
    return {
        "selected": selected,
        "swept": swept,
        "unswept": unswept,
        "carried": carried,
        "to_review": to_review,
    }


def split_loaded_coverage(
    conn: sqlite3.Connection,
    *,
    band: str,
    sector: str | None = None,
    loaded_tickers: list[str],
    pipeline_version: str = "v1",
    coverage_campaign_id: str | None = None,
) -> dict[str, list[str]]:
    """Split the pre-gate loaded set into covered and uncovered names.

    ``split_unswept`` deliberately works on post-gate selected names because
    only those can consume LLM review.  Coverage closure needs the wider
    pre-gate loaded set so a structural screen or carried verdict is still an
    explicit terminal touch rather than a silently missing ticker.
    """
    swept_union = swept_tickers_for_band(
        conn,
        band,
        sector,
        pipeline_version=pipeline_version,
        coverage_campaign_id=coverage_campaign_id,
    )
    loaded = list(
        dict.fromkeys(
            str(ticker).strip().upper() for ticker in loaded_tickers if str(ticker).strip()
        )
    )
    return {
        "loaded": loaded,
        "covered": [ticker for ticker in loaded if ticker in swept_union],
        "uncovered": [ticker for ticker in loaded if ticker not in swept_union],
    }


def record_zero_cost_coverage(
    conn: sqlite3.Connection,
    *,
    sector: str,
    market_cap_focus: str,
    source: str,
    tickers: list[str],
    candidate_dispositions: dict[str, str],
    pipeline_version: str = "v1",
    coverage_campaign_id: str | None = None,
    coverage_evidence_by_ticker: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Close pre-gate coverage when a delta needs no fresh LLM review.

    A cell containing only structural-screened or carried names legitimately
    exits at $0.  Prior to this writer such a cell returned before the loaded
    set was recorded, so the same names remained permanently uncovered.  The
    content-derived run id makes retries idempotent without pretending an LLM
    sector artifact exists.
    """
    normalized_dispositions = {
        str(ticker).strip().upper(): str(state or "").strip().upper()
        for ticker, state in candidate_dispositions.items()
        if str(ticker).strip()
    }
    normalized = sorted(
        dict.fromkeys(
            str(ticker).strip().upper()
            for ticker in tickers
            if str(ticker).strip() and str(ticker).strip().upper() in normalized_dispositions
        )
    )
    if not normalized:
        return {
            "run_id": None,
            "rows_inserted": 0,
            "tickers": [],
            "candidate_dispositions": {},
            "coverage_evidence_sha256": {},
            "coverage_campaign_id": str(coverage_campaign_id or "").strip() or None,
        }
    normalized_pipeline = str(pipeline_version or "v1").strip().lower()
    if normalized_pipeline != "v1":
        raise ValueError("zero-cost Classic coverage receipts require pipeline_version v1")
    if any(
        normalized_dispositions[ticker] not in {"STRUCTURAL_SCREENED", "CARRIED_VERDICT"}
        for ticker in normalized
    ):
        raise ValueError("zero-cost coverage can only close structural or carried dispositions")
    normalized_evidence = {
        str(ticker).strip().upper(): value
        for ticker, value in (coverage_evidence_by_ticker or {}).items()
        if str(ticker).strip()
        and str(ticker).strip().upper() in normalized
        and isinstance(value, dict)
    }
    if set(normalized_evidence) != set(normalized):
        raise ValueError("zero-cost coverage requires exact evidence for every ticker")
    normalized_sector = str(sector).strip().lower()
    normalized_focus = str(market_cap_focus).strip().lower()
    normalized_source = str(source).strip()
    normalized_campaign = str(coverage_campaign_id or "").strip() or None
    if (
        _SAFE_RUN_COMPONENT.fullmatch(normalized_sector) is None
        or normalized_focus not in V1_ATOMIC_BANDS
        or not normalized_source
    ):
        raise ValueError("zero-cost coverage requires a canonical sector, band, and source")
    fingerprint_payload = json.dumps(
        {
            "pipeline_version": normalized_pipeline,
            "sector": normalized_sector,
            "market_cap_focus": normalized_focus,
            "source": normalized_source,
            "tickers": normalized,
            "candidate_dispositions": {
                ticker: normalized_dispositions[ticker] for ticker in normalized
            },
            "coverage_evidence": normalized_evidence,
            "coverage_campaign_id": normalized_campaign,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    fingerprint = sha256(fingerprint_payload.encode("utf-8")).hexdigest()[:16]
    run_id = (
        f"coverage_only_{normalized_pipeline}_{normalized_sector}_{normalized_focus}_{fingerprint}"
    )
    receipt = _write_coverage_receipt(
        {
            "schema_version": _COVERAGE_RECEIPT_SCHEMA_VERSION,
            "kind": _ZERO_COST_RECEIPT,
            "run_id": run_id,
            "pipeline_version": normalized_pipeline,
            "sector": normalized_sector,
            "market_cap_focus": normalized_focus,
            "source": normalized_source,
            "coverage_campaign_id": normalized_campaign,
            "source_run_id": None,
            "source_market_cap_focus": None,
            "rows": [
                {
                    "ticker": ticker,
                    "candidate_disposition": normalized_dispositions[ticker],
                    "coverage_complete": True,
                    "coverage_evidence": normalized_evidence[ticker],
                }
                for ticker in normalized
            ],
        }
    )
    inserted = record_loaded_set(
        conn,
        run_id=run_id,
        sector=sector,
        market_cap_focus=market_cap_focus,
        source=source,
        tickers=normalized,
        pipeline_version=normalized_pipeline,
        candidate_dispositions=normalized_dispositions,
        coverage_campaign_id=coverage_campaign_id,
        coverage_evidence_by_ticker=normalized_evidence,
        coverage_authority_kind=receipt["kind"],
        coverage_authority_path=receipt["path"],
        coverage_authority_sha256=receipt["sha256"],
    )
    return {
        "run_id": run_id,
        "rows_inserted": inserted,
        "tickers": normalized,
        "candidate_dispositions": {
            ticker: normalized_dispositions[ticker] for ticker in normalized
        },
        "coverage_evidence_sha256": {
            ticker: sha256(
                json.dumps(
                    normalized_evidence[ticker],
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            for ticker in normalized
            if ticker in normalized_evidence
        },
        "coverage_campaign_id": str(coverage_campaign_id or "").strip() or None,
    }


def band_delta_report(
    band: str,
    *,
    sectors: list[str] | None = None,
    as_of: str | None = None,
    synced_after: str | None = None,
    pipeline_version: str | None = None,
    cap_classification_caches: dict[str, dict[str, Any]] | None = None,
    coverage_campaign_id: str | None = None,
    allow_carried_verdicts: bool = True,
    target_tickers: list[str] | None = None,
) -> dict[str, Any]:
    """Per-sector unswept counts for the band — zero LLM spend.

    Runs the same unbounded membership/cap loader the sweep uses, subtracts
    prior coverage, then applies the structural gate to uncovered names only.
    Covered names do not need filing-text re-screening merely to prove
    membership closure. Registrants first seen after `synced_after` are
    cross-checked into the delta.
    """
    from app.autonomous.sector_candidates import (
        AcceptedCensusRunAuthority,
        _apply_structural_gate,
        resolve_sector_candidate_tickers,
    )
    from app.config import (
        canonical_market_cap_focus,
        resolve_autonomous_sector_pipeline_version,
    )
    from app.db import get_db

    resolved_pipeline_version = resolve_autonomous_sector_pipeline_version(
        band,
        pipeline_version,
    )
    require_accepted_census = (
        resolved_pipeline_version == "v2" and canonical_market_cap_focus(band) == "large_and_mega"
    )
    accepted_census_authority = AcceptedCensusRunAuthority() if require_accepted_census else None
    sector_list = list(sectors) if sectors else list(CANONICAL_SWEEP_SECTORS)
    target_scope = set(_normalized_values(target_tickers)) if target_tickers is not None else None
    newly_synced: set[str] = set()
    with get_db() as conn:
        if synced_after:
            try:
                newly_synced = {
                    str(row["primary_ticker"]).upper()
                    for row in conn.execute(
                        "SELECT primary_ticker FROM sec_registrants "
                        "WHERE primary_ticker IS NOT NULL AND first_seen_at > ?",
                        (synced_after,),
                    ).fetchall()
                }
            except sqlite3.OperationalError:
                newly_synced = set()

    per_sector: dict[str, Any] = {}
    totals = {
        "loaded": 0,
        "selected": 0,
        "structural_screened": 0,
        "gate_errors": 0,
        "covered_loaded": 0,
        "coverage_uncovered": 0,
        "swept": 0,
        "carried": 0,
        "to_review": 0,
    }
    cross_check_violations: list[str] = []
    for sector in sector_list:
        selection = resolve_sector_candidate_tickers(
            sector=sector,
            explicit_tickers=None,
            market_cap_focus=band,
            max_candidates=None,
            as_of_date=as_of,
            pipeline_version=resolved_pipeline_version,
            filing_risk_use_llm=False,
            allow_live_market_data=False,
            coverage_only=True,
            cap_classification_cache=(
                cap_classification_caches.setdefault(sector, {})
                if cap_classification_caches is not None
                else None
            ),
            require_accepted_census=require_accepted_census,
            accepted_census_authority=accepted_census_authority,
        )
        scoped_loaded_tickers = [
            ticker
            for ticker in selection.loaded_tickers
            if target_scope is None or ticker in target_scope
        ]
        with get_db() as conn:
            loaded_split = split_loaded_coverage(
                conn,
                band=band,
                sector=sector,
                loaded_tickers=scoped_loaded_tickers,
                pipeline_version=resolved_pipeline_version,
                coverage_campaign_id=coverage_campaign_id,
            )
        warnings = [str(item) for item in (getattr(selection, "warnings", []) or [])]
        gate_results: dict[str, Any] = {}
        admitted_uncovered = list(loaded_split["uncovered"])
        if resolved_pipeline_version == "v1" and admitted_uncovered:
            admitted_uncovered, _gated, gate_results = _apply_structural_gate(
                admitted_uncovered,
                as_of_date=str(as_of or datetime.now(timezone.utc).date().isoformat()),
                cap_classifications=dict(getattr(selection, "cap_classifications", {}) or {}),
                db_path=None,
                warnings=warnings,
                exclude=True,
                include_evidence=True,
            )
        admitted_uncovered_set = set(admitted_uncovered)
        covered_loaded_set = set(loaded_split["covered"])
        selected_for_split = [
            ticker
            for ticker in loaded_split["loaded"]
            if ticker in covered_loaded_set or ticker in admitted_uncovered_set
        ]
        with get_db() as conn:
            split = split_unswept(
                conn,
                band=band,
                sector=sector,
                selected_tickers=selected_for_split,
                pipeline_version=resolved_pipeline_version,
                coverage_campaign_id=coverage_campaign_id,
                allow_carried_verdicts=allow_carried_verdicts,
            )
        if newly_synced:
            for ticker in split["selected"]:
                if ticker in newly_synced and ticker in split["swept"]:
                    cross_check_violations.append(f"{sector}:{ticker}")
        structural_screened = sorted(
            ticker
            for ticker, result in gate_results.items()
            if bool((result or {}).get("quarantined"))
        )
        gate_errors = sorted(
            ticker
            for ticker, result in gate_results.items()
            if bool((result or {}).get("excluded_error"))
        )
        source_errors = [
            warning for warning in warnings if warning.startswith("SECTOR_CANDIDATE_SOURCE_FAILED:")
        ]
        cap_classifications = dict(getattr(selection, "cap_classifications", {}) or {})
        unknown_cap_tickers = sorted(
            ticker
            for ticker in loaded_split["loaded"]
            if str((cap_classifications.get(ticker) or {}).get("cap_source")) == "unknown"
        )
        per_sector[sector] = {
            "source": str(getattr(selection, "source", "unknown")),
            "source_status": "ERROR" if source_errors else "OK",
            "gate_scope": "coverage_uncovered_only",
            "gate_evaluated_tickers": loaded_split["uncovered"],
            "loaded": len(loaded_split["loaded"]),
            "loaded_tickers": loaded_split["loaded"],
            "selected": len(split["selected"]),
            "selected_semantics": "previously_covered_or_gate_admitted_uncovered",
            "selected_tickers": split["selected"],
            "structural_screened": len(structural_screened),
            "structural_screened_tickers": structural_screened,
            "structural_quarantined_tickers": structural_screened,
            "gate_errors": len(gate_errors),
            "gate_error_tickers": gate_errors,
            "unknown_cap_tickers": unknown_cap_tickers,
            "covered_loaded": len(loaded_split["covered"]),
            "coverage_uncovered": len(loaded_split["uncovered"]),
            "coverage_uncovered_tickers": loaded_split["uncovered"],
            "swept": len(split["swept"]),
            "swept_tickers": split["swept"],
            "carried": {t: v["verdict"] for t, v in split["carried"].items()},
            "to_review": len(split["to_review"]),
            "to_review_tickers": split["to_review"],
            "source_errors": source_errors,
            "warnings": warnings,
        }
        totals["loaded"] += len(loaded_split["loaded"])
        totals["selected"] += len(split["selected"])
        totals["structural_screened"] += len(structural_screened)
        totals["gate_errors"] += len(gate_errors)
        totals["covered_loaded"] += len(loaded_split["covered"])
        totals["coverage_uncovered"] += len(loaded_split["uncovered"])
        totals["swept"] += len(split["swept"])
        totals["carried"] += len(split["carried"])
        totals["to_review"] += len(split["to_review"])
    return {
        "band": str(band).strip().lower(),
        "pipeline_version": resolved_pipeline_version,
        "coverage_campaign_id": str(coverage_campaign_id or "").strip() or None,
        "allow_carried_verdicts": bool(allow_carried_verdicts),
        "target_tickers": (sorted(target_scope) if target_scope is not None else None),
        "as_of": as_of,
        "sectors": per_sector,
        "totals": totals,
        "newly_synced_count": len(newly_synced),
        "newly_synced_cross_check_violations": cross_check_violations,
    }


def _current_v1_membership() -> dict[str, Any]:
    """Independent current-membership denominator for classic closure.

    This starts at latest sector inference rather than the candidate loader,
    so an accidental loader join/filter cannot define away its own omission.
    Registry removals and deterministically identified non-common securities
    are explicit out-of-scope categories; missing scorecards remain eligible.
    """
    from app.autonomous.sector_candidates import _security_filter_for_ticker
    from app.db import get_db

    marks = ",".join("?" for _ in CANONICAL_SWEEP_SECTORS)
    with get_db() as conn:
        all_ticker_count = int(
            conn.execute("SELECT COUNT(DISTINCT ticker) AS count FROM sector_inference").fetchone()[
                "count"
            ]
        )
        ever_sector_tagged_count = int(
            conn.execute(
                "SELECT COUNT(DISTINCT ticker) AS count FROM sector_inference "
                "WHERE inferred_sector IS NOT NULL"
            ).fetchone()["count"]
        )
        latest_null_previously_tagged = [
            str(row["ticker"]).strip().upper()
            for row in conn.execute(
                f"""
                WITH latest AS (
                    SELECT ticker, MAX(as_of_date) AS as_of_date
                    FROM sector_inference
                    GROUP BY ticker
                )
                SELECT DISTINCT UPPER(si.ticker) AS ticker
                FROM sector_inference si
                JOIN latest
                  ON latest.ticker = si.ticker
                 AND latest.as_of_date = si.as_of_date
                WHERE si.inferred_sector IS NULL
                  AND EXISTS (
                      SELECT 1 FROM sector_inference prior
                      WHERE prior.ticker = si.ticker
                        AND prior.inferred_sector IN ({marks})
                  )
                ORDER BY ticker
                """,
                list(CANONICAL_SWEEP_SECTORS),
            ).fetchall()
        ]
        rows = conn.execute(
            f"""
            WITH latest_valid AS (
                SELECT ticker, MAX(as_of_date) AS as_of_date
                FROM sector_inference
                WHERE inferred_sector IN ({marks})
                GROUP BY ticker
            )
            SELECT DISTINCT UPPER(si.ticker) AS ticker,
                            LOWER(si.inferred_sector) AS sector
            FROM sector_inference si
            JOIN latest_valid
              ON latest_valid.ticker = si.ticker
             AND latest_valid.as_of_date = si.as_of_date
            WHERE si.inferred_sector IN ({marks})
            ORDER BY ticker
            """,
            [*CANONICAL_SWEEP_SECTORS, *CANONICAL_SWEEP_SECTORS],
        ).fetchall()
        membership = [str(row["ticker"]) for row in rows]
        sector_by_ticker = {str(row["ticker"]): str(row["sector"]) for row in rows}
        try:
            registry_removed = {
                str(row["primary_ticker"]).strip().upper()
                for row in conn.execute(
                    "SELECT primary_ticker FROM sec_registrants "
                    "WHERE primary_ticker IS NOT NULL AND removed_at IS NOT NULL"
                ).fetchall()
            }
        except sqlite3.OperationalError:
            registry_removed = set()
        try:
            scorecard_tickers: set[str] = set()
            rows = conn.execute(
                "SELECT DISTINCT ticker FROM valuations WHERE method = 'scorecard'"
            ).fetchall()
            for row in rows:
                ticker = str(row["ticker"]).strip().upper()
                if (
                    latest_decision_eligible_valuation_row(
                        conn,
                        ticker=ticker,
                        method="scorecard",
                    )
                    is not None
                ):
                    scorecard_tickers.add(ticker)
        except sqlite3.OperationalError:
            scorecard_tickers = set()

    removed = sorted(set(membership) & registry_removed)
    security_excluded: list[str] = []
    eligible: list[str] = []
    for ticker in membership:
        if ticker in registry_removed:
            continue
        result = _security_filter_for_ticker(ticker)
        if result.is_common_equity:
            eligible.append(ticker)
        else:
            security_excluded.append(ticker)
    missing_scorecard = sorted(set(eligible) - scorecard_tickers)
    partitioned = set(eligible) | set(security_excluded) | set(removed)
    membership_unaccounted = sorted(set(membership) - partitioned)
    fingerprint = sha256(
        json.dumps(
            [(ticker, sector_by_ticker[ticker]) for ticker in membership],
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "all_sector_inference_tickers": all_ticker_count,
        "ever_sector_tagged": ever_sector_tagged_count,
        "current_sector_tagged": len(membership),
        "current_membership_fingerprint": fingerprint,
        "current_membership_tickers": membership,
        "latest_null_previously_tagged": len(latest_null_previously_tagged),
        "latest_null_previously_tagged_tickers": latest_null_previously_tagged,
        "latest_null_carried_last_valid_sector": len(latest_null_previously_tagged),
        "latest_null_carried_last_valid_sector_tickers": (latest_null_previously_tagged),
        "eligible_common_equity": len(eligible),
        "eligible_common_equity_tickers": eligible,
        "security_type_non_common_equity": len(security_excluded),
        "security_type_non_common_equity_tickers": sorted(security_excluded),
        "registry_removed": len(removed),
        "registry_removed_tickers": removed,
        "membership_partition_complete": not membership_unaccounted,
        "membership_unaccounted_tickers": membership_unaccounted,
        # Diagnostic only: classic now admits these as visible insufficient-
        # data candidates instead of silently deleting them at the SQL join.
        "missing_scorecard": len(missing_scorecard),
        "missing_scorecard_tickers": missing_scorecard,
    }


def _full_universe_cell_command(
    *,
    sector: str,
    band: str,
    as_of: str | None = None,
    coverage_campaign_id: str | None = None,
    allow_carried_verdicts: bool = True,
) -> str:
    command = (
        ".venv/bin/ivi autonomous-sector-run "
        f"--sector {sector} --market-cap-focus {band} "
        "--only-unswept --pipeline-version v1"
    )
    if as_of:
        command += f" --as-of {as_of}"
    if coverage_campaign_id:
        command += f" --coverage-campaign-id {coverage_campaign_id}"
    if not allow_carried_verdicts:
        command += " --no-carry-prior-verdicts"
    return command


def full_universe_delta_report(
    *,
    as_of: str | None = None,
    synced_after: str | None = None,
    coverage_campaign_id: str | None = None,
    allow_carried_verdicts: bool = True,
    target_tickers: list[str] | None = None,
) -> dict[str, Any]:
    """Resolve and reconcile every atomic classic sector/band cell at $0 LLM.

    Completion is fail-closed against independent current membership.  It is
    never inferred from historical cell presence, a hand-maintained sector
    group, or the lifetime loaded-set union.
    """
    membership = _current_v1_membership()
    membership_eligible = set(membership["eligible_common_equity_tickers"])
    target_scope = (
        membership_eligible
        if target_tickers is None
        else set(_normalized_values(target_tickers)) & membership_eligible
    )
    generated_at = _now()
    cells: list[dict[str, Any]] = []
    loaded_distinct: set[str] = set()
    selected_distinct: set[str] = set()
    structural_distinct: set[str] = set()
    covered_distinct: set[str] = set()
    uncovered_distinct: set[str] = set()
    to_review_distinct: set[str] = set()
    source_error_cells: list[str] = []
    gate_error_cells: list[str] = []
    cross_check_violations: list[str] = []
    occurrence_totals = {
        "loaded": 0,
        "selected": 0,
        "structural_screened": 0,
        "gate_errors": 0,
        "covered_loaded": 0,
        "coverage_uncovered": 0,
        "swept": 0,
        "carried": 0,
        "to_review": 0,
    }
    cap_classification_caches: dict[str, dict[str, Any]] = {}

    for band in V1_ATOMIC_BANDS:
        band_report = band_delta_report(
            band,
            sectors=list(CANONICAL_SWEEP_SECTORS),
            as_of=as_of,
            synced_after=synced_after,
            pipeline_version="v1",
            cap_classification_caches=cap_classification_caches,
            coverage_campaign_id=coverage_campaign_id,
            allow_carried_verdicts=allow_carried_verdicts,
            target_tickers=sorted(target_scope),
        )
        cross_check_violations.extend(
            f"{band}:{item}" for item in band_report["newly_synced_cross_check_violations"]
        )
        for key in occurrence_totals:
            occurrence_totals[key] += int(band_report["totals"][key])
        for sector in CANONICAL_SWEEP_SECTORS:
            detail = dict(band_report["sectors"][sector])
            cell_id = f"{sector}:{band}"
            command = _full_universe_cell_command(
                sector=sector,
                band=band,
                as_of=as_of,
                coverage_campaign_id=coverage_campaign_id,
                allow_carried_verdicts=allow_carried_verdicts,
            )
            detail.update(
                {
                    "cell_id": cell_id,
                    "sector": sector,
                    "band": band,
                    "pipeline_version": "v1",
                    "max_candidates": None,
                    "sector_context_limit": V1_SECTOR_CONTEXT_LIMIT,
                    "command": command,
                }
            )
            cells.append(detail)
            loaded_distinct.update(detail["loaded_tickers"])
            selected_distinct.update(detail["selected_tickers"])
            structural_distinct.update(detail["structural_screened_tickers"])
            covered_distinct.update(
                set(detail["loaded_tickers"]) - set(detail["coverage_uncovered_tickers"])
            )
            uncovered_distinct.update(detail["coverage_uncovered_tickers"])
            to_review_distinct.update(detail["to_review_tickers"])
            if detail["source_errors"]:
                source_error_cells.append(cell_id)
            if detail["gate_error_tickers"]:
                gate_error_cells.append(cell_id)

    eligible = target_scope
    loader_residual = sorted(eligible - loaded_distinct)
    coverage_residual = sorted(eligible & uncovered_distinct)
    pending_cells = [
        {
            "cell_id": cell["cell_id"],
            "sector": cell["sector"],
            "band": cell["band"],
            "coverage_uncovered": cell["coverage_uncovered"],
            "coverage_uncovered_tickers": cell["coverage_uncovered_tickers"],
            "to_review": cell["to_review"],
            "to_review_tickers": cell["to_review_tickers"],
            "sector_context_limit": cell["sector_context_limit"],
            "command": cell["command"],
        }
        for cell in cells
        if cell["coverage_uncovered"]
    ]
    complete = not any(
        (
            loader_residual,
            coverage_residual,
            source_error_cells,
            gate_error_cells,
            cross_check_violations,
            membership.get("membership_unaccounted_tickers", []),
        )
    )
    return {
        "schema_version": "classic_full_universe_delta_v1",
        "completion_semantics": (
            "Every current eligible ticker is loader-surfaced and has an "
            "auditable terminal state: structural screen, carried prior verdict, "
            "explicit NEEDS_DATA, or a successful per-ticker LLM candidate "
            "review. Raw legacy loader rows and deterministic memo fallbacks do "
            "not close coverage."
        ),
        "operator_contract": (
            "First use --backfill-artifacts to recover exact terminal evidence "
            "from persisted history. Execute pending cell commands, regenerate "
            "this manifest, and repeat until --require-complete exits zero. Each "
            "final packet receives its own LLM candidate review; provider failures "
            "remain open for retry."
        ),
        "status": "COMPLETE" if complete else "INCOMPLETE",
        "complete": complete,
        "pipeline_version": "v1",
        "coverage_campaign_id": str(coverage_campaign_id or "").strip() or None,
        "allow_carried_verdicts": bool(allow_carried_verdicts),
        "target_scope": "all_eligible" if target_tickers is None else "frozen_ticker_set",
        "target_ticker_count": len(target_scope),
        "target_tickers_sha256": sha256(
            json.dumps(sorted(target_scope), separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "generated_at": generated_at,
        "as_of": as_of,
        "atomic_bands": list(V1_ATOMIC_BANDS),
        "canonical_sectors": list(CANONICAL_SWEEP_SECTORS),
        "expected_cells": len(V1_ATOMIC_BANDS) * len(CANONICAL_SWEEP_SECTORS),
        "cells_resolved": len(cells),
        "membership": membership,
        "cell_occurrence_totals": occurrence_totals,
        "distinct_ticker_totals": {
            "loader_surfaced": len(loaded_distinct),
            "selected_for_delta_split": len(selected_distinct),
            "structural_screened": len(structural_distinct),
            "covered_loaded": len(covered_distinct),
            "coverage_uncovered": len(uncovered_distinct),
            "fresh_review_queue": len(to_review_distinct),
        },
        "loader_residual_tickers": loader_residual,
        "coverage_residual_tickers": coverage_residual,
        "source_error_cells": source_error_cells,
        "gate_error_cells": gate_error_cells,
        "newly_synced_cross_check_violations": cross_check_violations,
        "pending_cells": pending_cells,
        "cells": cells,
    }
