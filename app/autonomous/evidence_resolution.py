"""Evidence auto-resolution for held candidates (the KEQU/CXDO funnel gap).

The sweep used to hold finalists whose packets lacked fetchable inputs —
no fresh scorecard, no cached annual filing, no price — as BLOCKED or
DATA_INCOMPLETE without anyone running the tools that fetch those inputs.
This module makes DATA_INCOMPLETE mean "tools genuinely cannot fetch it":

* ``pre_assembly_data_gap_repair`` runs inside the sector run BEFORE signal
  packets are assembled. V1 retains its count cap. V2 uses that number only
  as a durable checkpoint batch size and walks every admitted name through
  membership, identity, cap, facts, filings, parsing, price, valuation, and
  packet-readiness stages without dropping the tail. It remains deterministic
  and makes no LLM calls.
* ``resolve_and_reaudit_pool`` replays the same repairs over the held rows of
  persisted sector artifacts and re-audits each name offline, reporting who
  promotes (BLOCKED -> DATA_INCOMPLETE/WATCHLIST_ONLY/PASS) and who is held
  for a genuine, non-fetchable reason.

Repairs are idempotent and budget-capped by count; every action is recorded
in the returned metadata, never applied silently.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import asdict, is_dataclass
from hashlib import sha256
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from app.autonomous.companyfacts_repair import repair_issuer_annual_companyfacts
from app.autonomous.price_repair import resolve_v2_price
from app.config import AppConfig, get_config
from app.util.credential_hygiene import sanitize_url_credentials
from app.util.financial_data_access import (
    ANNUAL_CACHED_FILING_FORM_TYPES,
    VALID_CACHED_FILING_STATUSES,
    foreign_normalized_facts_gap_reason,
    issuer_companyfacts_rows,
    issuer_filing_rows,
    normalize_cik,
    resolve_filing_issuer_scope,
)
from app.valuation.provenance import (
    V2_VALUATION_STALENESS_DAYS,
    scorecard_is_stale,
    validate_v2_scorecard_provenance,
)
from app.valuation.lineage import latest_decision_eligible_valuation_row

logger = logging.getLogger(__name__)


class V2ExecutionBoundDriftError(RuntimeError):
    """A persisted/prospective repair set differs from paid authority."""


# Hard blockers that name a fetchable input rather than a judgement: the
# deterministic repair lane can act on these. MISSING_*_EVIDENCE codes are
# run-scoped tool evidence — fetchable in-run, and offline they downgrade to
# the resolve-then-promote tier instead of reading as quality rejects.
FETCHABLE_HARD_BLOCKERS = frozenset(
    {
        "MISSING_VALUATION",
        "MISSING_BASE_RETURN_CASE",
        "MISSING_PRICE",
        "NO_FILING",
        "NO_READABLE_ANNUAL_FILING",
        "FILING_RISK_NO_FILING",
        "RISK_SECTION_NOT_FOUND",
        "FILING_RISK_SECTION_NOT_FOUND",
        "MISSING_RELEVANT_EXPECTED_RETURN_EVIDENCE",
        "MISSING_COMPANY_SPECIFIC_EVIDENCE",
    }
)

VALUATION_STALENESS_DAYS = V2_VALUATION_STALENESS_DAYS

# V2 treats the old per-run repair limit as a durability checkpoint cadence,
# not as a terminal exclusion.  Keep the names explicit and ordered because
# ``last_completed_stage`` is also used by diagnostics and offline replay.
V2_REPAIR_STAGE_SEQUENCE = (
    "MEMBERSHIP",
    "IDENTITY",
    "CAP",
    "FACTS_AVAILABILITY",
    "FILINGS",
    "PARSING",
    "PRICE",
    "VALUATION",
    "PACKET",
)
V2_REPAIR_CHECKPOINT_VERSION = 4
V2_SOURCE_LIMITED_REPAIR_POLICIES = frozenset(
    {
        "OBSERVE_ONLY",
        "LOCAL_EVIDENCE_ONLY",
    }
)
V2_CONTENT_AVAILABLE_FILING_STATUSES = VALID_CACHED_FILING_STATUSES + ("downloaded",)

# These are the normalized annual facts consumed by every non-financial
# company packet's core operating/valuation lenses.  A row count is not an
# availability contract: one isolated fact cannot support underwriting.  The
# bank path has a separate set because CFO/capex and conventional debt are not
# meaningful there.
PACKET_REQUIRED_NORMALIZED_FACT_HISTORY = {
    "revenue": 3,
    "operating_income": 3,
    "net_income": 1,
    "cfo": 1,
    "capex": 1,
    "cash": 1,
    "total_assets": 1,
    "total_debt": 1,
    "equity": 1,
    "shares_outstanding": 2,
}
PACKET_REQUIRED_FINANCIAL_FACT_HISTORY = {
    "revenue": 1,
    "net_income": 1,
    "cash": 1,
    "equity": 1,
    "total_assets": 1,
    "deposits": 1,
    "loans": 1,
    "shares_outstanding": 2,
}
# The insurance-common contract in app.insurance.valuation blocks only on
# book value, share count, and normalized ROE inputs. Requiring CFO/capex for
# an underwriter would make a non-applicable operating-company lens a data
# gate, so insurance uses the literal consumer contract.
PACKET_REQUIRED_INSURANCE_FACT_HISTORY = {
    "net_income": 1,
    "equity": 1,
    "shares_outstanding": 1,
}


def is_held_on_fetchable_gaps_only(hard_blockers: list[str] | None) -> bool:
    """True when every hard blocker names a fetchable input."""
    blockers = [str(item) for item in (hard_blockers or []) if str(item).strip()]
    if not blockers:
        return False
    return all(code in FETCHABLE_HARD_BLOCKERS for code in blockers)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def ensure_company_row(
    ticker: str,
    *,
    db_path: str | Path | None = None,
    issuer_cik: str | None = None,
) -> bool:
    """Idempotently ensure a companies row exists (filing ingest needs one)."""
    cfg = get_config()
    path = Path(db_path) if db_path is not None else Path(cfg.db_path)
    upper = str(ticker).upper()
    cik = normalize_cik(issuer_cik)
    if cik is None:
        from app.ingest.cik_registry import resolve

        try:
            cik = normalize_cik(resolve(upper))
        except Exception:  # noqa: BLE001 - no CIK, nothing to ensure
            return False
    if cik is None:
        return False
    conn = _connect(path)
    try:
        row = conn.execute("SELECT cik FROM companies WHERE ticker = ?", (upper,)).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO companies(ticker, cik, name, created_at) VALUES (?, ?, ?, ?)",
                (upper, str(cik), upper, _utc_now_iso()),
            )
            conn.commit()
        elif normalize_cik(row["cik"]) != cik:
            # Never rewrite an existing ticker onto a different registrant.
            return False
        return True
    finally:
        conn.close()


def _has_readable_annual_filing(conn: sqlite3.Connection, ticker: str) -> bool:
    """Legacy v1 ticker-exact 10-K availability check."""

    row = conn.execute(
        """
        SELECT 1 FROM filings
        WHERE ticker = ? AND form_type LIKE '10-K%'
          AND status IN ('OK', 'parsed') AND local_path IS NOT NULL
        LIMIT 1
        """,
        (str(ticker).upper(),),
    ).fetchone()
    return row is not None


def _has_readable_issuer_annual_filing(
    conn: sqlite3.Connection,
    ticker: str,
    *,
    as_of_date: str,
    issuer_cik: str | None = None,
    aliases: tuple[str, ...] = (),
    persist_recovered: bool = True,
) -> bool:
    """V2 issuer-aware annual availability through local filing context.

    A populated ``filings.local_path`` is not required. The local-only filing
    materializer also recognizes deterministic raw-cache and dossier-cache
    bytes, without making a network request. All supported annual families
    (10-K, 20-F, 40-F, including amendments) are eligible.
    """
    from app.research.filing_context import load_research_filing_context

    context = load_research_filing_context(
        ticker,
        as_of_date=as_of_date,
        issuer_cik=issuer_cik,
        aliases=aliases,
        issuer_aware=True,
        connection=conn,
        annual_filing_limit=1,
        quarters=0,
        include_material_events=False,
        allow_annual_download=False,
    )
    # Local raw/dossier recovery is a real content-state transition. Persist
    # the recovered path so the subsequent PARSING stage can consume exactly
    # those bytes instead of leaving a download_error row permanently
    # unparseable.
    from app.parse.document_store import file_hash

    for document in context.documents:
        if document.role != "annual" or not document.local_path:
            continue
        if not persist_recovered:
            continue
        row = conn.execute(
            "SELECT id, status, local_path, hash FROM filings WHERE accession = ? LIMIT 1",
            (document.accession,),
        ).fetchone()
        if row is None:
            continue
        status = str(row["status"] or "").strip().lower()
        if status in {"ok", "parsed"}:
            continue
        path = Path(document.local_path)
        digest = file_hash(path)
        if (
            status == "downloaded"
            and str(row["local_path"] or "") == str(path)
            and str(row["hash"] or "") == digest
        ):
            continue
        conn.execute(
            "UPDATE filings SET local_path = ?, hash = ?, status = 'downloaded', "
            "updated_at = ? WHERE id = ?",
            (str(path), digest, _utc_now_iso(), int(row["id"])),
        )
        conn.commit()
    return any(document.role == "annual" for document in context.documents)


def _latest_scorecard_asof(
    conn: sqlite3.Connection,
    ticker: str,
    *,
    cutoff_date: str | None = None,
) -> str | None:
    """Return raw scorecard timing for repair scheduling and diagnostics only.

    This helper does not authorize a valuation and must never feed a financial
    decision. V2 stage decisions use ``validate_v2_scorecard_provenance`` and
    checkpoint snapshots use ``_valuation_evidence_snapshot``; both select the
    normal newest candidate first and authorize that exact row.
    """

    clauses = ["ticker = ?", "method = 'scorecard'"]
    params: list[Any] = [str(ticker).upper()]
    if cutoff_date:
        clauses.append("as_of_date <= ?")
        params.append(str(cutoff_date))
    row = conn.execute(
        f"SELECT MAX(as_of_date) FROM valuations WHERE {' AND '.join(clauses)}",
        params,
    ).fetchone()
    return str(row[0]) if row and row[0] else None


def _v2_scorecard_provenance_state(
    conn: sqlite3.Connection,
    ticker: str,
    *,
    as_of_date: str,
    issuer_cik: str | None,
    issuer_aliases: tuple[str, ...],
    price_snapshot: Any | None,
) -> dict[str, Any]:
    return validate_v2_scorecard_provenance(
        conn,
        ticker,
        as_of_date=as_of_date,
        issuer_cik=issuer_cik,
        issuer_aliases=issuer_aliases,
        price_snapshot=price_snapshot,
    )


def _scorecard_is_stale(scorecard_asof: str | None, as_of_date: str) -> bool:
    return scorecard_is_stale(
        scorecard_asof,
        as_of_date,
        max_age_days=VALUATION_STALENESS_DAYS,
    )


def _normalized_tickers(tickers: list[str]) -> list[str]:
    return list(
        dict.fromkeys(str(ticker).strip().upper() for ticker in tickers if str(ticker).strip())
    )


def _ticker_needs_legacy_repair(
    conn: sqlite3.Connection,
    ticker: str,
    *,
    as_of_date: str,
) -> bool:
    return not _has_readable_annual_filing(conn, ticker) or _scorecard_is_stale(
        _latest_scorecard_asof(conn, ticker), as_of_date
    )


def _ticker_needs_v2_repair(
    conn: sqlite3.Connection,
    ticker: str,
    *,
    as_of_date: str,
    db_path: Path,
    cfg: AppConfig,
    candidate_context: dict[str, Any] | None,
) -> bool:
    identity = _resolved_security_identity_payload(
        ticker,
        as_of_date=as_of_date,
        db_path=db_path,
        candidate_context=candidate_context,
    )
    fact_coverage = _issuer_normalized_fact_coverage(
        conn,
        ticker,
        as_of_date=as_of_date,
        db_path=db_path,
        cfg=cfg,
        candidate_context=candidate_context,
    )
    return (
        not _has_readable_issuer_annual_filing(
            conn,
            ticker,
            as_of_date=as_of_date,
            issuer_cik=identity.get("issuer_cik"),
            aliases=tuple(identity.get("issuer_aliases") or ()),
            persist_recovered=False,
        )
        or fact_coverage["outcome"] != "AVAILABLE"
        or _scorecard_is_stale(
            _latest_scorecard_asof(conn, ticker, cutoff_date=as_of_date),
            as_of_date,
        )
    )


def _safe_checkpoint_token(value: str) -> str:
    token = "".join(char.lower() if char.isalnum() else "_" for char in str(value or "")).strip("_")
    return token[:48] or "sector"


def _stable_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _stable_json_value(item)
            for key, item in sorted(value.items(), key=lambda row: str(row[0]))
        }
    if isinstance(value, set):
        return sorted((_stable_json_value(item) for item in value), key=str)
    if isinstance(value, (list, tuple)):
        return [_stable_json_value(item) for item in value]
    converted = _as_plain_dict(value)
    if converted:
        return _stable_json_value(converted)
    return str(value)


def _json_fingerprint(value: Any) -> str:
    encoded = json.dumps(
        _stable_json_value(value),
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(encoded.encode("utf-8")).hexdigest()


def _candidate_context_revision(
    candidate_context: dict[str, Any] | None,
    explicit_revision: str | None,
) -> str:
    if explicit_revision:
        return str(explicit_revision)
    if not isinstance(candidate_context, dict):
        return "none"
    supplied = candidate_context.get("candidate_context_revision")
    if supplied:
        return str(supplied)
    # Deliberately exclude prior repair/checkpoint output: it changes while an
    # attempt is running and is not an input to membership/cap truth.
    projection = {
        key: candidate_context.get(key)
        for key in (
            "sector",
            "market_cap_focus",
            "source",
            "requested_tickers",
            "loaded_tickers",
            "membership_tickers",
            "execution_tickers",
            "deferred_by_bound_tickers",
            "excluded_tickers",
            "execution_bound",
            "membership_fingerprint",
            "execution_fingerprint",
            "execution_bound_frozen",
            "census_lineage",
            "cap_classifications",
            "candidate_dispositions",
            "structural_gate_results",
        )
        if key in candidate_context
    }
    return _json_fingerprint(projection)


def _v2_resume_input(
    *,
    as_of_date: str,
    checkpoint_scope: str | None,
    evidence_revision: str | None,
    candidate_context_revision: str | None,
    candidate_context: dict[str, Any] | None,
) -> dict[str, Any]:
    resolved_evidence_revision = str(
        evidence_revision
        or (
            candidate_context.get("evidence_revision")
            if isinstance(candidate_context, dict)
            else ""
        )
        or "stage_evidence_fingerprints"
    )
    resolved_context_revision = _candidate_context_revision(
        candidate_context,
        candidate_context_revision,
    )
    execution_tickers = _normalized_tickers(
        (candidate_context.get("execution_tickers") or [])
        if isinstance(candidate_context, dict)
        else []
    )
    execution_fingerprint = (
        str(
            candidate_context.get("execution_fingerprint")
            if isinstance(candidate_context, dict)
            else ""
        )
        .strip()
        .lower()
    )
    fingerprint = _json_fingerprint(
        {
            "as_of_date": str(as_of_date),
            "checkpoint_scope": str(checkpoint_scope or ""),
            "evidence_revision": resolved_evidence_revision,
            "candidate_context_revision": resolved_context_revision,
            "execution_tickers": execution_tickers,
            "execution_fingerprint": execution_fingerprint,
        }
    )
    return {
        "evidence_revision": resolved_evidence_revision,
        "candidate_context_revision": resolved_context_revision,
        "execution_tickers": execution_tickers,
        "execution_fingerprint": execution_fingerprint,
        "resume_fingerprint": fingerprint,
        # Compatibility name retained for stored-artifact consumers.  Unlike
        # the v2 prototype this is deliberately independent of the ephemeral
        # sector runtime run_id, so a fresh process can resume the frontier.
        "input_fingerprint": fingerprint,
    }


def _v2_attempt_input(
    *,
    run_id: str | None,
    as_of_date: str,
    checkpoint_scope: str | None,
    evidence_revision: str | None,
    candidate_context_revision: str | None,
    candidate_context: dict[str, Any] | None,
) -> dict[str, Any]:
    resume_input = _v2_resume_input(
        as_of_date=as_of_date,
        checkpoint_scope=checkpoint_scope,
        evidence_revision=evidence_revision,
        candidate_context_revision=candidate_context_revision,
        candidate_context=candidate_context,
    )
    resolved_run_id = str(
        run_id
        or (candidate_context.get("run_id") if isinstance(candidate_context, dict) else "")
        or f"repair_{as_of_date}_{checkpoint_scope or 'sector'}"
    )
    attempt_started_at = _utc_now_iso()
    attempt_fingerprint = _json_fingerprint(
        {
            "run_id": resolved_run_id,
            "resume_fingerprint": resume_input["resume_fingerprint"],
            "attempt_started_at": attempt_started_at,
        }
    )
    return {
        **resume_input,
        "run_id": resolved_run_id,
        "attempt_started_at": attempt_started_at,
        "attempt_id": (f"{_safe_checkpoint_token(resolved_run_id)}_{attempt_fingerprint[:16]}"),
    }


def _default_v2_repair_checkpoint_path(
    *,
    cfg: AppConfig,
    tickers: list[str],
    as_of_date: str,
    checkpoint_scope: str | None,
    attempt_input: dict[str, Any],
) -> Path:
    identity = json.dumps(
        {
            "as_of_date": str(as_of_date),
            "checkpoint_scope": str(checkpoint_scope or ""),
            "tickers": sorted(tickers),
            "resume_fingerprint": attempt_input["resume_fingerprint"],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = sha256(identity.encode("utf-8")).hexdigest()[:16]
    scope_token = _safe_checkpoint_token(checkpoint_scope or "sector")
    asof_token = _safe_checkpoint_token(as_of_date)
    return (
        Path(cfg.runs_dir)
        / "autonomous_sector_repair"
        / f"{asof_token}_{scope_token}_{digest}"
        / "repair_checkpoint.json"
    )


def v2_repair_checkpoint_path(
    *,
    tickers: list[str],
    as_of_date: str,
    checkpoint_scope: str,
    candidate_context: dict[str, Any] | None = None,
    evidence_revision: str | None = None,
    candidate_context_revision: str | None = None,
    cfg: AppConfig | None = None,
) -> Path:
    """Return the stable default checkpoint path without starting an attempt.

    Benchmark orchestration uses this before entering the sector runtime so an
    interrupt artifact can surface the exact repair frontier that a later
    process will resume.
    """

    resolved_cfg = cfg or get_config()
    resume_input = _v2_resume_input(
        as_of_date=as_of_date,
        checkpoint_scope=checkpoint_scope,
        evidence_revision=evidence_revision,
        candidate_context_revision=candidate_context_revision,
        candidate_context=candidate_context,
    )
    return _default_v2_repair_checkpoint_path(
        cfg=resolved_cfg,
        tickers=_normalized_tickers(tickers),
        as_of_date=as_of_date,
        checkpoint_scope=checkpoint_scope,
        attempt_input=resume_input,
    )


def _read_v2_repair_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unreadable v2 repair checkpoint: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid v2 repair checkpoint payload: {path}")
    return payload


def _write_v2_repair_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """Atomically persist the repair frontier after every completed stage."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload["updated_at"] = _utc_now_iso()
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)
    diagnostic_value = str(payload.get("diagnostic_path") or "").strip()
    if diagnostic_value:
        diagnostic_path = Path(diagnostic_value)
        diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
        candidate_diagnostics: dict[str, dict[str, Any]] = {}
        candidates = payload.get("candidates")
        if isinstance(candidates, dict):
            for ticker, candidate in candidates.items():
                if not isinstance(candidate, dict):
                    continue
                stages = candidate.get("stages")
                stage_statuses: dict[str, Any] = {}
                stage_errors: dict[str, Any] = {}
                if isinstance(stages, dict):
                    for stage_name, stage_payload in stages.items():
                        if not isinstance(stage_payload, dict):
                            continue
                        stage_statuses[str(stage_name)] = stage_payload.get("status")
                        if stage_payload.get("error"):
                            stage_errors[str(stage_name)] = stage_payload.get("error")
                candidate_diagnostics[str(ticker)] = {
                    "queue_status": candidate.get("queue_status"),
                    "last_completed_stage": candidate.get("last_completed_stage"),
                    "next_stage": candidate.get("next_stage"),
                    "stage_statuses": stage_statuses,
                    "stage_errors": stage_errors,
                }
        diagnostic_payload = {
            "artifact_type": "autonomous_sector_data_repair_attempt_diagnostic_v2",
            "attempt_id": payload.get("attempt_id"),
            "run_id": payload.get("run_id"),
            "attempt_started_at": payload.get("attempt_started_at"),
            "resume_fingerprint": payload.get("resume_fingerprint"),
            "input_fingerprint": payload.get("input_fingerprint"),
            "evidence_revision": payload.get("evidence_revision"),
            "candidate_context_revision": payload.get("candidate_context_revision"),
            "pipeline_version": payload.get("pipeline_version"),
            "as_of_date": payload.get("as_of_date"),
            "checkpoint_scope": payload.get("checkpoint_scope"),
            "execution_status": payload.get("execution_status"),
            "created_at": payload.get("created_at"),
            "updated_at": payload.get("updated_at"),
            "checkpoint_path": str(path),
            "candidate_order": list(payload.get("candidate_order") or []),
            "candidate_diagnostics": candidate_diagnostics,
        }
        diagnostic_tmp = diagnostic_path.with_suffix(f"{diagnostic_path.suffix}.tmp")
        diagnostic_tmp.write_text(
            json.dumps(diagnostic_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        diagnostic_tmp.replace(diagnostic_path)


def _new_v2_candidate_checkpoint(ticker: str) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "queue_status": "PENDING",
        "last_completed_stage": None,
        "next_stage": V2_REPAIR_STAGE_SEQUENCE[0],
        "stages": {stage: {"status": "PENDING"} for stage in V2_REPAIR_STAGE_SEQUENCE},
    }


def _initialize_v2_repair_checkpoint(
    *,
    path: Path,
    tickers: list[str],
    as_of_date: str,
    checkpoint_scope: str | None,
    checkpoint_batch_size: int,
    attempt_input: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    previous = _read_v2_repair_checkpoint(path)
    same_scope = bool(previous) and (
        previous.get("pipeline_version") == "v2"
        and str(previous.get("as_of_date") or "") == str(as_of_date)
        and str(previous.get("checkpoint_scope") or "") == str(checkpoint_scope or "")
    )
    if previous and not same_scope:
        raise V2ExecutionBoundDriftError(
            "V2 repair checkpoint does not match the requested pipeline scope."
        )
    normalized_requested = _normalized_tickers(tickers)
    if previous:
        prior_order = _normalized_tickers(previous.get("candidate_order") or [])
        prior_candidate_keys = _normalized_tickers(
            list((previous.get("candidates") or {}).keys())
            if isinstance(previous.get("candidates"), dict)
            else []
        )
        if prior_order != normalized_requested or set(prior_candidate_keys) != set(
            normalized_requested
        ):
            raise V2ExecutionBoundDriftError(
                "V2 repair checkpoint frozen execution set does not match the request."
            )
    resumed = bool(
        previous
        and previous.get("checkpoint_version") == V2_REPAIR_CHECKPOINT_VERSION
        and (previous.get("resume_fingerprint") or previous.get("input_fingerprint"))
        == attempt_input["resume_fingerprint"]
    )
    if resumed:
        payload = previous
        prior_attempt = {
            "attempt_id": previous.get("attempt_id"),
            "run_id": previous.get("run_id"),
            "attempt_started_at": previous.get("attempt_started_at"),
            "execution_status": previous.get("execution_status"),
            "diagnostic_path": previous.get("diagnostic_path"),
        }
        attempt_history = list(previous.get("attempt_history") or [])
        if prior_attempt.get("attempt_id"):
            attempt_history.append(prior_attempt)
        payload.update(attempt_input)
        payload["attempt_history"] = attempt_history
        payload["prior_attempt"] = prior_attempt
        payload["diagnostic_path"] = str(
            path.parent / "attempt_diagnostics" / f"{attempt_input['attempt_id']}.json"
        )
    else:
        if previous and not same_scope:
            raise V2ExecutionBoundDriftError(
                "V2 repair checkpoint does not match the requested pipeline scope."
            )
        created_at = _utc_now_iso()
        diagnostic_path = (
            path.parent / "attempt_diagnostics" / f"{attempt_input['attempt_id']}.json"
        )
        payload = {
            "artifact_type": "autonomous_sector_data_repair_checkpoint_v2",
            "checkpoint_version": V2_REPAIR_CHECKPOINT_VERSION,
            "pipeline_version": "v2",
            "as_of_date": str(as_of_date),
            "checkpoint_scope": str(checkpoint_scope or ""),
            **attempt_input,
            "diagnostic_path": str(diagnostic_path),
            "created_at": created_at,
            "updated_at": created_at,
            "execution_status": "IN_PROGRESS",
            "checkpoint_batch_size": checkpoint_batch_size,
            "candidate_order": [],
            "candidates": {},
        }
        if previous:
            payload["prior_attempt"] = {
                "attempt_id": previous.get("attempt_id"),
                "run_id": previous.get("run_id"),
                "attempt_started_at": previous.get("attempt_started_at"),
                "input_fingerprint": previous.get("input_fingerprint"),
                "resume_fingerprint": previous.get("resume_fingerprint"),
                "execution_status": previous.get("execution_status"),
                "diagnostic_path": previous.get("diagnostic_path"),
            }

    candidate_order = list(normalized_requested)
    candidates = payload.get("candidates")
    if not isinstance(candidates, dict):
        candidates = {}
    for ticker in candidate_order:
        if not isinstance(candidates.get(ticker), dict):
            candidates[ticker] = _new_v2_candidate_checkpoint(ticker)
    candidates = {ticker: candidates[ticker] for ticker in candidate_order}
    payload["candidate_order"] = candidate_order
    payload["candidates"] = candidates
    payload["checkpoint_batch_size"] = checkpoint_batch_size
    payload["execution_status"] = "IN_PROGRESS"
    _write_v2_repair_checkpoint(path, payload)
    return payload, resumed


def _cap_stage_context(
    ticker: str,
    candidate_context: dict[str, Any] | None,
) -> dict[str, Any]:
    classifications = (
        candidate_context.get("cap_classifications")
        if isinstance(candidate_context, dict)
        else None
    )
    row = classifications.get(ticker) if isinstance(classifications, dict) else None
    return dict(row) if isinstance(row, dict) else {}


def _terminal_cap_search_is_authorized(callback: Any | None) -> bool:
    if callback is None or not bool(
        getattr(callback, "_voe_terminal_cap_search_authorized", False)
    ):
        return False
    from app.autonomous.terminal_cap_search import TerminalCapSearchAuthorization

    return isinstance(
        getattr(callback, "authorization", None),
        TerminalCapSearchAuthorization,
    )


def _security_identity_from_repair_payload(
    ticker: str,
    payload: dict[str, Any],
) -> Any:
    from app.autonomous.cap_resolver import SecurityIdentity

    return SecurityIdentity(
        ticker=str(ticker).upper(),
        issuer_cik=payload.get("issuer_cik"),
        issuer_primary_ticker=payload.get("issuer_primary_ticker"),
        issuer_listed_tickers=tuple(payload.get("issuer_listed_tickers") or ()),
        security_role=str(payload.get("security_role") or "UNKNOWN"),
        is_secondary_class=payload.get("is_secondary_class"),
        is_adr=payload.get("is_adr"),
        adr_ratio=payload.get("adr_ratio"),
        share_class_ratio=payload.get("share_class_ratio"),
        ratio_source_url=payload.get("ratio_source_url"),
        identity_source=payload.get("identity_source"),
        identity_source_url=payload.get("identity_source_url"),
        identity_as_of_date=payload.get("identity_as_of_date"),
        identity_confidence=payload.get("identity_confidence"),
    )


def _classification_from_terminal_cap_evidence(
    *,
    ticker: str,
    as_of_date: str,
    identity: Any,
    evidence: Any,
) -> dict[str, Any]:
    from app.autonomous.cap_resolver import (
        CAP_SOURCE_TERMINAL_EXCHANGE,
        CAP_SOURCE_TERMINAL_LOCAL,
        CAP_SOURCE_TERMINAL_PROVIDER,
        CAP_SOURCE_TERMINAL_SEARCH,
        CAP_SOURCE_TERMINAL_SEC,
        CapClassification,
        band_for_market_cap,
    )

    cap_source_by_kind = {
        "LOCAL_AUTHORITATIVE": CAP_SOURCE_TERMINAL_LOCAL,
        "SEC": CAP_SOURCE_TERMINAL_SEC,
        "EXCHANGE": CAP_SOURCE_TERMINAL_EXCHANGE,
        "PROVIDER": CAP_SOURCE_TERMINAL_PROVIDER,
        "SEARCH": CAP_SOURCE_TERMINAL_SEARCH,
    }
    classification = CapClassification(
        ticker=str(ticker).upper(),
        as_of_date=str(as_of_date),
        market_cap_mm=float(evidence.market_cap_mm),
        cap_source=cap_source_by_kind[str(evidence.source_kind)],
        cap_band=band_for_market_cap(float(evidence.market_cap_mm)),
        cap_effective_as_of_date=evidence.as_of_date,
        cap_source_kind=evidence.source_kind,
        cap_source_name=evidence.source_name,
        cap_source_url=evidence.source_url,
        cap_confidence=evidence.confidence,
        issuer_cik=identity.issuer_cik,
        issuer_primary_ticker=identity.issuer_primary_ticker,
        issuer_listed_tickers=identity.issuer_listed_tickers,
        security_role=identity.security_role,
        is_secondary_class=identity.is_secondary_class,
        is_adr=identity.is_adr,
        adr_ratio=identity.adr_ratio,
        share_class_ratio=identity.share_class_ratio,
        identity_source=identity.identity_source,
        identity_source_url=identity.identity_source_url,
        identity_as_of_date=identity.identity_as_of_date,
        identity_confidence=identity.identity_confidence,
        ratio_source_url=identity.ratio_source_url,
        ratio_source_accession=identity.ratio_source_accession,
        ratio_security_symbol=identity.ratio_security_symbol,
        detail=evidence.detail or "direct issuer cap from authorized terminal search",
    )
    return classification.to_dict()


def _v2_cap_price_payload(
    ticker: str,
    candidate_context: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Project cap-stage price evidence without inventing missing provenance."""

    cap = _cap_stage_context(ticker, candidate_context)
    price = cap.get("price_used")
    if isinstance(price, bool) or not isinstance(price, (int, float)):
        return None
    return {
        "ticker": ticker,
        "price": price,
        "as_of_date": cap.get("price_as_of_date") or cap.get("as_of_date"),
        "currency": cap.get("price_currency"),
        "source": cap.get("price_source") or cap.get("cap_source"),
        "source_url": cap.get("price_source_url"),
        "confidence": cap.get("price_confidence") or cap.get("cap_confidence"),
    }


def _prior_price_snapshot(
    prior_stage_results: dict[str, dict[str, Any]] | None,
) -> dict[str, Any] | None:
    price_result = (
        prior_stage_results.get("PRICE") if isinstance(prior_stage_results, dict) else None
    )
    if not isinstance(price_result, dict):
        return None
    if str(price_result.get("outcome") or "").upper() != "AVAILABLE":
        return None
    snapshot = price_result.get("snapshot")
    return dict(snapshot) if isinstance(snapshot, dict) else None


def _resolve_v2_local_stage_price(
    ticker: str,
    *,
    as_of_date: str,
    db_path: Path,
    cfg: AppConfig,
    candidate_context: dict[str, Any] | None,
    run_id: str | None,
    prior_stage_results: dict[str, dict[str, Any]] | None,
):
    """Revalidate the immutable PRICE-stage snapshot before downstream use."""

    seed = _prior_price_snapshot(prior_stage_results) or _v2_cap_price_payload(
        ticker, candidate_context
    )
    return resolve_v2_price(
        ticker,
        as_of_date=as_of_date,
        db_path=db_path,
        cap_stage_price=seed,
        run_id=run_id,
        cfg=cfg,
        allow_provider=False,
        persist=False,
    )


def _as_plain_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if is_dataclass(value) and not isinstance(value, type):
        converted = asdict(value)
        return dict(converted) if isinstance(converted, dict) else {}
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        return dict(converted) if isinstance(converted, dict) else {}
    return {}


def _candidate_disposition_context(
    ticker: str,
    candidate_context: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(candidate_context, dict):
        return {}
    dispositions = candidate_context.get("candidate_dispositions")
    if isinstance(dispositions, dict):
        return _as_plain_dict(dispositions.get(ticker))
    if not isinstance(dispositions, (list, tuple)):
        return {}
    for item in dispositions:
        row = _as_plain_dict(item)
        if str(row.get("ticker") or "").strip().upper() == ticker:
            return row
    return {}


def _membership_stage_context(
    ticker: str,
    candidate_context: dict[str, Any] | None,
) -> dict[str, Any]:
    """Resolve membership from the supplied scope evidence when present."""

    disposition = _candidate_disposition_context(ticker, candidate_context)
    if disposition:
        scope_status = str(disposition.get("scope_status") or "").upper()
        terminal_state = str(disposition.get("terminal_state") or "").upper()
        if scope_status == "OUT_OF_SCOPE" or terminal_state == "OUT_OF_SCOPE":
            return {
                "outcome": "OUT_OF_SCOPE",
                "scope_status": "OUT_OF_SCOPE",
                "source": "candidate_disposition",
                "terminal_state": terminal_state or "OUT_OF_SCOPE",
                "reasons": list(disposition.get("reason_codes") or []),
            }
        if scope_status == "IN_SCOPE":
            return {
                "outcome": "ADMITTED",
                "scope_status": "IN_SCOPE",
                "source": "candidate_disposition",
                "terminal_state": terminal_state or None,
                "reasons": list(disposition.get("reason_codes") or []),
            }

    cap = _cap_stage_context(ticker, candidate_context)
    explicit_in_band = cap.get("in_requested_band")
    if not isinstance(explicit_in_band, bool):
        explicit_in_band = cap.get("market_cap_in_band")
    if isinstance(explicit_in_band, bool):
        return {
            "outcome": "ADMITTED" if explicit_in_band else "OUT_OF_SCOPE",
            "scope_status": "IN_SCOPE" if explicit_in_band else "OUT_OF_SCOPE",
            "source": "cap_classification",
            "market_cap_mm": cap.get("market_cap_mm"),
            "cap_band": cap.get("cap_band") or cap.get("band_label"),
        }

    focus = (
        str(candidate_context.get("market_cap_focus") or "").strip().lower()
        if isinstance(candidate_context, dict)
        else ""
    )
    market_cap_mm = cap.get("market_cap_mm")
    if focus and isinstance(market_cap_mm, (int, float)):
        from app.autonomous.sector_candidates import MARKET_CAP_FOCUS_TIERS

        bounds = MARKET_CAP_FOCUS_TIERS.get(focus)
        if bounds is not None:
            lower, upper = bounds
            in_band = (lower is None or float(market_cap_mm) >= float(lower)) and (
                upper is None or float(market_cap_mm) < float(upper)
            )
            return {
                "outcome": "ADMITTED" if in_band else "OUT_OF_SCOPE",
                "scope_status": "IN_SCOPE" if in_band else "OUT_OF_SCOPE",
                "source": "cap_classification_bounds",
                "market_cap_mm": float(market_cap_mm),
                "cap_band": cap.get("cap_band") or cap.get("band_label"),
                "market_cap_focus": focus,
            }

    if cap and not isinstance(market_cap_mm, (int, float)):
        return {
            # Discovery membership and cap-band resolution are separate v2
            # states. Keep the security admitted to the research ledger so the
            # CAP stage can terminally explain the missing evidence instead of
            # leaving the candidate in an infinite MEMBERSHIP retry.
            "outcome": "ADMITTED",
            "scope_status": "IN_SCOPE",
            "source": "cap_classification",
            "cap_resolution_status": "NEEDS_DATA",
            "reason_code": "MARKET_CAP_UNRESOLVED",
            "cap_source": cap.get("cap_source"),
        }

    excluded = {
        str(value).strip().upper()
        for value in (
            candidate_context.get("excluded_tickers") if isinstance(candidate_context, dict) else []
        )
        or []
        if str(value).strip()
    }
    if ticker in excluded:
        return {
            "outcome": "OUT_OF_SCOPE",
            "scope_status": "OUT_OF_SCOPE",
            "source": "candidate_selection",
            "reason": "EXCLUDED_BY_CANDIDATE_SELECTION",
        }
    return {
        "outcome": "ADMITTED",
        "scope_status": "IN_SCOPE",
        "source": "repair_queue",
    }


def _accepted_census_identity_payload(
    ticker: str,
    *,
    as_of_date: str,
    cap: dict[str, Any],
) -> dict[str, Any] | None:
    """Project accepted-census identity without consulting legacy registries.

    The accepted census already proved one issuer and one primary common
    security. Re-running legacy company/submission heuristics here can bind a
    different registrant or share class before the repair/readiness lane even
    starts. Accepted evidence therefore takes a strict, fail-closed fast path.
    """

    if str(cap.get("cap_source") or "").strip().lower() != "accepted_census":
        return None

    upper = str(ticker or "").strip().upper()
    cik = normalize_cik(cap.get("issuer_cik"))
    primary = str(cap.get("issuer_primary_ticker") or "").strip().upper()
    listed = tuple(
        dict.fromkeys(
            str(value or "").strip().upper()
            for value in cap.get("issuer_listed_tickers") or ()
            if str(value or "").strip()
        )
    )
    required_values = {
        "issuer_cik": cik,
        "issuer_key": cap.get("issuer_key"),
        "security_key": cap.get("security_key"),
        "census_run_id": cap.get("census_run_id"),
        "census_input_fingerprint": cap.get("census_input_fingerprint"),
        "census_semantic_output_fingerprint": cap.get("census_semantic_output_fingerprint"),
        "census_cohort_fingerprint": cap.get("census_cohort_fingerprint"),
        "identity_source": cap.get("identity_source"),
        "identity_source_url": cap.get("identity_source_url"),
        "identity_as_of_date": cap.get("identity_as_of_date"),
        "identity_confidence": cap.get("identity_confidence"),
    }
    missing = sorted(key for key, value in required_values.items() if not str(value or "").strip())
    invalid = list(missing)
    if str(cap.get("as_of_date") or "")[:10] != str(as_of_date)[:10]:
        invalid.append("as_of_date")
    if str(cap.get("cap_source_kind") or "").strip().upper() != "ACCEPTED_CENSUS":
        invalid.append("cap_source_kind")
    if primary != upper:
        invalid.append("issuer_primary_ticker")
    if listed != (upper,):
        invalid.append("issuer_listed_tickers")
    if str(cap.get("security_role") or "").strip().upper() != "PRIMARY":
        invalid.append("security_role")
    if cap.get("is_secondary_class") is not False:
        invalid.append("is_secondary_class")
    if cap.get("is_adr") is not False:
        invalid.append("is_adr")
    identity_as_of = str(cap.get("identity_as_of_date") or "")[:10]
    cap_effective_as_of = str(cap.get("cap_effective_as_of_date") or "")[:10]
    if identity_as_of and identity_as_of > str(as_of_date)[:10]:
        invalid.append("identity_as_of_date")
    if not cap_effective_as_of or cap_effective_as_of > str(as_of_date)[:10]:
        invalid.append("cap_effective_as_of_date")
    if invalid:
        raise V2ExecutionBoundDriftError(
            f"Accepted-census identity drifted for {upper}: {','.join(sorted(set(invalid)))}"
        )

    return {
        "ticker": upper,
        "issuer_cik": cik,
        "issuer_primary_ticker": primary,
        "issuer_listed_tickers": list(listed),
        "security_role": "PRIMARY",
        "is_secondary_class": False,
        "is_adr": False,
        "adr_ratio": None,
        "share_class_ratio": None,
        "ratio_source_url": None,
        "ratio_source_accession": None,
        "ratio_security_symbol": None,
        "identity_source": cap.get("identity_source"),
        "identity_source_url": cap.get("identity_source_url"),
        "identity_as_of_date": identity_as_of,
        "identity_confidence": str(cap.get("identity_confidence") or "").upper(),
        "issuer_key": cap.get("issuer_key"),
        "security_key": cap.get("security_key"),
        "census_run_id": cap.get("census_run_id"),
        "census_input_fingerprint": cap.get("census_input_fingerprint"),
        "census_semantic_output_fingerprint": cap.get("census_semantic_output_fingerprint"),
        "census_cohort_fingerprint": cap.get("census_cohort_fingerprint"),
        "identity_authority": "accepted_census",
    }


def _resolved_security_identity_payload(
    ticker: str,
    *,
    as_of_date: str,
    db_path: Path,
    candidate_context: dict[str, Any] | None,
) -> dict[str, Any]:
    cap = _cap_stage_context(ticker, candidate_context)
    payload = _accepted_census_identity_payload(
        ticker,
        as_of_date=as_of_date,
        cap=cap,
    )
    if payload is None:
        from app.autonomous.cap_resolver import resolve_security_identity

        identity = resolve_security_identity(
            ticker,
            as_of_date=as_of_date,
            db_path=db_path,
            identity_evidence=cap or None,
        )
        payload = identity.to_dict()
    conn = _connect(db_path)
    try:
        scope = resolve_filing_issuer_scope(
            conn,
            ticker,
            issuer_cik=payload.get("issuer_cik"),
            aliases=tuple(payload.get("issuer_listed_tickers") or ()),
        )
    finally:
        conn.close()
    aliases = tuple(
        dict.fromkeys(
            [
                ticker,
                *(payload.get("issuer_listed_tickers") or ()),
                *(scope.aliases or ()),
            ]
        )
    )
    payload["issuer_aliases"] = list(aliases)
    payload["issuer_cik"] = payload.get("issuer_cik") or scope.issuer_cik
    payload["outcome"] = "RESOLVED" if payload.get("issuer_cik") else "UNRESOLVED"
    return payload


def _packet_fact_requirement_history(
    ticker: str,
    *,
    available_line_items: set[str],
    as_of_date: str,
    db_path: Path,
    cfg: AppConfig,
    issuer_cik: str | None,
    aliases: tuple[str, ...],
    candidate_context: dict[str, Any] | None,
    issuer_classification_hint: str | None = None,
) -> tuple[str, dict[str, int]]:
    if str(issuer_classification_hint or "").upper() == "BANK":
        return "FINANCIAL", dict(PACKET_REQUIRED_FINANCIAL_FACT_HISTORY)
    if {"deposits", "loans"} <= available_line_items:
        return "FINANCIAL", dict(PACKET_REQUIRED_FINANCIAL_FACT_HISTORY)

    sector = (
        str(candidate_context.get("sector") or "").strip().lower()
        if isinstance(candidate_context, dict)
        else ""
    )
    if sector in {"financial_services", "insurance"}:
        try:
            from app.insurance.routing import (
                ISSUER_INSURANCE_UNDERWRITER,
                SECURITY_COMMON,
                route_security,
            )

            routing = route_security(
                ticker,
                as_of_date=as_of_date,
                pipeline_version="v2",
                issuer_cik=issuer_cik,
                aliases=aliases,
                db_path=db_path,
                cfg=cfg,
            )
            if (
                routing.issuer_type == ISSUER_INSURANCE_UNDERWRITER
                and routing.security_type == SECURITY_COMMON
            ):
                return "INSURANCE_COMMON", dict(PACKET_REQUIRED_INSURANCE_FACT_HISTORY)
        except Exception:
            pass
    return "OPERATING", dict(PACKET_REQUIRED_NORMALIZED_FACT_HISTORY)


def _independent_issuer_classification_hint(
    conn: sqlite3.Connection,
    ticker: str,
    *,
    issuer_cik: str | None,
    candidate_context: dict[str, Any] | None,
) -> str | None:
    explicit: Any = None
    if isinstance(candidate_context, dict):
        by_ticker = candidate_context.get("issuer_classifications")
        if isinstance(by_ticker, dict):
            explicit = by_ticker.get(ticker)
        explicit = explicit or candidate_context.get("issuer_classification")
        cap = _cap_stage_context(ticker, candidate_context)
        explicit = explicit or cap.get("issuer_classification")
    normalized = str(explicit or "").strip().upper()
    if normalized in {"BANK", "INSURANCE", "OPERATING"}:
        return normalized

    cik = normalize_cik(issuer_cik)
    if cik is None:
        return None
    columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(sec_registrants)").fetchall()}
    required = {"cik", "name", "sic", "sic_description", "sector"}
    if required - columns:
        return None
    row = conn.execute(
        "SELECT name, sic, sic_description, sector FROM sec_registrants "
        "WHERE CAST(cik AS INTEGER) = ? LIMIT 1",
        (int(cik),),
    ).fetchone()
    if row is None:
        return None
    sic = str(row["sic"] or "").strip()
    text = " ".join(
        str(row[key] or "").strip().lower() for key in ("name", "sic_description", "sector")
    )
    bank_sics = {
        "6021",
        "6022",
        "6029",
        "6035",
        "6036",
        "6061",
        "6062",
        "6081",
        "6082",
        "6099",
    }
    if sic in bank_sics or any(
        token in text
        for token in (
            " bank",
            "bank ",
            "banking",
            "depository",
            "savings institution",
            "credit union",
        )
    ):
        return "BANK"
    if any(token in text for token in ("insurance", "insurer", "underwriter")):
        return "INSURANCE"
    return "OPERATING"


def _issuer_normalized_fact_coverage(
    conn: sqlite3.Connection,
    ticker: str,
    *,
    as_of_date: str,
    db_path: Path,
    cfg: AppConfig,
    candidate_context: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return issuer-aware normalized annual fact coverage for packet inputs."""

    identity = _resolved_security_identity_payload(
        ticker,
        as_of_date=as_of_date,
        db_path=db_path,
        candidate_context=candidate_context,
    )
    aliases = tuple(identity.get("issuer_aliases") or [ticker])
    scope = resolve_filing_issuer_scope(
        conn,
        ticker,
        issuer_cik=identity.get("issuer_cik"),
        aliases=aliases,
    )
    columns = {
        str(row[1]) for row in conn.execute("PRAGMA table_info(companyfacts_facts)").fetchall()
    }
    required_columns = {
        "ticker",
        "fiscal_year",
        "period_type",
        "period_end",
        "line_item",
        "value",
    }
    if required_columns - columns:
        default_required = dict(PACKET_REQUIRED_NORMALIZED_FACT_HISTORY)
        return {
            "outcome": "MISSING",
            "reason_code": "NORMALIZED_FACTS_TABLE_UNAVAILABLE",
            "issuer_cik": scope.issuer_cik,
            "aliases": list(scope.aliases),
            "required_line_items": list(default_required),
            "required_history_years_by_line_item": default_required,
            "missing_line_items": list(default_required),
            "insufficient_history_line_items": [],
            "annual_rows": 0,
            "annual_years": 0,
            "latest_period_end": None,
        }

    scope, rows = issuer_companyfacts_rows(
        conn,
        ticker,
        columns=("ticker", "fiscal_year", "period_end", "line_item"),
        issuer_cik=scope.issuer_cik,
        aliases=scope.aliases,
        period_types=("FY",),
        as_of_date=as_of_date,
        value_not_null=True,
        require_filed_asof=True,
        order_by="fiscal_year ASC, line_item ASC",
    )

    years_by_line_item: dict[str, set[int]] = {}
    source_tickers: set[str] = set()
    latest_period_end: str | None = None
    for row in rows:
        line_item = str(row["line_item"] or "")
        fiscal_year = row["fiscal_year"]
        if line_item and isinstance(fiscal_year, int):
            years_by_line_item.setdefault(line_item, set()).add(fiscal_year)
        source_ticker = str(row["ticker"] or "").strip().upper()
        if source_ticker:
            source_tickers.add(source_ticker)
        period_end = str(row["period_end"] or "")
        if period_end and (latest_period_end is None or period_end > latest_period_end):
            latest_period_end = period_end

    available = set(years_by_line_item)
    issuer_classification, required_history = _packet_fact_requirement_history(
        ticker,
        available_line_items=available,
        as_of_date=as_of_date,
        db_path=db_path,
        cfg=cfg,
        issuer_cik=scope.issuer_cik,
        aliases=scope.aliases,
        candidate_context=candidate_context,
        issuer_classification_hint=_independent_issuer_classification_hint(
            conn,
            ticker,
            issuer_cik=scope.issuer_cik,
            candidate_context=candidate_context,
        ),
    )
    required = tuple(required_history)
    missing = [line_item for line_item in required if line_item not in available]
    insufficient_history = [
        line_item
        for line_item in required
        if line_item in available
        and len(years_by_line_item[line_item]) < required_history[line_item]
    ]

    _filing_scope, annual_filings = issuer_filing_rows(
        conn,
        ticker,
        columns=("form_type",),
        issuer_cik=scope.issuer_cik,
        aliases=scope.aliases,
        form_types=ANNUAL_CACHED_FILING_FORM_TYPES,
        statuses=V2_CONTENT_AVAILABLE_FILING_STATUSES,
        as_of_date=as_of_date,
        limit=1,
    )
    latest_form = str(annual_filings[0]["form_type"] or "") if annual_filings else None
    foreign_gap_reason = None
    if missing or insufficient_history:
        foreign_gap_reason = foreign_normalized_facts_gap_reason(
            conn,
            ticker,
            issuer_cik=scope.issuer_cik,
            form_type=latest_form,
            as_of_date=as_of_date,
            aliases=scope.aliases,
            required_line_items=required,
            minimum_years=1,
            require_filed_asof=True,
        )

    if foreign_gap_reason:
        outcome = "NEEDS_DATA"
        reason_code = foreign_gap_reason
    elif missing:
        outcome = "NEEDS_DATA"
        reason_code = "NORMALIZED_FACTS_REQUIRED_LINE_ITEMS_MISSING"
    elif insufficient_history:
        outcome = "NEEDS_DATA"
        reason_code = "NORMALIZED_FACTS_HISTORY_INSUFFICIENT"
    else:
        outcome = "AVAILABLE"
        reason_code = None
    all_years = {year for years in years_by_line_item.values() for year in years}
    return {
        "outcome": outcome,
        "reason_code": reason_code,
        "issuer_cik": scope.issuer_cik,
        "aliases": list(scope.aliases),
        "source_tickers": sorted(source_tickers),
        "issuer_classification": issuer_classification,
        "latest_annual_form": latest_form,
        "required_line_items": list(required),
        "available_line_items": sorted(available),
        "missing_line_items": missing,
        "insufficient_history_line_items": insufficient_history,
        "required_history_years_by_line_item": required_history,
        "history_years_by_line_item": {
            line_item: sorted(years_by_line_item.get(line_item, set())) for line_item in required
        },
        "annual_rows": len(rows),
        "annual_years": len(all_years),
        "latest_period_end": latest_period_end,
    }


def _facts_source_attempt(
    *,
    source: str,
    result: dict[str, Any],
    as_of_date: str,
) -> dict[str, Any]:
    """Return the durable, JSON-safe evidence for one facts source attempt."""

    return {
        "source": source,
        "as_of_date": as_of_date,
        **{
            key: result.get(key)
            for key in (
                "outcome",
                "reason_code",
                "reason_detail",
                "terminal",
                "retryable",
                "issuer_cik",
                "storage_ticker",
                "source_url",
                "source_resolution",
                "network_attempted",
                "attempts_made",
                "normalized_rows",
                "visible_rows",
                "future_rows_rejected",
                "undated_rows_rejected",
                "rows_written",
                "vintages_written",
                "annual_rows",
                "annual_years",
                "latest_period_end",
                "missing_line_items",
                "insufficient_history_line_items",
            )
            if result.get(key) is not None
        },
    }


def _observe_v2_fact_coverage(
    ticker: str,
    *,
    as_of_date: str,
    db_path: Path,
    cfg: AppConfig,
    candidate_context: dict[str, Any] | None,
) -> dict[str, Any]:
    conn = _connect(db_path)
    try:
        return _issuer_normalized_fact_coverage(
            conn,
            ticker,
            as_of_date=as_of_date,
            db_path=db_path,
            cfg=cfg,
            candidate_context=candidate_context,
        )
    finally:
        conn.close()


def _resolve_v2_fact_availability(
    ticker: str,
    *,
    as_of_date: str,
    db_path: Path,
    cfg: AppConfig,
    candidate_context: dict[str, Any] | None,
    apply_repairs: bool,
) -> dict[str, Any]:
    """Observe, repair, and re-observe issuer-bound fixed-as-of facts.

    ``NEEDS_DATA`` is terminal only after the permitted source sequence has
    been exhausted (or repair is explicitly disabled). Provider/runtime
    failures return ``INCOMPLETE`` and remain on the durable retry frontier.
    """

    before = _observe_v2_fact_coverage(
        ticker,
        as_of_date=as_of_date,
        db_path=db_path,
        cfg=cfg,
        candidate_context=candidate_context,
    )
    source_attempts = [
        _facts_source_attempt(
            source="LOCAL_NORMALIZED_COMPANYFACTS",
            result=before,
            as_of_date=as_of_date,
        )
    ]
    if str(before.get("outcome") or "").upper() == "AVAILABLE":
        return {
            **before,
            "terminal": True,
            "retryable": False,
            "source_exhausted": False,
            "repair_policy": "LOCAL_EVIDENCE",
            "source_attempts": source_attempts,
            "coverage_before": before,
            "coverage_after": before,
        }

    if not apply_repairs:
        source_attempts.append(
            {
                "source": "SEC_COMPANYFACTS",
                "as_of_date": as_of_date,
                "outcome": "NOT_ATTEMPTED",
                "reason_code": "FACTS_REPAIR_DISABLED",
            }
        )
        return {
            **before,
            "outcome": "NEEDS_DATA",
            "terminal": True,
            "retryable": False,
            "source_exhausted": True,
            "repair_policy": "OBSERVE_ONLY",
            "source_attempts": source_attempts,
            "coverage_before": before,
            "coverage_after": before,
        }

    identity = _resolved_security_identity_payload(
        ticker,
        as_of_date=as_of_date,
        db_path=db_path,
        candidate_context=candidate_context,
    )
    issuer_cik = normalize_cik(identity.get("issuer_cik"))
    storage_ticker = str(identity.get("issuer_primary_ticker") or ticker).strip().upper()
    repair = repair_issuer_annual_companyfacts(
        ticker,
        issuer_cik=issuer_cik,
        as_of_date=as_of_date,
        db_path=db_path,
        cfg=cfg,
        storage_ticker=storage_ticker,
    )
    source_attempts.append(
        _facts_source_attempt(
            source="SEC_COMPANYFACTS",
            result=repair,
            as_of_date=as_of_date,
        )
    )

    if str(repair.get("outcome") or "").upper() != "FETCHED":
        terminal = bool(repair.get("terminal"))
        return {
            **before,
            "outcome": "NEEDS_DATA" if terminal else "INCOMPLETE",
            "reason_code": str(
                repair.get("reason_code")
                or before.get("reason_code")
                or "COMPANYFACTS_SOURCE_UNRESOLVED"
            ),
            "reason_detail": repair.get("reason_detail"),
            "terminal": terminal,
            "retryable": not terminal,
            "source_exhausted": terminal,
            "repair_policy": "LOCAL_THEN_SEC",
            "source_url": repair.get("source_url"),
            "source_resolution": repair.get("source_resolution"),
            "network_attempted": bool(repair.get("network_attempted")),
            "attempts_made": int(repair.get("attempts_made") or 0),
            "source_attempts": source_attempts,
            "coverage_before": before,
            "coverage_after": before,
            "repair_result": repair,
        }

    after = _observe_v2_fact_coverage(
        ticker,
        as_of_date=as_of_date,
        db_path=db_path,
        cfg=cfg,
        candidate_context=candidate_context,
    )
    available = str(after.get("outcome") or "").upper() == "AVAILABLE"
    return {
        **after,
        "outcome": "AVAILABLE" if available else "NEEDS_DATA",
        "reason_code": (
            None
            if available
            else str(after.get("reason_code") or "COMPANYFACTS_FETCHED_COVERAGE_INCOMPLETE")
        ),
        "terminal": True,
        "retryable": False,
        "source_exhausted": not available,
        "repair_policy": "LOCAL_THEN_SEC",
        "source_url": repair.get("source_url"),
        "source_resolution": repair.get("source_resolution"),
        "network_attempted": bool(repair.get("network_attempted")),
        "attempts_made": int(repair.get("attempts_made") or 0),
        "source_attempts": source_attempts,
        "coverage_before": before,
        "coverage_after": after,
        "repair_result": repair,
    }


def _latest_price_observation(
    conn: sqlite3.Connection,
    ticker: str,
    *,
    as_of_date: str,
) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT price, currency, provider, as_of_date, source_url, status
        FROM price_quotes
        WHERE ticker = ? AND as_of_date <= ? AND price IS NOT NULL
        ORDER BY as_of_date DESC, id DESC
        LIMIT 1
        """,
        (ticker, as_of_date),
    ).fetchone()
    if row is None:
        return {
            "outcome": "MISSING",
            "reason_code": "PRICE_QUOTE_NOT_FOUND_AS_OF_DATE",
            "terminal": False,
            "retryable": True,
            "ticker": ticker,
            "requested_as_of_date": as_of_date,
        }
    return {
        "outcome": "AVAILABLE",
        "reason_code": None,
        "terminal": True,
        "retryable": False,
        "price": row["price"],
        "currency": row["currency"],
        "provider": row["provider"],
        "as_of_date": row["as_of_date"],
        "source_url": sanitize_url_credentials(row["source_url"]),
        "quote_status": row["status"],
    }


def _filing_evidence_snapshot(
    ticker: str,
    *,
    as_of_date: str,
    db_path: Path,
    candidate_context: dict[str, Any] | None,
) -> dict[str, Any]:
    identity = _resolved_security_identity_payload(
        ticker,
        as_of_date=as_of_date,
        db_path=db_path,
        candidate_context=candidate_context,
    )
    conn = _connect(db_path)
    try:
        scope, rows = issuer_filing_rows(
            conn,
            ticker,
            columns=(
                "accession",
                "form_type",
                "filing_date",
                "local_path",
                "hash",
                "status",
                "updated_at",
            ),
            issuer_cik=identity.get("issuer_cik"),
            aliases=identity.get("issuer_aliases") or (),
            form_types=ANNUAL_CACHED_FILING_FORM_TYPES,
            statuses=None,
            as_of_date=as_of_date,
        )
    finally:
        conn.close()
    evidence_rows: list[dict[str, Any]] = []
    for row in rows:
        local_path = str(row["local_path"] or "")
        size: int | None = None
        modified_ns: int | None = None
        if local_path:
            try:
                stat = Path(local_path).stat()
                size = int(stat.st_size)
                modified_ns = int(stat.st_mtime_ns)
            except OSError:
                pass
        evidence_rows.append(
            {
                "accession": row["accession"],
                "form_type": row["form_type"],
                "filing_date": row["filing_date"],
                "local_path": local_path or None,
                "content_hash": row["hash"],
                "status": row["status"],
                "updated_at": row["updated_at"],
                "file_size": size,
                "file_modified_ns": modified_ns,
            }
        )
    return {
        "issuer_cik": scope.issuer_cik,
        "aliases": list(scope.aliases),
        "filings": evidence_rows,
    }


def _issuer_annual_parsing_state(
    conn: sqlite3.Connection,
    ticker: str,
    *,
    issuer_cik: str,
    aliases: tuple[str, ...],
    as_of_date: str,
) -> dict[str, Any]:
    scope, rows = issuer_filing_rows(
        conn,
        ticker,
        columns=("accession", "status", "local_path", "form_type"),
        issuer_cik=issuer_cik,
        aliases=aliases,
        form_types=ANNUAL_CACHED_FILING_FORM_TYPES,
        statuses=None,
        as_of_date=as_of_date,
    )
    parsed_accessions: set[str] = set()
    try:
        parsed_accessions = {
            str(row[0])
            for row in conn.execute("SELECT accession FROM parsed_filings").fetchall()
            if row[0]
        }
    except sqlite3.DatabaseError:
        pass
    complete: list[str] = []
    pending: list[str] = []
    failed: list[str] = []
    for row in rows:
        accession = str(row["accession"] or "")
        status = str(row["status"] or "").strip().lower()
        if accession in parsed_accessions or status in {"ok", "parsed"}:
            complete.append(accession)
        elif status in {"downloaded", "new"}:
            pending.append(accession)
        elif status in {"download_error", "parse_skipped"}:
            failed.append(accession)
    return {
        "issuer_cik": scope.issuer_cik,
        "annual_record_count": len(rows),
        "complete_accessions": complete,
        "pending_accessions": pending,
        "failed_accessions": failed,
    }


def _valuation_evidence_snapshot(
    conn: sqlite3.Connection,
    ticker: str,
    *,
    as_of_date: str,
) -> dict[str, Any]:
    """Return only the newest exact-source-authorized scorecard snapshot.

    Candidate selection precedes authorization inside the central selector.
    Therefore an unaudited or tampered newest row produces an empty snapshot
    and cannot silently fall back to an older authorized scorecard.
    """

    row = latest_decision_eligible_valuation_row(
        conn,
        ticker=ticker,
        method="scorecard",
        as_of_date=as_of_date,
    )
    if row is None:
        return {}
    return {
        field: row[field]
        for field in (
            "id",
            "as_of_date",
            "created_at",
            "outputs_json",
            "quality_gate_verdict",
            "source_run_id",
            "source_artifact_path",
            "source_artifact_sha256",
            "financial_integrity_fingerprint",
        )
    }


def _persisted_cap_evidence_snapshot(
    conn: sqlite3.Connection,
    ticker: str,
    *,
    as_of_date: str,
) -> dict[str, Any]:
    columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(market_caps)").fetchall()}
    required = {
        "ticker",
        "effective_as_of_date",
        "market_cap",
        "market_cap_status",
    }
    if required - columns:
        return {}
    selected = [
        name
        for name in (
            "id",
            "effective_as_of_date",
            "market_cap",
            "market_cap_status",
            "provider",
            "source_url",
            "payload_json",
        )
        if name in columns
    ]
    rows = conn.execute(
        f"SELECT {', '.join(selected)} FROM market_caps "
        "WHERE UPPER(ticker) = ? AND effective_as_of_date <= ? "
        "ORDER BY effective_as_of_date DESC"
        + (", id DESC" if "id" in columns else "")
        + " LIMIT 5",
        (ticker, as_of_date),
    ).fetchall()
    return {"rows": [dict(row) for row in rows]}


def _stage_evidence_fingerprint(
    stage: str,
    *,
    ticker: str,
    as_of_date: str,
    db_path: Path,
    cfg: AppConfig,
    candidate_context: dict[str, Any] | None,
) -> str:
    """Fingerprint the read-side evidence a completed stage depended on."""

    if stage == "MEMBERSHIP":
        snapshot: Any = _membership_stage_context(ticker, candidate_context)
    elif stage == "IDENTITY":
        snapshot = _resolved_security_identity_payload(
            ticker,
            as_of_date=as_of_date,
            db_path=db_path,
            candidate_context=candidate_context,
        )
    elif stage == "CAP":
        conn = _connect(db_path)
        try:
            snapshot = {
                "classification": _cap_stage_context(ticker, candidate_context),
                "persisted_cap_evidence": _persisted_cap_evidence_snapshot(
                    conn,
                    ticker,
                    as_of_date=as_of_date,
                ),
            }
        finally:
            conn.close()
    elif stage == "FACTS_AVAILABILITY":
        conn = _connect(db_path)
        try:
            snapshot = _issuer_normalized_fact_coverage(
                conn,
                ticker,
                as_of_date=as_of_date,
                db_path=db_path,
                cfg=cfg,
                candidate_context=candidate_context,
            )
        finally:
            conn.close()
    elif stage in {"FILINGS", "PARSING"}:
        snapshot = _filing_evidence_snapshot(
            ticker,
            as_of_date=as_of_date,
            db_path=db_path,
            candidate_context=candidate_context,
        )
    elif stage == "PRICE":
        conn = _connect(db_path)
        try:
            snapshot = {
                "cap": _cap_stage_context(ticker, candidate_context),
                "quote": _latest_price_observation(
                    conn,
                    ticker,
                    as_of_date=as_of_date,
                ),
            }
        finally:
            conn.close()
    elif stage == "VALUATION":
        conn = _connect(db_path)
        try:
            snapshot = _valuation_evidence_snapshot(
                conn,
                ticker,
                as_of_date=as_of_date,
            )
        finally:
            conn.close()
    elif stage == "PACKET":
        snapshot = {
            dependency: _stage_evidence_fingerprint(
                dependency,
                ticker=ticker,
                as_of_date=as_of_date,
                db_path=db_path,
                cfg=cfg,
                candidate_context=candidate_context,
            )
            for dependency in V2_REPAIR_STAGE_SEQUENCE[:-1]
        }
    else:
        raise ValueError(f"Unknown v2 repair stage: {stage}")
    return _json_fingerprint(snapshot)


def _execute_v2_repair_stage(
    stage: str,
    *,
    ticker: str,
    as_of_date: str,
    db_path: Path,
    cfg: AppConfig,
    candidate_context: dict[str, Any] | None,
    apply_repairs: bool,
    run_id: str | None = None,
    prior_stage_results: dict[str, dict[str, Any]] | None = None,
    terminal_cap_search: Any | None = None,
    price_provider: Any | None = None,
    filing_windows_days: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Execute or observe one idempotent stage of the v2 repair sequence."""

    if stage == "MEMBERSHIP":
        return _membership_stage_context(ticker, candidate_context)

    if stage == "IDENTITY":
        repair_policy = "LOCAL_THEN_CIK_REGISTRY" if apply_repairs else "LOCAL_EVIDENCE_ONLY"
        identity = _resolved_security_identity_payload(
            ticker,
            as_of_date=as_of_date,
            db_path=db_path,
            candidate_context=candidate_context,
        )
        if identity.get("outcome") == "UNRESOLVED" and apply_repairs:
            ensure_company_row(ticker, db_path=db_path)
            identity = _resolved_security_identity_payload(
                ticker,
                as_of_date=as_of_date,
                db_path=db_path,
                candidate_context=candidate_context,
            )
        if identity.get("outcome") != "RESOLVED":
            return {
                **identity,
                "outcome": "NEEDS_DATA",
                "reason_code": "ISSUER_IDENTITY_UNRESOLVED",
                "terminal": True,
                "retryable": False,
                "source_exhausted": True,
                "repair_policy": repair_policy,
            }
        return {
            **identity,
            "terminal": True,
            "retryable": False,
            "source_exhausted": False,
            "repair_policy": repair_policy,
        }

    if stage == "CAP":
        repair_policy = "LOCAL_THEN_TERMINAL_SEARCH" if apply_repairs else "LOCAL_EVIDENCE_ONLY"
        cap = _cap_stage_context(ticker, candidate_context)
        resolved = isinstance(cap.get("market_cap_mm"), (int, float))
        if (
            not resolved
            and apply_repairs
            and _terminal_cap_search_is_authorized(terminal_cap_search)
        ):
            identity_payload = _resolved_security_identity_payload(
                ticker,
                as_of_date=as_of_date,
                db_path=db_path,
                candidate_context=candidate_context,
            )
            identity = _security_identity_from_repair_payload(ticker, identity_payload)
            search_result = terminal_cap_search(ticker, as_of_date, identity)
            search_usage = [
                {
                    **dict(record),
                    "ticker": ticker,
                    "attempt_status": getattr(search_result, "status", "FAILED"),
                    "reason_code": getattr(
                        search_result,
                        "reason_code",
                        "INVALID_TERMINAL_CAP_SEARCH_RESULT",
                    ),
                }
                for record in getattr(search_result, "usage_records", ())
                if isinstance(record, dict)
            ]
            search_metadata = {
                "status": getattr(search_result, "status", "FAILED"),
                "reason_code": getattr(
                    search_result,
                    "reason_code",
                    "INVALID_TERMINAL_CAP_SEARCH_RESULT",
                ),
                "response_id": getattr(search_result, "response_id", None),
                "web_search_call_count": int(
                    getattr(search_result, "web_search_call_count", 0) or 0
                ),
                "cited_source_urls": list(getattr(search_result, "cited_source_urls", ()) or ()),
                "cost_estimate_usd": float(getattr(search_result, "total_cost_usd", 0.0) or 0.0),
                "ledger_path": str(getattr(terminal_cap_search, "ledger_path", "") or "") or None,
            }
            evidence = getattr(search_result, "evidence", None)
            if evidence is not None:
                # Search evidence is accepted only after the durable ledger's
                # generic loader reconstructs the same issuer-bound record.
                from app.autonomous.terminal_cap_evidence import (
                    terminal_cap_lookup_from_path,
                )

                ledger_path = getattr(terminal_cap_search, "ledger_path", None)
                validated = (
                    list(
                        terminal_cap_lookup_from_path(ledger_path)(
                            ticker,
                            as_of_date,
                            identity,
                        )
                    )
                    if ledger_path
                    else []
                )
                evidence = next(
                    (
                        row
                        for row in validated
                        if row.source_url == evidence.source_url
                        and row.as_of_date == evidence.as_of_date
                        and abs(float(row.market_cap_mm) - float(evidence.market_cap_mm)) < 1e-9
                    ),
                    None,
                )
                if evidence is None:
                    return {
                        "outcome": "INCOMPLETE",
                        "reason_code": "TERMINAL_CAP_EVIDENCE_LEDGER_VALIDATION_FAILED",
                        "terminal": False,
                        "retryable": True,
                        "source_exhausted": False,
                        "repair_policy": repair_policy,
                        "market_cap_mm": None,
                        "terminal_cap_search": search_metadata,
                        "usage_records": search_usage,
                        "actions": ["TERMINAL_CAP_SEARCH_ATTEMPTED"],
                    }
                cap = _classification_from_terminal_cap_evidence(
                    ticker=ticker,
                    as_of_date=as_of_date,
                    identity=identity,
                    evidence=evidence,
                )
                if isinstance(candidate_context, dict):
                    classifications = candidate_context.setdefault("cap_classifications", {})
                    if isinstance(classifications, dict):
                        classifications[ticker] = dict(cap)
                resolved = True
                search_metadata["evidence"] = {
                    "issuer_cik": evidence.issuer_cik,
                    "market_cap_mm": evidence.market_cap_mm,
                    "source_kind": evidence.source_kind,
                    "source_name": evidence.source_name,
                    "source_url": evidence.source_url,
                    "as_of_date": evidence.as_of_date,
                    "confidence": evidence.confidence,
                }
                search_metadata["ledger_validation_status"] = "VALIDATED"
            if not resolved:
                search_status = str(search_metadata["status"]).upper()
                search_not_found = search_status == "NOT_FOUND"
                return {
                    "outcome": "NEEDS_DATA" if search_not_found else "INCOMPLETE",
                    "reason_code": str(search_metadata["reason_code"]),
                    "terminal": search_not_found,
                    "retryable": not search_not_found,
                    "source_exhausted": search_not_found,
                    "repair_policy": repair_policy,
                    "market_cap_mm": None,
                    "terminal_cap_search": search_metadata,
                    "usage_records": search_usage,
                    "actions": ["TERMINAL_CAP_SEARCH_ATTEMPTED"],
                }
        elif not resolved and apply_repairs:
            return {
                "outcome": "INCOMPLETE",
                "reason_code": "MARKET_CAP_UNRESOLVED_TERMINAL_SEARCH_NOT_AUTHORIZED",
                "terminal": False,
                "retryable": True,
                "source_exhausted": False,
                "repair_policy": repair_policy,
                "market_cap_mm": None,
                "terminal_cap_search": {
                    "status": "NOT_AUTHORIZED",
                    "reason_code": "WHOLE_RUN_COST_PREFLIGHT_AUTHORIZATION_REQUIRED",
                },
                "usage_records": [],
                "actions": [],
            }
        search_result_payload = search_metadata if "search_metadata" in locals() else None
        search_usage_payload = search_usage if "search_usage" in locals() else []
        return {
            "outcome": "RESOLVED" if resolved else "NEEDS_DATA",
            "reason_code": (None if resolved else "MARKET_CAP_UNRESOLVED_AFTER_SOURCE_EXHAUSTION"),
            "terminal": True,
            "retryable": False,
            "source_exhausted": not resolved,
            "repair_policy": repair_policy,
            "market_cap_mm": cap.get("market_cap_mm"),
            "cap_band": cap.get("cap_band") or cap.get("band_label"),
            "cap_source": cap.get("cap_source"),
            "price_used": cap.get("price_used"),
            "shares_mm": cap.get("shares_mm"),
            "as_of_date": cap.get("as_of_date"),
            "cap_effective_as_of_date": cap.get("cap_effective_as_of_date"),
            "cap_source_kind": cap.get("cap_source_kind"),
            "cap_source_name": cap.get("cap_source_name"),
            "cap_source_url": cap.get("cap_source_url"),
            "cap_confidence": cap.get("cap_confidence"),
            "detail": cap.get("detail"),
            "terminal_cap_search": search_result_payload,
            "usage_records": search_usage_payload,
            "actions": (
                ["TERMINAL_CAP_SEARCH_ATTEMPTED", "TERMINAL_CAP_EVIDENCE_PERSISTED"]
                if search_result_payload is not None and resolved
                else []
            ),
        }

    if stage == "FACTS_AVAILABILITY":
        return _resolve_v2_fact_availability(
            ticker,
            as_of_date=as_of_date,
            db_path=db_path,
            cfg=cfg,
            candidate_context=candidate_context,
            apply_repairs=apply_repairs,
        )

    if stage == "FILINGS":
        repair_policy = "LOCAL_THEN_SEC" if apply_repairs else "LOCAL_EVIDENCE_ONLY"
        identity = _resolved_security_identity_payload(
            ticker,
            as_of_date=as_of_date,
            db_path=db_path,
            candidate_context=candidate_context,
        )
        issuer_cik = identity.get("issuer_cik")
        aliases = tuple(identity.get("issuer_aliases") or ())
        if not issuer_cik:
            return {
                "outcome": "NEEDS_DATA",
                "reason_code": "ISSUER_CIK_UNRESOLVED",
                "terminal": True,
                "retryable": False,
                "source_exhausted": True,
                "repair_policy": repair_policy,
                "had_readable_annual_filing": False,
                "actions": [],
                "source_attempts": [],
            }
        conn = _connect(db_path)
        try:
            had_filing = _has_readable_issuer_annual_filing(
                conn,
                ticker,
                as_of_date=as_of_date,
                issuer_cik=issuer_cik,
                aliases=aliases,
                persist_recovered=apply_repairs,
            )
        finally:
            conn.close()
        if had_filing:
            return {
                "outcome": "AVAILABLE",
                "reason_code": None,
                "terminal": True,
                "retryable": False,
                "source_exhausted": False,
                "repair_policy": repair_policy,
                "had_readable_annual_filing": True,
                "actions": [],
                "source_attempts": [
                    {
                        "source": "LOCAL_ANNUAL_FILING_CACHE",
                        "outcome": "AVAILABLE",
                        "reason_code": None,
                    }
                ],
            }
        if not apply_repairs:
            return {
                "outcome": "NEEDS_DATA",
                "reason_code": "ANNUAL_FILING_NOT_CACHED",
                "terminal": True,
                "retryable": False,
                "source_exhausted": True,
                "repair_policy": repair_policy,
                "had_readable_annual_filing": False,
                "actions": [],
                "source_attempts": [
                    {
                        "source": "LOCAL_ANNUAL_FILING_CACHE",
                        "outcome": "MISSING",
                        "reason_code": "ANNUAL_FILING_NOT_CACHED",
                    },
                    {
                        "source": "SEC_FILING_INGEST",
                        "outcome": "NOT_ATTEMPTED",
                        "reason_code": "FILING_REPAIR_DISABLED",
                    },
                ],
            }
        actions: list[str] = []
        filing_storage_ticker = str(identity.get("issuer_primary_ticker") or ticker).strip().upper()
        if not ensure_company_row(
            filing_storage_ticker,
            db_path=db_path,
            issuer_cik=issuer_cik,
        ):
            return {
                "outcome": "NEEDS_DATA",
                "reason_code": "ISSUER_CIK_CONFLICT",
                "terminal": True,
                "retryable": False,
                "source_exhausted": True,
                "repair_policy": repair_policy,
                "had_readable_annual_filing": False,
                "actions": actions,
                "source_attempts": [],
            }
        from app.ingest.filings import ingest_with_policy

        ingest_result = ingest_with_policy(
            as_of_date=as_of_date,
            run_id=f"evidence_resolution_{str(as_of_date).replace('-', '')}",
            tickers=[filing_storage_ticker],
            db_path=db_path,
            windows_days=filing_windows_days,
        )
        actions.append("FILING_INGEST_ATTEMPTED")
        conn = _connect(db_path)
        try:
            has_filing_after = _has_readable_issuer_annual_filing(
                conn,
                ticker,
                as_of_date=as_of_date,
                issuer_cik=issuer_cik,
                aliases=aliases,
                persist_recovered=apply_repairs,
            )
        finally:
            conn.close()
        filing_snapshot = _filing_evidence_snapshot(
            ticker,
            as_of_date=as_of_date,
            db_path=db_path,
            candidate_context=candidate_context,
        )
        evidence_rows = filing_snapshot.get("filings") or []
        download_failed = bool(int(ingest_result.get("filing_download_errors") or 0)) or any(
            str(row.get("status") or "").lower() == "download_error"
            for row in evidence_rows
            if isinstance(row, dict)
        )
        reason_code = (
            None
            if has_filing_after
            else "ANNUAL_FILING_DOWNLOAD_FAILED"
            if download_failed
            else "ANNUAL_FILING_CONTENT_UNAVAILABLE"
            if evidence_rows
            else "ANNUAL_FILING_NOT_FOUND_AS_OF_DATE"
        )
        outcome = (
            "AVAILABLE" if has_filing_after else "INCOMPLETE" if download_failed else "NEEDS_DATA"
        )
        return {
            "outcome": outcome,
            "reason_code": reason_code,
            "terminal": not download_failed,
            "retryable": download_failed,
            "source_exhausted": not has_filing_after and not download_failed,
            "repair_policy": repair_policy,
            "had_readable_annual_filing": has_filing_after,
            "actions": actions,
            "ingest_result": ingest_result,
            "filing_evidence": filing_snapshot,
            "source_attempts": [
                {
                    "source": "LOCAL_ANNUAL_FILING_CACHE",
                    "outcome": "MISSING",
                    "reason_code": "ANNUAL_FILING_NOT_CACHED",
                },
                {
                    "source": "SEC_FILING_INGEST",
                    "outcome": outcome,
                    "reason_code": reason_code,
                    "filings_considered": int(ingest_result.get("filings_considered") or 0),
                    "filings_upserted": int(ingest_result.get("filings_upserted") or 0),
                },
            ],
        }

    if stage == "PARSING":
        repair_policy = "LOCAL_THEN_PARSE" if apply_repairs else "LOCAL_EVIDENCE_ONLY"
        identity = _resolved_security_identity_payload(
            ticker,
            as_of_date=as_of_date,
            db_path=db_path,
            candidate_context=candidate_context,
        )
        issuer_cik = identity.get("issuer_cik")
        aliases = tuple(identity.get("issuer_aliases") or (ticker,))
        if not issuer_cik:
            return {
                "outcome": "NEEDS_DATA",
                "reason_code": "ISSUER_CIK_UNRESOLVED",
                "terminal": True,
                "retryable": False,
                "source_exhausted": True,
                "repair_policy": repair_policy,
                "readable_before": False,
                "readable_after": False,
                "parsed_count": 0,
                "actions": [],
            }
        conn = _connect(db_path)
        try:
            content_available_before = _has_readable_issuer_annual_filing(
                conn,
                ticker,
                as_of_date=as_of_date,
                issuer_cik=issuer_cik,
                aliases=aliases,
                persist_recovered=apply_repairs,
            )
            parsing_before = _issuer_annual_parsing_state(
                conn,
                ticker,
                issuer_cik=str(issuer_cik),
                aliases=aliases,
                as_of_date=as_of_date,
            )
        finally:
            conn.close()
        parsed_before = bool(parsing_before["complete_accessions"])
        if content_available_before and parsed_before:
            return {
                "outcome": "READABLE",
                "reason_code": None,
                "terminal": True,
                "retryable": False,
                "source_exhausted": False,
                "repair_policy": repair_policy,
                "readable_before": True,
                "readable_after": True,
                "content_available_before": True,
                "content_available_after": True,
                "parsing_state_before": parsing_before,
                "parsing_state_after": parsing_before,
                "parsed_count": 0,
                "actions": [],
            }
        parsed = 0
        actions: list[str] = []
        if apply_repairs:
            from app.parse.filing_parser import parse_pending_filings

            parsed = int(
                parse_pending_filings(
                    limit=10,
                    tickers=list(aliases),
                    db_path=db_path,
                    issuer_cik=issuer_cik,
                    form_types=ANNUAL_CACHED_FILING_FORM_TYPES,
                    as_of_date=as_of_date,
                )
                or 0
            )
            if parsed:
                actions.append(f"FILINGS_PARSED:{parsed}")
        conn = _connect(db_path)
        try:
            content_available_after = _has_readable_issuer_annual_filing(
                conn,
                ticker,
                as_of_date=as_of_date,
                issuer_cik=issuer_cik,
                aliases=aliases,
                persist_recovered=apply_repairs,
            )
            parsing_after = _issuer_annual_parsing_state(
                conn,
                ticker,
                issuer_cik=str(issuer_cik),
                aliases=aliases,
                as_of_date=as_of_date,
            )
        finally:
            conn.close()
        readable_after = bool(content_available_after and parsing_after["complete_accessions"])
        filing_snapshot = _filing_evidence_snapshot(
            ticker,
            as_of_date=as_of_date,
            db_path=db_path,
            candidate_context=candidate_context,
        )
        evidence_rows = filing_snapshot.get("filings") or []
        reason_code = (
            None
            if readable_after
            else "ANNUAL_FILING_RECORD_MISSING"
            if not evidence_rows and not parsing_after["annual_record_count"]
            else "ANNUAL_FILING_EXTRACTION_GAP"
            if apply_repairs or content_available_after
            else "ANNUAL_FILING_PARSE_NOT_ATTEMPTED"
        )
        return {
            "outcome": "READABLE" if readable_after else "NEEDS_DATA",
            "reason_code": reason_code,
            "terminal": True,
            "retryable": False,
            "source_exhausted": not readable_after,
            "repair_policy": repair_policy,
            "readable_before": bool(content_available_before and parsed_before),
            "readable_after": readable_after,
            "content_available_before": content_available_before,
            "content_available_after": content_available_after,
            "parsing_state_before": parsing_before,
            "parsing_state_after": parsing_after,
            "parsed_count": parsed,
            "actions": actions,
            "filing_evidence": filing_snapshot,
        }

    if stage == "PRICE":
        cap_stage_price = _v2_cap_price_payload(ticker, candidate_context)
        resolution = resolve_v2_price(
            ticker,
            as_of_date=as_of_date,
            db_path=db_path,
            cap_stage_price=cap_stage_price,
            run_id=run_id,
            cfg=cfg,
            provider=price_provider,
            allow_provider=apply_repairs,
            persist=apply_repairs,
        )
        payload = resolution.to_dict()
        snapshot = payload.get("snapshot") if isinstance(payload.get("snapshot"), dict) else {}
        status = str(payload.get("status") or "INCOMPLETE").upper()
        outcome = (
            "AVAILABLE"
            if status == "RESOLVED"
            else "NEEDS_DATA"
            if status == "NEEDS_DATA"
            else "INCOMPLETE"
        )
        return {
            "outcome": outcome,
            "reason_code": payload.get("reason_code"),
            "terminal": status != "INCOMPLETE",
            "retryable": status == "INCOMPLETE",
            "source_exhausted": status == "NEEDS_DATA",
            "repair_policy": ("LOCAL_THEN_PROVIDER" if apply_repairs else "LOCAL_EVIDENCE_ONLY"),
            "price": snapshot.get("price"),
            "currency": snapshot.get("currency"),
            "provider": snapshot.get("source"),
            "source": payload.get("source_resolution"),
            "as_of_date": snapshot.get("as_of_date"),
            "source_url": snapshot.get("url"),
            "confidence": snapshot.get("confidence"),
            "snapshot": snapshot or None,
            "source_attempts": list(payload.get("attempts") or []),
            "persisted": bool(payload.get("persisted")),
            "persistence_error": payload.get("persistence_error"),
            "actions": (["PRICE_SNAPSHOT_PERSISTED"] if payload.get("persisted") else []),
        }

    if stage == "VALUATION":
        repair_policy = "LOCAL_THEN_REFRESH" if apply_repairs else "LOCAL_EVIDENCE_ONLY"
        identity = _resolved_security_identity_payload(
            ticker,
            as_of_date=as_of_date,
            db_path=db_path,
            candidate_context=candidate_context,
        )
        issuer_cik = identity.get("issuer_cik")
        issuer_aliases = tuple(identity.get("issuer_aliases") or (ticker,))
        conn = _connect(db_path)
        try:
            fact_coverage = _issuer_normalized_fact_coverage(
                conn,
                ticker,
                as_of_date=as_of_date,
                db_path=db_path,
                cfg=cfg,
                candidate_context=candidate_context,
            )
        finally:
            conn.close()
        price_resolution = _resolve_v2_local_stage_price(
            ticker,
            as_of_date=as_of_date,
            db_path=db_path,
            cfg=cfg,
            candidate_context=candidate_context,
            run_id=run_id,
            prior_stage_results=prior_stage_results,
        )
        price_snapshot = (
            price_resolution.snapshot if price_resolution.status == "RESOLVED" else None
        )
        missing_inputs: list[str] = []
        if fact_coverage.get("outcome") != "AVAILABLE":
            missing_inputs.append("FACTS")
        if price_snapshot is None:
            missing_inputs.append("PRICE")
        conn = _connect(db_path)
        try:
            before_state = _v2_scorecard_provenance_state(
                conn,
                ticker,
                as_of_date=as_of_date,
                issuer_cik=issuer_cik,
                issuer_aliases=issuer_aliases,
                price_snapshot=price_snapshot,
            )
        finally:
            conn.close()
        before = before_state["validated_asof"]
        actions: list[str] = []
        refresh_exception: Exception | None = None
        refresh_attempted = False
        written_valuation_source_records: list[dict[str, Any]] = []
        if apply_repairs and not missing_inputs:
            from app.valuation.valuation_writer import ensure_valuation

            refresh_attempted = True
            actions.append("VALUATION_REFRESH_ATTEMPTED")
            try:
                written_valuation_source_records = ensure_valuation(
                    ticker,
                    as_of_date,
                    run_id=run_id,
                    price_override=float(price_snapshot.price),
                    force_refresh=True,
                    cfg=cfg,
                    db_path=db_path,
                    raise_on_error=True,
                    issuer_cik=issuer_cik,
                    issuer_aliases=issuer_aliases,
                    require_filed_asof=True,
                    price_provenance={
                        **_as_plain_dict(price_snapshot),
                        "source_resolution": price_resolution.source_resolution,
                    },
                )
            except Exception as exc:  # noqa: BLE001 - checkpointed retry boundary
                refresh_exception = exc
        conn = _connect(db_path)
        try:
            after_state = _v2_scorecard_provenance_state(
                conn,
                ticker,
                as_of_date=as_of_date,
                issuer_cik=issuer_cik,
                issuer_aliases=issuer_aliases,
                price_snapshot=price_snapshot,
            )
        finally:
            conn.close()
        after = after_state["validated_asof"]
        raw_after = after_state["raw_asof"]
        valuation_status = "FRESH" if after else "PROVENANCE_MISMATCH" if raw_after else "MISSING"
        if (
            refresh_exception is None
            and refresh_attempted
            and valuation_status == "FRESH"
            and not written_valuation_source_records
        ):
            refresh_exception = RuntimeError(
                "valuation refresh produced no controlled source records"
            )
        if refresh_exception is not None:
            return {
                "outcome": "INCOMPLETE",
                "reason_code": "VALUATION_REFRESH_EXCEPTION",
                "terminal": False,
                "retryable": True,
                "source_exhausted": False,
                "repair_policy": repair_policy,
                "valuation_status": valuation_status,
                "scorecard_asof_before": before,
                "scorecard_asof_after": after,
                "raw_scorecard_asof_before": before_state["raw_asof"],
                "raw_scorecard_asof_after": raw_after,
                "provenance_mismatches_before": before_state["mismatch_reasons"],
                "provenance_mismatches_after": after_state["mismatch_reasons"],
                "is_stale_after": bool(raw_after and _scorecard_is_stale(raw_after, as_of_date)),
                "missing_inputs": missing_inputs,
                "refresh_attempted": refresh_attempted,
                "error_type": type(refresh_exception).__name__,
                "error_detail": str(refresh_exception),
                "actions": actions,
                "valuation_source_records": [],
            }
        if valuation_status == "FRESH" and refresh_attempted:
            actions.append("VALUATION_REFRESHED")
        reason_code = None
        if valuation_status != "FRESH":
            reason_code = (
                "VALUATION_INPUTS_MISSING"
                if missing_inputs
                else "VALUATION_PROVENANCE_MISMATCH_AFTER_REFRESH"
                if valuation_status == "PROVENANCE_MISMATCH" and refresh_attempted
                else "VALUATION_MISSING_AFTER_REFRESH"
                if valuation_status == "MISSING" and refresh_attempted
                else "VALUATION_PROVENANCE_MISMATCH"
                if valuation_status == "PROVENANCE_MISMATCH"
                else "VALUATION_MISSING"
            )
        return {
            "outcome": "FRESH" if valuation_status == "FRESH" else "NEEDS_DATA",
            "reason_code": reason_code,
            "terminal": True,
            "retryable": False,
            "source_exhausted": valuation_status != "FRESH",
            "repair_policy": repair_policy,
            "valuation_status": valuation_status,
            "scorecard_asof_before": before,
            "scorecard_asof_after": after,
            "raw_scorecard_asof_before": before_state["raw_asof"],
            "raw_scorecard_asof_after": raw_after,
            "provenance_mismatches_before": before_state["mismatch_reasons"],
            "provenance_mismatches_after": after_state["mismatch_reasons"],
            "is_stale_after": bool(raw_after and _scorecard_is_stale(raw_after, as_of_date)),
            "missing_inputs": missing_inputs,
            "refresh_attempted": refresh_attempted,
            "price_source": (
                price_resolution.source_resolution if price_snapshot is not None else None
            ),
            "actions": actions,
            "valuation_source_records": written_valuation_source_records,
        }

    if stage == "PACKET":
        identity = _resolved_security_identity_payload(
            ticker,
            as_of_date=as_of_date,
            db_path=db_path,
            candidate_context=candidate_context,
        )
        issuer_cik = identity.get("issuer_cik")
        aliases = tuple(identity.get("issuer_aliases") or ())
        conn = _connect(db_path)
        try:
            has_filing = _has_readable_issuer_annual_filing(
                conn,
                ticker,
                as_of_date=as_of_date,
                issuer_cik=issuer_cik,
                aliases=aliases,
                persist_recovered=apply_repairs,
            )
            facts = _issuer_normalized_fact_coverage(
                conn,
                ticker,
                as_of_date=as_of_date,
                db_path=db_path,
                cfg=cfg,
                candidate_context=candidate_context,
            )
        finally:
            conn.close()
        cap = _cap_stage_context(ticker, candidate_context)
        price_resolution = _resolve_v2_local_stage_price(
            ticker,
            as_of_date=as_of_date,
            db_path=db_path,
            cfg=cfg,
            candidate_context=candidate_context,
            run_id=run_id,
            prior_stage_results=prior_stage_results,
        )
        has_price = price_resolution.status == "RESOLVED"
        conn = _connect(db_path)
        try:
            scorecard_state = _v2_scorecard_provenance_state(
                conn,
                ticker,
                as_of_date=as_of_date,
                issuer_cik=issuer_cik,
                issuer_aliases=aliases,
                price_snapshot=(price_resolution.snapshot if has_price else None),
            )
        finally:
            conn.close()
        scorecard_asof = scorecard_state["validated_asof"]
        missing: list[str] = []
        if identity.get("outcome") != "RESOLVED":
            missing.append("IDENTITY")
        if not isinstance(cap.get("market_cap_mm"), (int, float)):
            missing.append("CAP")
        if facts.get("outcome") != "AVAILABLE":
            missing.append("FACTS")
        if not has_filing:
            missing.append("FILING")
        if not has_price:
            missing.append("PRICE")
        if _scorecard_is_stale(scorecard_asof, as_of_date):
            missing.append("VALUATION")
        return {
            "outcome": "READY_FOR_ASSEMBLY" if not missing else "NEEDS_DATA",
            "reason_code": (None if not missing else "PACKET_INPUTS_INCOMPLETE"),
            "terminal": True,
            "retryable": False,
            "stage_kind": "PACKET_READINESS_CHECK",
            "packet_materialized": False,
            "missing_inputs": missing,
            "fact_coverage": facts,
            "price_resolution": price_resolution.to_dict(),
            "valuation_status": (
                "FRESH"
                if scorecard_asof
                else "PROVENANCE_MISMATCH"
                if scorecard_state["raw_asof"]
                else "MISSING"
            ),
            "scorecard_asof": scorecard_asof,
            "raw_scorecard_asof": scorecard_state["raw_asof"],
            "valuation_provenance_mismatches": scorecard_state["mismatch_reasons"],
        }

    raise ValueError(f"Unknown v2 repair stage: {stage}")


def resolve_data_gaps_for_ticker(
    ticker: str,
    *,
    as_of_date: str,
    db_path: str | Path | None = None,
    cfg: AppConfig | None = None,
    fetch_filing: bool = True,
    refresh_valuation: bool = True,
) -> dict[str, Any]:
    """Deterministic repair pass for one name. Returns the actions taken."""
    cfg = cfg or get_config()
    path = Path(db_path) if db_path is not None else Path(cfg.db_path)
    upper = str(ticker).upper()
    actions: list[str] = []
    errors: list[str] = []

    conn = _connect(path)
    try:
        had_filing = _has_readable_annual_filing(conn, upper)
        scorecard_asof = _latest_scorecard_asof(conn, upper)
    finally:
        conn.close()

    if fetch_filing and not had_filing:
        try:
            if ensure_company_row(upper, db_path=path):
                from app.ingest.filings import ingest_with_policy

                ingest_with_policy(
                    as_of_date=as_of_date,
                    run_id=f"evidence_resolution_{str(as_of_date).replace('-', '')}",
                    tickers=[upper],
                )
                actions.append("FILING_INGEST_ATTEMPTED")
                # A downloaded filing is invisible to packet readers until it
                # parses (VALID_CACHED_FILING_STATUSES is OK/parsed) — finish
                # the job here or the repair changes nothing downstream.
                from app.parse.filing_parser import parse_pending_filings

                parsed = parse_pending_filings(limit=10, tickers=[upper])
                if parsed:
                    actions.append(f"FILINGS_PARSED:{parsed}")
            else:
                errors.append("FILING_INGEST_SKIPPED:NO_CIK")
        except Exception as exc:  # noqa: BLE001 - repair records, never aborts
            errors.append(f"FILING_INGEST_ERROR:{type(exc).__name__}")

    if refresh_valuation and _scorecard_is_stale(scorecard_asof, as_of_date):
        try:
            from app.valuation.valuation_writer import ensure_valuation

            ensure_valuation(
                upper,
                as_of_date,
                require_filed_asof=True,
            )
            actions.append("VALUATION_REFRESHED")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"VALUATION_REFRESH_ERROR:{type(exc).__name__}")

    conn = _connect(path)
    try:
        has_filing_now = _has_readable_annual_filing(conn, upper)
        scorecard_now = _latest_scorecard_asof(conn, upper)
    finally:
        conn.close()

    return {
        "ticker": upper,
        "actions": actions,
        "errors": errors,
        "had_readable_annual_filing": had_filing,
        "has_readable_annual_filing": has_filing_now,
        "scorecard_asof_before": scorecard_asof,
        "scorecard_asof_after": scorecard_now,
    }


def _pre_assembly_data_gap_repair_v1(
    *,
    tickers: list[str],
    as_of_date: str,
    db_path: str | Path | None = None,
    cfg: AppConfig | None = None,
    max_repairs: int | None = None,
    apply_repairs: bool = True,
) -> dict[str, Any]:
    """Legacy count-capped repair hook retained unchanged for v1 runs.

    Only names that actually need repair consume budget. The cap bounds
    pre-LLM wall clock; skipped names are counted, never silently dropped —
    the next run picks them up (idempotent).
    """
    cfg = cfg or get_config()
    path = Path(db_path) if db_path is not None else Path(cfg.db_path)
    cap = (
        max_repairs
        if max_repairs is not None
        else int(getattr(cfg, "data_gap_repair_max_per_run", 25) or 25)
    )

    needs: list[str] = []
    conn = _connect(path)
    try:
        for ticker in tickers:
            upper = str(ticker).upper()
            if _ticker_needs_legacy_repair(conn, upper, as_of_date=as_of_date):
                needs.append(upper)
    finally:
        conn.close()

    repaired: list[dict[str, Any]] = []
    if apply_repairs:
        for upper in needs[: max(0, cap)]:
            repaired.append(
                resolve_data_gaps_for_ticker(upper, as_of_date=as_of_date, db_path=path, cfg=cfg)
            )
        skipped = needs[max(0, cap) :]
    else:
        skipped = list(needs)
    return {
        "status": (
            "APPLIED"
            if repaired
            else "NOTHING_TO_REPAIR"
            if not needs
            else "SKIPPED_PRE_AUTHORIZATION"
            if not apply_repairs
            else "SKIPPED"
        ),
        "examined": len(tickers),
        "needed_repair": len(needs),
        "repaired": repaired,
        "repaired_count": len(repaired),
        "skipped_over_cap": skipped,
        "max_repairs": cap,
        "apply_repairs": bool(apply_repairs),
    }


def _candidate_stage_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    stages = candidate.get("stages") if isinstance(candidate.get("stages"), dict) else {}
    identity_result = (
        (stages.get("IDENTITY") or {}).get("result")
        if isinstance((stages.get("IDENTITY") or {}).get("result"), dict)
        else {}
    )
    packet_result = (
        (stages.get("PACKET") or {}).get("result")
        if isinstance((stages.get("PACKET") or {}).get("result"), dict)
        else {}
    )
    facts_result = (
        (stages.get("FACTS_AVAILABILITY") or {}).get("result")
        if isinstance((stages.get("FACTS_AVAILABILITY") or {}).get("result"), dict)
        else {}
    )
    price_result = (
        (stages.get("PRICE") or {}).get("result")
        if isinstance((stages.get("PRICE") or {}).get("result"), dict)
        else {}
    )
    cap_result = (
        (stages.get("CAP") or {}).get("result")
        if isinstance((stages.get("CAP") or {}).get("result"), dict)
        else {}
    )
    valuation_result = (
        (stages.get("VALUATION") or {}).get("result")
        if isinstance((stages.get("VALUATION") or {}).get("result"), dict)
        else {}
    )
    return {
        "ticker": candidate.get("ticker"),
        "queue_status": candidate.get("queue_status"),
        "last_completed_stage": candidate.get("last_completed_stage"),
        "next_stage": candidate.get("next_stage"),
        "packet_readiness_outcome": packet_result.get("outcome"),
        "packet_missing_inputs": [
            str(item) for item in packet_result.get("missing_inputs") or [] if str(item).strip()
        ],
        "facts_reason_code": facts_result.get("reason_code"),
        "price_reason_code": price_result.get("reason_code"),
        "terminal_cap_search": dict(cap_result.get("terminal_cap_search") or {}),
        "valuation_source_records": [
            dict(record)
            for record in valuation_result.get("valuation_source_records") or []
            if isinstance(record, dict)
        ],
        # This is the immutable input-evidence projection consumed by packet
        # assembly. Keep the complete stage results, including source attempts,
        # rather than collapsing them to an availability boolean.
        "packet_inputs": {
            "facts": dict(facts_result),
            "price": dict(price_result),
        },
        "issuer_identity": {
            key: identity_result.get(key)
            for key in (
                "issuer_cik",
                "issuer_primary_ticker",
                "issuer_listed_tickers",
                "issuer_aliases",
                "security_role",
                "is_secondary_class",
                "is_adr",
                "adr_ratio",
                "share_class_ratio",
                "identity_source",
                "identity_source_url",
                "identity_as_of_date",
                "identity_confidence",
                "ratio_source_url",
            )
            if identity_result.get(key) is not None
        },
        "stage_states": {
            stage: {
                "status": (stages.get(stage) or {}).get("status"),
                "outcome": ((stages.get(stage) or {}).get("result") or {}).get("outcome"),
                "reason_code": ((stages.get(stage) or {}).get("result") or {}).get("reason_code"),
                "terminal": ((stages.get(stage) or {}).get("result") or {}).get("terminal"),
                "retryable": ((stages.get(stage) or {}).get("result") or {}).get("retryable"),
                "error_type": ((stages.get(stage) or {}).get("error") or {}).get("type"),
            }
            for stage in V2_REPAIR_STAGE_SEQUENCE
        },
    }


def _terminal_cap_search_usage_rollup(
    candidates: dict[str, Any],
    candidate_order: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for ticker in candidate_order:
        candidate = candidates.get(ticker)
        stages = candidate.get("stages") if isinstance(candidate, dict) else None
        cap_state = stages.get("CAP") if isinstance(stages, dict) else None
        result = cap_state.get("result") if isinstance(cap_state, dict) else None
        for record in result.get("usage_records") or [] if isinstance(result, dict) else []:
            if not isinstance(record, dict):
                continue
            key = (
                record.get("authorization_run_id"),
                record.get("attempt_number"),
                record.get("ticker") or ticker,
                record.get("call_type"),
                record.get("call_id"),
            )
            if key in seen:
                continue
            seen.add(key)
            records.append({**record, "ticker": str(record.get("ticker") or ticker).upper()})
    return records, _terminal_cap_search_usage_records_rollup(records)


def _terminal_cap_search_usage_records_rollup(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "response_calls": sum(
            1 for record in records if record.get("call_type") == "responses_model"
        ),
        "web_search_calls": sum(
            1 for record in records if record.get("call_type") == "web_search_call"
        ),
        "input_tokens": sum(int(record.get("input_tokens") or 0) for record in records),
        "cached_input_tokens": sum(
            int(record.get("cached_input_tokens") or 0) for record in records
        ),
        "output_tokens": sum(int(record.get("output_tokens") or 0) for record in records),
        "cost_estimate_usd": round(
            sum(float(record.get("cost_estimate_usd") or 0.0) for record in records),
            6,
        ),
    }


_V2_STAGE_RESOLVED_OUTCOMES = {
    "MEMBERSHIP": {"ADMITTED", "OUT_OF_SCOPE"},
    "IDENTITY": {"RESOLVED", "NEEDS_DATA"},
    "CAP": {"RESOLVED", "NEEDS_DATA"},
    "FACTS_AVAILABILITY": {"AVAILABLE", "NEEDS_DATA"},
    "FILINGS": {"AVAILABLE", "NEEDS_DATA"},
    "PARSING": {"READABLE", "NEEDS_DATA"},
    "PRICE": {"AVAILABLE", "NEEDS_DATA"},
    "VALUATION": {"FRESH", "NEEDS_DATA"},
    "PACKET": {"READY_FOR_ASSEMBLY", "NEEDS_DATA"},
}


def _v2_stage_requires_retry(stage: str, result: dict[str, Any]) -> bool:
    """Return whether an observed stage outcome must be retried next attempt."""

    outcome = str(result.get("outcome") or "").strip().upper()
    retryable = result.get("retryable")
    if isinstance(retryable, bool):
        return retryable
    terminal = result.get("terminal")
    if isinstance(terminal, bool):
        return not terminal
    # Tests and extension hooks historically use OK as their generic resolved
    # result. NOT_APPLICABLE is terminal only after an out-of-scope membership
    # decision has explicitly short-circuited the downstream stages.
    if outcome in {"OK", "NOT_APPLICABLE"}:
        return False
    return outcome not in _V2_STAGE_RESOLVED_OUTCOMES.get(stage, set())


def _prepare_v2_retry_frontier(candidate: dict[str, Any]) -> None:
    """Invalidate downstream observations after the first retryable stage."""

    stages = candidate.get("stages")
    if not isinstance(stages, dict):
        return
    first_retry_index: int | None = None
    for index, stage in enumerate(V2_REPAIR_STAGE_SEQUENCE):
        state = stages.get(stage) if isinstance(stages.get(stage), dict) else {}
        if state.get("status") != "COMPLETED":
            first_retry_index = index
            break
    if first_retry_index is None:
        return
    for stage in V2_REPAIR_STAGE_SEQUENCE[first_retry_index + 1 :]:
        state = stages.get(stage) if isinstance(stages.get(stage), dict) else {}
        if state.get("status") == "PENDING":
            continue
        stages[stage] = {
            "status": "PENDING",
            "invalidated_at": _utc_now_iso(),
            "invalidation_reason": "UPSTREAM_STAGE_RETRY",
        }


def _refresh_v2_candidate_frontier(candidate: dict[str, Any]) -> bool:
    """Persist the contiguous completion frontier; return True when terminal."""

    stages = candidate.get("stages")
    if not isinstance(stages, dict):
        candidate["queue_status"] = "NEEDS_RETRY"
        candidate["last_completed_stage"] = None
        candidate["next_stage"] = V2_REPAIR_STAGE_SEQUENCE[0]
        return False
    first_incomplete: int | None = None
    for index, stage in enumerate(V2_REPAIR_STAGE_SEQUENCE):
        state = stages.get(stage) if isinstance(stages.get(stage), dict) else {}
        if state.get("status") != "COMPLETED":
            first_incomplete = index
            break
    if first_incomplete is None:
        candidate["queue_status"] = "COMPLETED"
        candidate["last_completed_stage"] = V2_REPAIR_STAGE_SEQUENCE[-1]
        candidate["current_stage"] = None
        candidate["next_stage"] = None
        return True
    candidate["queue_status"] = "NEEDS_RETRY"
    candidate["last_completed_stage"] = (
        V2_REPAIR_STAGE_SEQUENCE[first_incomplete - 1] if first_incomplete > 0 else None
    )
    candidate["current_stage"] = None
    candidate["next_stage"] = V2_REPAIR_STAGE_SEQUENCE[first_incomplete]
    return False


def _v2_completed_stage_was_source_limited(result: dict[str, Any]) -> bool:
    """Return whether an observe-only result must reopen under repair policy."""

    return bool(
        result.get("source_exhausted") is True
        and str(result.get("repair_policy") or "").upper() in V2_SOURCE_LIMITED_REPAIR_POLICIES
    )


def _invalidate_stale_completed_stages(
    candidate: dict[str, Any],
    *,
    ticker: str,
    as_of_date: str,
    db_path: Path,
    cfg: AppConfig,
    candidate_context: dict[str, Any] | None,
    apply_repairs: bool,
) -> tuple[str, str] | None:
    """Invalidate a completed stage and its dependants when evidence changed."""

    stages = candidate.get("stages")
    if not isinstance(stages, dict):
        return V2_REPAIR_STAGE_SEQUENCE[0], "CHECKPOINT_STAGE_STATE_INVALID"
    stale_index: int | None = None
    invalidation_reason: str | None = None
    for stage_index, stage in enumerate(V2_REPAIR_STAGE_SEQUENCE):
        state = stages.get(stage) if isinstance(stages.get(stage), dict) else {}
        if state.get("status") != "COMPLETED":
            break
        result = state.get("result") if isinstance(state.get("result"), dict) else {}
        if result.get("outcome") == "NOT_APPLICABLE":
            continue
        if apply_repairs and _v2_completed_stage_was_source_limited(result):
            stale_index = stage_index
            invalidation_reason = "REPAIR_POLICY_EXPANDED"
            break
        current = _stage_evidence_fingerprint(
            stage,
            ticker=ticker,
            as_of_date=as_of_date,
            db_path=db_path,
            cfg=cfg,
            candidate_context=candidate_context,
        )
        if state.get("evidence_fingerprint") != current:
            stale_index = stage_index
            invalidation_reason = "STAGE_EVIDENCE_FINGERPRINT_CHANGED"
            break
    if stale_index is None:
        return None

    assert invalidation_reason is not None
    for stage in V2_REPAIR_STAGE_SEQUENCE[stale_index:]:
        stages[stage] = {
            "status": "PENDING",
            "invalidated_at": _utc_now_iso(),
            "invalidation_reason": invalidation_reason,
        }
    previous_stage = V2_REPAIR_STAGE_SEQUENCE[stale_index - 1] if stale_index > 0 else None
    candidate["queue_status"] = "PENDING"
    candidate["last_completed_stage"] = previous_stage
    candidate["next_stage"] = V2_REPAIR_STAGE_SEQUENCE[stale_index]
    candidate["current_stage"] = None
    return V2_REPAIR_STAGE_SEQUENCE[stale_index], invalidation_reason


def _pre_assembly_data_gap_repair_v2(
    *,
    tickers: list[str],
    as_of_date: str,
    db_path: str | Path | None,
    cfg: AppConfig,
    max_repairs: int | None,
    checkpoint_path: str | Path | None,
    checkpoint_scope: str | None,
    candidate_context: dict[str, Any] | None,
    apply_repairs: bool,
    run_id: str | None,
    evidence_revision: str | None,
    candidate_context_revision: str | None,
    terminal_cap_search: Any | None,
    price_provider: Any | None,
    filing_windows_days: dict[str, int] | None,
) -> dict[str, Any]:
    """Run every admitted v2 candidate through a durable staged frontier.

    ``max_repairs`` is intentionally a checkpoint *batch size*.  It never
    truncates the queue and therefore cannot create a terminal
    ``skipped_over_cap`` population.  Failed stages remain retryable while
    independent candidates continue through the same invocation.
    """

    path = Path(db_path) if db_path is not None else Path(cfg.db_path)
    terminal_usage_start = len(getattr(terminal_cap_search, "usage_records", ()) or ())
    normalized = _normalized_tickers(tickers)
    configured_size = (
        max_repairs
        if max_repairs is not None
        else int(getattr(cfg, "data_gap_repair_max_per_run", 25) or 25)
    )
    checkpoint_batch_size = max(1, int(configured_size))
    if isinstance(candidate_context, dict) and bool(
        candidate_context.get("execution_bound_frozen")
    ):
        authorized = _normalized_tickers(candidate_context.get("execution_tickers") or [])
        if normalized != authorized:
            raise V2ExecutionBoundDriftError(
                "V2 repair tickers do not match the frozen execution set."
            )
        expected_fingerprint = (
            str(candidate_context.get("execution_fingerprint") or "").strip().lower()
        )
        if expected_fingerprint and expected_fingerprint != _json_fingerprint(authorized):
            raise V2ExecutionBoundDriftError(
                "V2 repair execution fingerprint drifted before repair."
            )
    attempt_input = _v2_attempt_input(
        run_id=run_id,
        as_of_date=as_of_date,
        checkpoint_scope=checkpoint_scope,
        evidence_revision=evidence_revision,
        candidate_context_revision=candidate_context_revision,
        candidate_context=candidate_context,
    )
    resolved_checkpoint_path = (
        Path(checkpoint_path)
        if checkpoint_path is not None
        else _default_v2_repair_checkpoint_path(
            cfg=cfg,
            tickers=normalized,
            as_of_date=as_of_date,
            checkpoint_scope=checkpoint_scope,
            attempt_input=attempt_input,
        )
    )
    checkpoint, resumed = _initialize_v2_repair_checkpoint(
        path=resolved_checkpoint_path,
        tickers=normalized,
        as_of_date=as_of_date,
        checkpoint_scope=checkpoint_scope,
        checkpoint_batch_size=checkpoint_batch_size,
        attempt_input=attempt_input,
    )
    candidate_order = list(checkpoint["candidate_order"])
    candidates = checkpoint["candidates"]

    needs: list[str] = []
    conn = _connect(path)
    try:
        for ticker in candidate_order:
            if _ticker_needs_v2_repair(
                conn,
                ticker,
                as_of_date=as_of_date,
                db_path=path,
                cfg=cfg,
                candidate_context=candidate_context,
            ):
                needs.append(ticker)
    finally:
        conn.close()

    for batch_start in range(0, len(candidate_order), checkpoint_batch_size):
        batch = candidate_order[batch_start : batch_start + checkpoint_batch_size]
        batch_number = (batch_start // checkpoint_batch_size) + 1
        checkpoint["active_batch"] = batch_number
        checkpoint["active_batch_tickers"] = batch
        _write_v2_repair_checkpoint(resolved_checkpoint_path, checkpoint)
        for ticker in batch:
            candidate = candidates[ticker]
            candidate["batch_number"] = batch_number
            invalidation = _invalidate_stale_completed_stages(
                candidate,
                ticker=ticker,
                as_of_date=as_of_date,
                db_path=path,
                cfg=cfg,
                candidate_context=candidate_context,
                apply_repairs=apply_repairs,
            )
            if invalidation:
                invalidated_stage, invalidation_reason = invalidation
                candidate.setdefault("invalidation_events", []).append(
                    {
                        "stage": invalidated_stage,
                        "reason": invalidation_reason,
                        "at": _utc_now_iso(),
                    }
                )
                _write_v2_repair_checkpoint(resolved_checkpoint_path, checkpoint)
            _prepare_v2_retry_frontier(candidate)
            if candidate.get("queue_status") == "COMPLETED" and all(
                (candidate.get("stages", {}).get(stage) or {}).get("status") == "COMPLETED"
                for stage in V2_REPAIR_STAGE_SEQUENCE
            ):
                continue
            candidate["queue_status"] = "IN_PROGRESS"
            for stage_index, stage in enumerate(V2_REPAIR_STAGE_SEQUENCE):
                stage_state = candidate["stages"].setdefault(stage, {"status": "PENDING"})
                if stage_state.get("status") == "COMPLETED":
                    continue
                candidate["current_stage"] = stage
                candidate["next_stage"] = stage
                stage_state.update(
                    {
                        "status": "IN_PROGRESS",
                        "started_at": _utc_now_iso(),
                        "completed_at": None,
                    }
                )
                stage_state.pop("error", None)
                _write_v2_repair_checkpoint(resolved_checkpoint_path, checkpoint)
                try:
                    result = _execute_v2_repair_stage(
                        stage,
                        ticker=ticker,
                        as_of_date=as_of_date,
                        db_path=path,
                        cfg=cfg,
                        candidate_context=candidate_context,
                        apply_repairs=apply_repairs,
                        run_id=run_id,
                        prior_stage_results={
                            prior_stage: dict(prior_result)
                            for prior_stage in V2_REPAIR_STAGE_SEQUENCE[:stage_index]
                            if isinstance(
                                prior_result := (
                                    (candidate["stages"].get(prior_stage) or {}).get("result")
                                ),
                                dict,
                            )
                        },
                        terminal_cap_search=terminal_cap_search,
                        price_provider=price_provider,
                        filing_windows_days=filing_windows_days,
                    )
                except BaseException as exc:
                    is_process_interruption = not isinstance(exc, Exception)
                    stage_state.update(
                        {
                            "status": ("INTERRUPTED" if is_process_interruption else "FAILED"),
                            "completed_at": _utc_now_iso(),
                            "error": {
                                "type": type(exc).__name__,
                                "message": str(exc),
                            },
                        }
                    )
                    candidate["queue_status"] = "NEEDS_RETRY"
                    candidate["next_stage"] = stage
                    checkpoint["execution_status"] = (
                        "INTERRUPTED" if is_process_interruption else "INCOMPLETE"
                    )
                    _write_v2_repair_checkpoint(resolved_checkpoint_path, checkpoint)
                    if is_process_interruption:
                        raise
                    break
                result_payload = dict(result or {})
                requires_retry = _v2_stage_requires_retry(stage, result_payload)
                stage_state.update(
                    {
                        "status": "NEEDS_RETRY" if requires_retry else "COMPLETED",
                        "completed_at": _utc_now_iso(),
                        "result": result_payload,
                        "evidence_fingerprint": _stage_evidence_fingerprint(
                            stage,
                            ticker=ticker,
                            as_of_date=as_of_date,
                            db_path=path,
                            cfg=cfg,
                            candidate_context=candidate_context,
                        ),
                    }
                )
                if requires_retry:
                    stage_state["retry_reason"] = (
                        f"UNRESOLVED_STAGE_OUTCOME:{str(result_payload.get('outcome') or 'MISSING')}"
                    )
                else:
                    stage_state.pop("retry_reason", None)
                    candidate["last_completed_stage"] = stage
                candidate["next_stage"] = (
                    V2_REPAIR_STAGE_SEQUENCE[stage_index + 1]
                    if stage_index + 1 < len(V2_REPAIR_STAGE_SEQUENCE)
                    else None
                )
                if (
                    stage == "MEMBERSHIP"
                    and str((result or {}).get("outcome") or "").upper() == "OUT_OF_SCOPE"
                ):
                    membership_fingerprint = stage_state["evidence_fingerprint"]
                    for downstream in V2_REPAIR_STAGE_SEQUENCE[stage_index + 1 :]:
                        candidate["stages"][downstream] = {
                            "status": "COMPLETED",
                            "started_at": stage_state["completed_at"],
                            "completed_at": stage_state["completed_at"],
                            "result": {
                                "outcome": "NOT_APPLICABLE",
                                "reason": "CANDIDATE_OUT_OF_SCOPE",
                            },
                            "evidence_fingerprint": membership_fingerprint,
                        }
                    candidate["last_completed_stage"] = V2_REPAIR_STAGE_SEQUENCE[-1]
                    candidate["next_stage"] = None
                    _write_v2_repair_checkpoint(
                        resolved_checkpoint_path,
                        checkpoint,
                    )
                    break
                _write_v2_repair_checkpoint(resolved_checkpoint_path, checkpoint)
            _refresh_v2_candidate_frontier(candidate)
            _write_v2_repair_checkpoint(resolved_checkpoint_path, checkpoint)

    completed = [
        ticker
        for ticker in candidate_order
        if candidates[ticker].get("queue_status") == "COMPLETED"
    ]
    pending = [ticker for ticker in candidate_order if ticker not in set(completed)]
    checkpoint["execution_status"] = "COMPLETED" if not pending else "INCOMPLETE"
    checkpoint["active_batch"] = None
    checkpoint["active_batch_tickers"] = []
    checkpoint["completed_tickers"] = completed
    checkpoint["pending_tickers"] = pending
    _write_v2_repair_checkpoint(resolved_checkpoint_path, checkpoint)

    repaired: list[dict[str, Any]] = []
    for ticker in needs:
        candidate = candidates[ticker]
        actions: list[str] = []
        errors: list[str] = []
        for stage in V2_REPAIR_STAGE_SEQUENCE:
            state = candidate["stages"].get(stage) or {}
            result = state.get("result") if isinstance(state.get("result"), dict) else {}
            actions.extend(str(item) for item in result.get("actions") or [])
            error = state.get("error") if isinstance(state.get("error"), dict) else {}
            if error:
                errors.append(f"{stage}:{error.get('type')}:{error.get('message')}")
        repaired.append(
            {
                "ticker": ticker,
                "actions": list(dict.fromkeys(actions)),
                "errors": errors,
                "queue_status": candidate.get("queue_status"),
                "last_completed_stage": candidate.get("last_completed_stage"),
            }
        )

    terminal_usage_records, terminal_usage_rollup = _terminal_cap_search_usage_rollup(
        candidates,
        candidate_order,
    )
    if terminal_cap_search is not None and hasattr(terminal_cap_search, "usage_records"):
        terminal_usage_records = [
            dict(record)
            for record in list(getattr(terminal_cap_search, "usage_records", ()) or ())[
                terminal_usage_start:
            ]
            if isinstance(record, dict)
        ]
        terminal_usage_rollup = _terminal_cap_search_usage_records_rollup(terminal_usage_records)

    return {
        "status": "COMPLETED" if not pending else "INCOMPLETE",
        "pipeline_version": "v2",
        "attempt_id": checkpoint.get("attempt_id"),
        "run_id": checkpoint.get("run_id"),
        "attempt_started_at": checkpoint.get("attempt_started_at"),
        "resume_fingerprint": checkpoint.get("resume_fingerprint"),
        "input_fingerprint": checkpoint.get("input_fingerprint"),
        "evidence_revision": checkpoint.get("evidence_revision"),
        "candidate_context_revision": checkpoint.get("candidate_context_revision"),
        "diagnostic_path": checkpoint.get("diagnostic_path"),
        "attempt_history": list(checkpoint.get("attempt_history") or []),
        "execution_status": checkpoint["execution_status"],
        "examined": len(candidate_order),
        "needed_repair": len(needs),
        "repaired": repaired,
        "repaired_count": len([row for row in repaired if row["queue_status"] == "COMPLETED"]),
        # Transitional compatibility field: v2 is required to keep it empty.
        "skipped_over_cap": [],
        "max_repairs": configured_size,
        "checkpoint_batch_size": checkpoint_batch_size,
        "batch_count": (
            (len(candidate_order) + checkpoint_batch_size - 1) // checkpoint_batch_size
        ),
        "checkpoint_path": str(resolved_checkpoint_path),
        "resumed_from_checkpoint": resumed,
        "completed_tickers": completed,
        "pending_tickers": pending,
        "terminal_cap_search_usage_records": terminal_usage_records,
        "terminal_cap_search_accounting": {
            "lane_totals": {"terminal_cap_search": terminal_usage_rollup},
            "aggregate": dict(terminal_usage_rollup),
            "aggregate_reconciles": True,
            "ledger_path": (str(getattr(terminal_cap_search, "ledger_path", "") or "") or None),
        },
        "candidate_states": [
            _candidate_stage_summary(candidates[ticker]) for ticker in candidate_order
        ],
    }


def pre_assembly_data_gap_repair(
    *,
    tickers: list[str],
    as_of_date: str,
    db_path: str | Path | None = None,
    cfg: AppConfig | None = None,
    max_repairs: int | None = None,
    pipeline_version: str = "v1",
    checkpoint_path: str | Path | None = None,
    checkpoint_scope: str | None = None,
    candidate_context: dict[str, Any] | None = None,
    apply_repairs: bool = True,
    run_id: str | None = None,
    evidence_revision: str | None = None,
    candidate_context_revision: str | None = None,
    terminal_cap_search: Any | None = None,
    price_provider: Any | None = None,
    filing_windows_days: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Repair fetchable packet inputs before assembly.

    V1 retains the legacy per-run cap and result shape.  V2 processes the
    complete queue in durable checkpoint batches and resumes incomplete
    stages when given the same scope/as-of/ticker set (or an explicit path).
    """

    cfg = cfg or get_config()
    normalized_pipeline = str(pipeline_version or "v1").strip().lower()
    if normalized_pipeline == "v1":
        return _pre_assembly_data_gap_repair_v1(
            tickers=tickers,
            as_of_date=as_of_date,
            db_path=db_path,
            cfg=cfg,
            max_repairs=max_repairs,
            apply_repairs=apply_repairs,
        )
    if normalized_pipeline != "v2":
        raise ValueError("pipeline_version must be v1 or v2")
    resolved_candidate_context = candidate_context if isinstance(candidate_context, dict) else {}
    return _pre_assembly_data_gap_repair_v2(
        tickers=tickers,
        as_of_date=as_of_date,
        db_path=db_path,
        cfg=cfg,
        max_repairs=max_repairs,
        checkpoint_path=checkpoint_path,
        checkpoint_scope=checkpoint_scope,
        candidate_context=resolved_candidate_context,
        apply_repairs=apply_repairs,
        run_id=run_id,
        evidence_revision=evidence_revision,
        candidate_context_revision=candidate_context_revision,
        terminal_cap_search=terminal_cap_search,
        price_provider=price_provider,
        filing_windows_days=filing_windows_days,
    )


def reaudit_held_candidate(
    ticker: str,
    *,
    sector: str,
    as_of_date: str,
) -> dict[str, Any]:
    """Offline single-name re-audit on freshly built packet + scenarios.

    No run-scoped tool evidence exists offline, so MISSING_*_EVIDENCE codes
    can remain — the audit maps a candidate held purely on those to
    DATA_INCOMPLETE (resolve-then-promote), which is the honest offline
    ceiling; PASS/WATCHLIST_ONLY appear when packet-level gaps were the only
    holds. Imports stay lazy: the sector runtime is heavy.
    """
    from app.autonomous.sector_financial_packets import build_sector_company_financial_packets
    from app.autonomous.sector_scenarios import build_expected_return_scenarios_for_packets
    from app.autonomous.sector_runtime import _selection_audit_for_ticker

    upper = str(ticker).upper()
    packets = build_sector_company_financial_packets([upper], sector=sector, as_of_date=as_of_date)
    if not packets:
        return {"ticker": upper, "status": "PACKET_UNAVAILABLE", "hard_blockers": []}
    packet = packets[0]
    scenarios_by_ticker = build_expected_return_scenarios_for_packets(packets)
    scenarios = scenarios_by_ticker.get(upper, [])
    audit = _selection_audit_for_ticker(
        selected_ticker=upper,
        packets_by_ticker={packet.ticker: packet},
        scenarios=scenarios,
        tool_calls=[],
        evidence=[],
        degraded_states=[],
        framework=None,
    )
    return {
        "ticker": upper,
        "status": str(audit.get("status")),
        "hard_blockers": list(audit.get("hard_blockers") or []),
        "confidence_caps": list(audit.get("confidence_caps") or []),
        "best_base_annualized_return": audit.get("best_base_annualized_return"),
        "data_quality_status": packet.data_quality_status,
    }


def _latest_artifact_per_sector(runs_dir: Path) -> dict[str, Path]:
    latest: dict[str, tuple[str, Path]] = {}
    for artifact_path in runs_dir.glob("autonomous_sector_*/autonomous_sector_run.json"):
        try:
            payload = json.loads(artifact_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        sector = str(payload.get("sector") or "")
        created = str(payload.get("created_at") or "")
        if not sector:
            continue
        current = latest.get(sector)
        if current is None or created > current[0]:
            latest[sector] = (created, artifact_path)
    return {sector: path for sector, (_, path) in latest.items()}


def resolve_and_reaudit_pool(
    *,
    as_of_date: str | None = None,
    sectors: list[str] | None = None,
    runs_dir: str | Path | None = None,
    db_path: str | Path | None = None,
    cfg: AppConfig | None = None,
    max_names: int | None = None,
    repair: bool = True,
) -> dict[str, Any]:
    """Repair + re-audit the held rows of the latest sector artifacts.

    Held = relative-ranking rows with audit status BLOCKED or DATA_INCOMPLETE
    whose hard blockers are all fetchable. Produces the promotion report the
    universe program requires (who promotes under the new loop, who stays
    held and why).
    """
    cfg = cfg or get_config()
    asof = str(as_of_date or "").strip() or date.today().isoformat()
    base = Path(runs_dir) if runs_dir is not None else Path("data/outputs/runs/autonomous_sector")
    artifacts = _latest_artifact_per_sector(base)
    if sectors:
        wanted = {str(s).strip() for s in sectors if str(s).strip()}
        artifacts = {s: p for s, p in artifacts.items() if s in wanted}

    held: list[dict[str, Any]] = []
    for sector, path in sorted(artifacts.items()):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for row in payload.get("relative_ranking") or []:
            status = str(row.get("audit_status") or "")
            if status not in {"BLOCKED", "DATA_INCOMPLETE"}:
                continue
            blockers = [str(item) for item in (row.get("hard_blockers") or [])]
            held.append(
                {
                    "sector": sector,
                    "ticker": str(row.get("ticker") or "").upper(),
                    "prior_status": status,
                    "prior_hard_blockers": blockers,
                    "fetchable_only": is_held_on_fetchable_gaps_only(blockers),
                    "buy_candidate": bool(row.get("buy_candidate")),
                    "rank": row.get("rank"),
                }
            )

    eligible = [item for item in held if item["fetchable_only"]]
    if max_names is not None:
        eligible = eligible[: int(max_names)]

    outcomes: list[dict[str, Any]] = []
    transitions: dict[str, int] = {}
    for item in eligible:
        ticker = item["ticker"]
        repair_result = (
            resolve_data_gaps_for_ticker(ticker, as_of_date=asof, db_path=db_path, cfg=cfg)
            if repair
            else {"actions": [], "errors": []}
        )
        audit = reaudit_held_candidate(ticker, sector=item["sector"], as_of_date=asof)
        transition = f"{item['prior_status']}->{audit['status']}"
        transitions[transition] = transitions.get(transition, 0) + 1
        outcomes.append(
            {**item, "repair": repair_result, "reaudit": audit, "transition": transition}
        )

    promoted = [
        o
        for o in outcomes
        if o["prior_status"] == "BLOCKED"
        and o["reaudit"]["status"] in {"PASS", "WATCHLIST_ONLY", "DATA_INCOMPLETE"}
    ] + [
        o
        for o in outcomes
        if o["prior_status"] == "DATA_INCOMPLETE"
        and o["reaudit"]["status"] in {"PASS", "WATCHLIST_ONLY"}
    ]
    return {
        "as_of_date": asof,
        "generated_at": _utc_now_iso(),
        "artifacts_examined": {sector: str(path) for sector, path in sorted(artifacts.items())},
        "held_total": len(held),
        "held_fetchable_only": sum(1 for item in held if item["fetchable_only"]),
        "held_genuine": [
            {
                "sector": item["sector"],
                "ticker": item["ticker"],
                "hard_blockers": item["prior_hard_blockers"],
            }
            for item in held
            if not item["fetchable_only"]
        ],
        "reaudited": len(outcomes),
        "transitions": dict(sorted(transitions.items(), key=lambda kv: -kv[1])),
        "promoted_count": len(promoted),
        "outcomes": outcomes,
    }


def write_evidence_resolution_report(
    report: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
) -> dict[str, str]:
    base = Path(output_dir) if output_dir is not None else Path("data/outputs/universe")
    base.mkdir(parents=True, exist_ok=True)
    stamp = str(report.get("as_of_date") or date.today().isoformat()).replace("-", "")
    json_path = base / f"evidence_resolution_{stamp}.json"
    md_path = base / f"evidence_resolution_{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True))

    lines = [
        f"# Evidence Resolution — {report.get('as_of_date')}",
        "",
        f"Held rows in latest artifacts: {report['held_total']} "
        f"({report['held_fetchable_only']} held on fetchable gaps only)",
        f"Re-audited: {report['reaudited']} — promoted: **{report['promoted_count']}**",
        "",
        "## Transitions",
    ]
    for transition, count in report["transitions"].items():
        lines.append(f"- {transition}: {count}")
    genuine = report.get("held_genuine") or []
    lines += [
        "",
        f"## Held for genuine (non-fetchable) reasons: {len(genuine)}",
        "",
        "These names carry at least one non-fetchable hard blocker "
        "(gate block, solvency, capital-loss class); the loop correctly leaves them held.",
    ]
    md_path.write_text("\n".join(lines) + "\n")
    return {"json": str(json_path), "md": str(md_path)}
