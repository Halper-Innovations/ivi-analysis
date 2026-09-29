from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from app.alpha.llm_runtime import _provider_enabled, get_alpha_llm_provider
from app.autonomous.financial_integrity import (
    FinancialIntegrityScope,
    InvalidFinancialInputError,
    require_unchanged_financial_integrity_scope,
)
from app.autonomous.sector_runtime import (
    _estimate_llm_cost_usd,
    _estimate_tokens_from_text,
    _provider_failure_state,
    _provider_name,
    _synthesize_provider_json_with_meta,
)
from app.autonomous.v1_financial_context import (
    BoundV1FinancialScope,
    bind_v1_financial_scope,
    build_canonical_v1_financial_context,
    financial_input_scenario,
)
from app.config import get_config
from app.llm.providers.retry_guard import (
    llm_cost_budget,
    llm_physical_attempt_guard,
)
from app.research.current_event_context import load_current_event_context
from app.util.html_strip import strip_html
from app.watchlist.contract import WatchlistEntry
from app.watchlist.schema import resolve_db_path
from app.watchlist.store import (
    get_latest,
    list_active,
    record_reevaluation_attempt_artifact,
    record_reevaluation_result,
    watchlist_entry_revision_fingerprint,
)


REEVALUATION_FORMS = ("10-Q", "10-Q/A", "8-K", "8-K/A")
REFRESHABLE_STATUSES = {"ACTIVE", "DEPLOY_READY", "BUY_CONFIRMED", "UNCERTAIN"}
# DATA_INCOMPLETE is a conviction GRADE (orthogonal to price STATUS). Such rows
# are always refreshable so a newly-fetched filing can resolve the data gap and
# promote the grade, regardless of the row's current price status.
REFRESHABLE_GRADES = {"DATA_INCOMPLETE"}
# Resolution codes that a filing-section reference (detect_new_evidence
# source_type == "filing") clears. These are the NO_FILING-class
# data-availability gaps that surface as open_questions on a DATA_INCOMPLETE row.
FILING_RESOLVABLE_CODES = {
    "NO_FILING",
    "FILING_RISK_NO_FILING",
    "NO_READABLE_ANNUAL_FILING",
    "RISK_SECTION_NOT_FOUND",
    "FILING_RISK_SECTION_NOT_FOUND",
}
MAX_PROMPT_EVIDENCE_ITEMS = 6
MAX_EVIDENCE_EXCERPT_CHARS = 900
REEVALUATION_MAX_OUTPUT_TOKENS = 1000
CURRENT_EVENT_WATERMARK_SCHEMA = "watchlist_current_event_watermark_v1"


REEVALUATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "evaluation": {
            "type": "string",
            "enum": ["CONFIRMED", "CONTRADICTED", "UNCERTAIN", "NO_MATERIAL_CHANGE"],
        },
        "summary": {"type": "string"},
        "evidence_references": {"type": "array", "items": {"type": "string"}},
        "thesis_components": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "component": {"type": "string"},
                    "verdict": {
                        "type": "string",
                        "enum": ["still_holds", "weakened", "broken", "strengthened"],
                    },
                    "evidence": {"type": "string"},
                },
                "required": ["component", "verdict", "evidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["evaluation", "summary", "evidence_references", "thesis_components"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class WatchlistEvidenceReference:
    evidence_id: str
    source_type: str
    ticker: str
    title: str
    date: str | None
    source_ref: str
    source_url: str | None = None
    accession: str | None = None
    form_type: str | None = None
    local_path: str | None = None
    excerpt: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WatchlistEvidenceSnapshot:
    evidence: list[WatchlistEvidenceReference]
    current_event_watermark: dict[str, Any] | None


@dataclass(frozen=True)
class WatchlistReevaluationResult:
    run_id: str
    ticker: str
    prior_status: str
    new_status: str
    evaluation: str
    summary: str
    evidence_count: int
    llm_called: bool
    cost_estimate_usd: float = 0.0
    artifact_path: str | None = None
    status_reason: str | None = None
    degraded_states: list[str] = field(default_factory=list)
    evidence_references: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WatchlistRefreshSummary:
    run_id: str
    results: list[WatchlistReevaluationResult]

    @property
    def total_cost_estimate_usd(self) -> float:
        return round(sum(result.cost_estimate_usd for result in self.results), 6)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "total_cost_estimate_usd": self.total_cost_estimate_usd,
            "results": [result.to_dict() for result in self.results],
        }


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _run_id(prefix: str = "watchlist_reevaluation") -> str:
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"{prefix}_{day}_{secrets.token_hex(3)}"


def _parse_dateish(value: str | None) -> date:
    if not value:
        return date.today()
    raw = str(value).strip()
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).date()
    except ValueError:
        return date.fromisoformat(raw[:10])


def _connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    from app.db import connect

    return connect(resolve_db_path(db_path))


def _compact_text(value: str, *, limit: int = MAX_EVIDENCE_EXCERPT_CHARS) -> str:
    compact = " ".join(str(value or "").split())
    return compact[:limit].strip()


def _filing_excerpt(local_path: str | None) -> str | None:
    if not local_path:
        return None
    path = Path(local_path)
    if not path.exists():
        return None
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    return _compact_text(strip_html(raw))


def _detect_filing_evidence(
    ticker: str,
    *,
    since_date: date,
    db_path: str | Path | None = None,
    _conn: sqlite3.Connection | None = None,
) -> list[WatchlistEvidenceReference]:
    placeholders = ", ".join("?" for _form in REEVALUATION_FORMS)
    owns_connection = _conn is None
    conn = _conn or _connect(db_path)
    try:
        rows = conn.execute(
            f"""
            SELECT id, ticker, accession, form_type, filing_date, primary_doc_url, local_path
            FROM filings
            WHERE UPPER(ticker) = ?
              AND UPPER(form_type) IN ({placeholders})
              AND filing_date > ?
            ORDER BY filing_date ASC, id ASC
            """,
            (ticker.upper(), *REEVALUATION_FORMS, since_date.isoformat()),
        ).fetchall()
    except sqlite3.OperationalError as exc:
        if "no such table" not in str(exc).lower():
            raise
        rows = []
    finally:
        if owns_connection:
            conn.close()

    evidence: list[WatchlistEvidenceReference] = []
    for index, row in enumerate(rows, start=1):
        form_type = str(row["form_type"] or "").upper()
        accession = str(row["accession"] or "")
        filing_date = str(row["filing_date"] or "")
        evidence.append(
            WatchlistEvidenceReference(
                evidence_id=f"F{index}",
                source_type="filing",
                ticker=ticker.upper(),
                title=f"{form_type} filed {filing_date}",
                date=filing_date,
                source_ref=accession,
                source_url=str(row["primary_doc_url"] or "") or None,
                accession=accession,
                form_type=form_type,
                local_path=str(row["local_path"] or "") or None,
                excerpt=_filing_excerpt(row["local_path"])
                or "Local filing text unavailable; metadata only.",
            )
        )
    return evidence


def _detect_corporate_event_evidence(
    ticker: str,
    *,
    since_date: date,
    db_path: str | Path | None = None,
    _conn: sqlite3.Connection | None = None,
) -> list[WatchlistEvidenceReference]:
    """Evidence from the events feed (corporate_events), not the filings table.

    The events feed detects 8-Ks straight off the EDGAR submissions API and
    records them in corporate_events/corporate_event_filings — those filings
    are usually NOT ingested into the filings table, so without this source a
    refresh cannot see the evidence behind its own EVENT_PENDING flags.
    """
    from app.events.store import QUEUE_PROTECTION_EVENT_TYPES

    protection_placeholders = ", ".join("?" for _ in QUEUE_PROTECTION_EVENT_TYPES)
    owns_connection = _conn is None
    conn = _conn or _connect(db_path)
    try:
        rows = conn.execute(
            f"""
            SELECT
                e.id AS event_id,
                e.event_type,
                e.status,
                e.detection_date,
                e.detail_json,
                f.accession,
                f.form_type,
                f.filing_date
            FROM corporate_events e
            LEFT JOIN corporate_event_filings f ON f.event_id = e.id
            WHERE UPPER(e.ticker) = ?
              AND (
                COALESCE(f.filing_date, e.detection_date) > ?
                -- An OPEN queue-protection event is undisposed backlog
                -- regardless of the row's last_evaluated_at watermark: the
                -- evaluation that advanced the watermark could not have seen
                -- this filing (it is not in the filings table). Other event
                -- types and DECIDED/EXPIRED events defer to the watermark.
                OR (
                    e.status NOT IN ('DECIDED', 'EXPIRED')
                    AND e.event_type IN ({protection_placeholders})
                )
              )
            ORDER BY COALESCE(f.filing_date, e.detection_date) ASC, e.id ASC, f.id ASC
            """,
            (ticker.upper(), since_date.isoformat(), *QUEUE_PROTECTION_EVENT_TYPES),
        ).fetchall()
    except sqlite3.OperationalError as exc:
        if "no such table" not in str(exc).lower():
            raise
        rows = []
    finally:
        if owns_connection:
            conn.close()

    evidence: list[WatchlistEvidenceReference] = []
    for row in rows:
        event_type = str(row["event_type"] or "").upper()
        form_type = str(row["form_type"] or "").upper() or None
        event_date = str(row["filing_date"] or row["detection_date"] or "")
        accession = str(row["accession"] or "") or None
        if form_type:
            title = f"{form_type} filed {event_date} ({event_type} event)"
        else:
            title = f"{event_type} event detected {event_date}"
        detail_excerpt = _compact_text(str(row["detail_json"] or ""))
        evidence.append(
            WatchlistEvidenceReference(
                evidence_id=f"EV{len(evidence) + 1}",
                source_type="corporate_event",
                ticker=ticker.upper(),
                title=title,
                date=event_date or None,
                source_ref=accession or f"corporate_event:{row['event_id']}",
                accession=accession,
                form_type=form_type,
                excerpt=detail_excerpt
                or "Events-feed detection; filing text not ingested (metadata only).",
            )
        )
    return evidence


def _parse_event_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()
    except ValueError:
        try:
            return date.fromisoformat(str(value)[:10])
        except ValueError:
            return None


def _canonical_event_timestamp(value: str | None) -> str | None:
    if not value:
        return None
    raw = str(value).strip()
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return raw
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def _current_event_identity(document: Any) -> str:
    source_type = str(getattr(document, "source_type", "") or "").strip().lower()
    source_url = str(getattr(document, "source_url", "") or "").strip()
    identity_payload: dict[str, str | None] = {
        "schema": "watchlist_current_event_identity_v1",
        "source_type": source_type,
        "source_url": source_url,
    }
    if not source_url:
        identity_payload["title"] = " ".join(
            str(getattr(document, "title", "") or "").lower().split()
        )
        identity_payload["published_at"] = _canonical_event_timestamp(
            getattr(document, "published_at", None)
        )
    encoded = json.dumps(
        identity_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _current_event_watermark_events(
    watermark: dict[str, Any] | None,
) -> dict[str, dict[str, str | None]] | None:
    if watermark is None:
        return None
    if watermark.get("schema") != CURRENT_EVENT_WATERMARK_SCHEMA:
        raise ValueError("Unsupported current-event watermark schema")
    raw_events = watermark.get("events")
    if not isinstance(raw_events, list):
        raise ValueError("Current-event watermark events must be a list")
    events: dict[str, dict[str, str | None]] = {}
    for item in raw_events:
        if not isinstance(item, dict):
            raise ValueError("Current-event watermark item must be an object")
        event_identity = str(item.get("event_identity") or "").strip()
        if not event_identity:
            raise ValueError("Current-event watermark item is missing event_identity")
        events[event_identity] = {
            "event_identity": event_identity,
            "published_at": (
                str(item["published_at"]) if item.get("published_at") is not None else None
            ),
            "source_ref": (str(item["source_ref"]) if item.get("source_ref") is not None else None),
        }
    return events


def _current_event_watermark(
    *,
    prior_watermark: dict[str, Any] | None,
    documents: list[Any],
    as_of_date: str,
) -> dict[str, Any]:
    prior_events = _current_event_watermark_events(prior_watermark) or {}
    merged_events = dict(prior_events)
    for document in documents:
        event_identity = _current_event_identity(document)
        merged_events[event_identity] = {
            "event_identity": event_identity,
            "published_at": _canonical_event_timestamp(getattr(document, "published_at", None)),
            "source_ref": str(getattr(document, "source_url", "") or "") or None,
        }
    return {
        "schema": CURRENT_EVENT_WATERMARK_SCHEMA,
        "as_of_date": str(as_of_date),
        "events": sorted(
            merged_events.values(),
            key=lambda item: (
                str(item.get("published_at") or ""),
                str(item.get("event_identity") or ""),
            ),
        ),
    }


def _detect_current_event_evidence(
    ticker: str,
    *,
    since_date: date,
    as_of_date: str,
    current_event_watermark: dict[str, Any] | None,
) -> tuple[list[WatchlistEvidenceReference], dict[str, Any]]:
    try:
        context = load_current_event_context(ticker.upper(), as_of_date=as_of_date)
    except Exception:
        return (
            [],
            _current_event_watermark(
                prior_watermark=current_event_watermark,
                documents=[],
                as_of_date=as_of_date,
            ),
        )
    documents = list(context.ordered_documents)
    prior_events = _current_event_watermark_events(current_event_watermark)
    evidence: list[WatchlistEvidenceReference] = []
    dated_documents: list[Any] = []
    for document in documents:
        event_date = _parse_event_date(document.published_at)
        if event_date is None:
            continue
        dated_documents.append(document)
        event_identity = _current_event_identity(document)
        # Legacy rows have no identity watermark. Include the boundary day once
        # (rather than dropping all same-day documents via date truncation),
        # then the persisted identity set makes subsequent default runs exact.
        if prior_events is None and event_date < since_date:
            continue
        if prior_events is not None and event_identity in prior_events:
            continue
        evidence.append(
            WatchlistEvidenceReference(
                evidence_id=f"CE{len(evidence) + 1}",
                source_type="current_event",
                ticker=ticker.upper(),
                title=document.title,
                date=document.published_at,
                source_ref=document.source_url,
                source_url=document.source_url,
                excerpt=_compact_text(document.summary),
            )
        )
    return (
        evidence,
        _current_event_watermark(
            prior_watermark=current_event_watermark,
            documents=dated_documents,
            as_of_date=as_of_date,
        ),
    )


def detect_new_evidence_snapshot(
    ticker: str,
    *,
    since: str | date,
    as_of_date: str | None = None,
    db_path: str | Path | None = None,
    include_current_events: bool = True,
    current_event_watermark: dict[str, Any] | None = None,
    _conn: sqlite3.Connection | None = None,
) -> WatchlistEvidenceSnapshot:
    since_date = since if isinstance(since, date) else _parse_dateish(str(since))
    as_of = as_of_date or date.today().isoformat()
    evidence = _detect_filing_evidence(
        ticker,
        since_date=since_date,
        db_path=db_path,
        _conn=_conn,
    )
    seen_accessions = {item.accession for item in evidence if item.accession}
    for item in _detect_corporate_event_evidence(
        ticker,
        since_date=since_date,
        db_path=db_path,
        _conn=_conn,
    ):
        if item.accession and item.accession in seen_accessions:
            continue
        evidence.append(item)
        if item.accession:
            seen_accessions.add(item.accession)
    publication_watermark = current_event_watermark
    if include_current_events:
        event_evidence, publication_watermark = _detect_current_event_evidence(
            ticker,
            since_date=since_date,
            as_of_date=as_of,
            current_event_watermark=current_event_watermark,
        )
        used_ids = {item.evidence_id for item in evidence}
        for item in event_evidence:
            evidence_id = item.evidence_id
            if evidence_id in used_ids:
                evidence_id = f"CE{len(used_ids) + 1}"
            evidence.append(
                WatchlistEvidenceReference(**{**item.to_dict(), "evidence_id": evidence_id})
            )
            used_ids.add(evidence_id)
    return WatchlistEvidenceSnapshot(
        evidence=evidence,
        current_event_watermark=publication_watermark,
    )


def detect_new_evidence(
    ticker: str,
    *,
    since: str | date,
    as_of_date: str | None = None,
    db_path: str | Path | None = None,
    include_current_events: bool = True,
    current_event_watermark: dict[str, Any] | None = None,
    _conn: sqlite3.Connection | None = None,
) -> list[WatchlistEvidenceReference]:
    return detect_new_evidence_snapshot(
        ticker,
        since=since,
        as_of_date=as_of_date,
        db_path=db_path,
        include_current_events=include_current_events,
        current_event_watermark=current_event_watermark,
        _conn=_conn,
    ).evidence


def reevaluation_evidence_fingerprint(
    evidence: list[WatchlistEvidenceReference],
    *,
    ticker: str,
    since: str,
    as_of_date: str,
    include_current_events: bool,
    current_event_watermark: dict[str, Any] | None = None,
    proposed_current_event_watermark: dict[str, Any] | None = None,
) -> str:
    """Hash the exact evidence snapshot and query boundary used for publication."""

    payload = {
        "schema": "watchlist_reevaluation_evidence_v3",
        "ticker": ticker.strip().upper(),
        "since": str(since),
        "as_of_date": str(as_of_date),
        "include_current_events": bool(include_current_events),
        "prior_current_event_watermark": current_event_watermark,
        "proposed_current_event_watermark": proposed_current_event_watermark,
        "evidence": [item.to_dict() for item in evidence],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _grade_after_resolution(
    entry: WatchlistEntry,
    evidence: list[WatchlistEvidenceReference],
) -> str | None:
    """Return the promoted conviction grade for a resolved DATA_INCOMPLETE row,
    or ``None`` when the grade should be left untouched.

    A legacy DATA_INCOMPLETE row is promoted to the conservative WATCHLIST_ONLY
    floor only when every named resolution code in ``open_questions`` is covered
    by the newly-detected evidence. V2 provenance is immutable at this boundary:
    resolving a NEEDS_DATA gap makes the name eligible for another sector pass,
    but only that pass may change its disposition/decision basis/grade together.
    The full re-grade to ACTIONABLE never happens inline here.
    """
    if str(entry.pipeline_version or "").strip().lower() == "v2":
        return None
    if str(entry.conviction_grade or "").upper() != "DATA_INCOMPLETE":
        return None

    codes = [str(code).strip().upper() for code in entry.open_questions if str(code).strip()]
    if not codes:
        return None

    has_filing_evidence = any(str(item.source_type or "").lower() == "filing" for item in evidence)
    for code in codes:
        if code in FILING_RESOLVABLE_CODES:
            if not has_filing_evidence:
                return None
        else:
            # An unrecognized / non-fetchable open question means we cannot prove
            # the data gap is resolved; leave the grade as DATA_INCOMPLETE.
            return None
    return "WATCHLIST_ONLY"


def _entry_context_payload(entry: WatchlistEntry) -> dict[str, Any]:
    return {
        "ticker": entry.ticker,
        "prior_status": entry.status,
        "conviction_grade": entry.conviction_grade,
        "valuation_anchor_method": entry.valuation_anchor_method,
        "valuation_anchor_value": entry.valuation_anchor_value,
        "buy_price_target": entry.buy_price_target,
        "current_price_at_addition": entry.current_price_at_addition,
        "thesis_text": entry.thesis_text,
        "key_risks": entry.key_risks,
        "falsifiers": entry.falsifiers,
        "open_questions": entry.open_questions,
        "source_run_id": entry.source_run_id,
        "source_sector": entry.source_sector,
        "added_at": entry.added_at,
        "last_evaluated_at": entry.last_evaluated_at,
        "status_reason": entry.status_reason,
    }


def _format_entry_context(entry: WatchlistEntry) -> str:
    return json.dumps(
        _entry_context_payload(entry),
        indent=2,
        sort_keys=True,
    )


def _format_evidence_context(evidence: list[WatchlistEvidenceReference]) -> str:
    items = []
    for item in evidence[:MAX_PROMPT_EVIDENCE_ITEMS]:
        items.append(
            {
                "evidence_id": item.evidence_id,
                "source_type": item.source_type,
                "title": item.title,
                "date": item.date,
                "source_ref": item.source_ref,
                "source_url": item.source_url,
                "form_type": item.form_type,
                "accession": item.accession,
                "excerpt": item.excerpt,
            }
        )
    return json.dumps(items, indent=2, sort_keys=True)


def compose_reevaluation_prompt(
    entry: WatchlistEntry,
    *,
    since: str,
    evidence: list[WatchlistEvidenceReference],
) -> str:
    return f"""You are re-evaluating one persistent IVI watchlist entry.

Goal: decide whether new evidence since {since} confirms, contradicts, weakens, or does not materially change the original watchlist thesis.

Important semantics:
- DEPLOY_READY means price is at/below the stored buy-price target; it does not mean automatic buy.
- Preserve the original status unless new evidence directly contradicts or materially clouds the thesis.
- Use only the supplied evidence. If evidence is insufficient, choose UNCERTAIN rather than over-claiming.
- Cite evidence by evidence_id/source_ref in evidence_references.

Original watchlist entry:
{_format_entry_context(entry)}

New evidence:
{_format_evidence_context(evidence)}

Return the required JSON schema only. The summary must be one sentence and decision-useful.
"""


def _status_for_evaluation(prior_status: str, evaluation: str) -> str:
    normalized = evaluation.upper()
    if normalized == "CONTRADICTED":
        return "CONTRADICTED"
    if normalized == "UNCERTAIN":
        return "UNCERTAIN"
    return prior_status.upper()


def estimate_preflight_cost_usd(provider: Any, *, prompt: str, max_output_tokens: int) -> float:
    provider_name = _provider_name(provider)
    cfg = getattr(provider, "cfg", None)
    if provider_name == "anthropic":
        model = str(getattr(cfg, "anthropic_model", "") or "")
    elif provider_name == "openai":
        model = str(getattr(cfg, "openai_model", "") or "")
    else:
        model = ""
    return _estimate_llm_cost_usd(
        provider_name=provider_name,
        model=model,
        input_tokens=_estimate_tokens_from_text(prompt),
        output_tokens=max_output_tokens,
    )


def _bind_reevaluation_financial_scope(
    *,
    entry: WatchlistEntry,
    prompt: str,
    as_of_date: str,
    db_path: str | Path | None,
) -> BoundV1FinancialScope:
    """Authorize the exact finance-bearing request sent to the provider."""

    canonical_context = build_canonical_v1_financial_context(
        tickers=[entry.ticker],
        as_of_date=as_of_date,
        db_path=db_path,
    )
    packet = canonical_context.packets.get(entry.ticker)
    entry_context = _entry_context_payload(entry)
    numeric_decision_fields = {
        key: value
        for key, value in entry_context.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    financial_inputs = {
        "entry_context": entry_context,
        "numeric_decision_fields": numeric_decision_fields,
        "provider_request": {
            "prompt": prompt,
            "schema": REEVALUATION_SCHEMA,
            "schema_name": "watchlist_reevaluation",
            "max_output_tokens": REEVALUATION_MAX_OUTPUT_TOKENS,
        },
    }
    packet_for_scenario: Any = packet
    if packet_for_scenario is None:
        packet_for_scenario = {
            "ticker": entry.ticker,
            "quote_snapshot_id": None,
            "current_price": None,
            "current_price_unit": None,
            "price_basis": None,
        }
    scenario = financial_input_scenario(
        packet_for_scenario,
        financial_inputs=financial_inputs,
    )
    return bind_v1_financial_scope(
        context=f"watchlist_reevaluation:{entry.ticker}",
        run_as_of_date=canonical_context.as_of_date,
        packets=(packet,) if packet is not None else (),
        scenarios=(scenario,),
    )


def _require_current_reevaluation_authorization(
    *,
    entry: WatchlistEntry,
    since: str,
    as_of_date: str,
    db_path: str | Path | None,
    include_current_events: bool,
    use_current_event_watermark: bool,
    authorized_scope: BoundV1FinancialScope,
) -> tuple[WatchlistEntry, list[WatchlistEvidenceReference], str, str]:
    """Re-read mutable watchlist/evidence state after the provider returns."""

    current_entry = get_latest(entry.ticker, db_path=db_path)
    if current_entry is None:
        # Produce the standard fail-closed financial integrity result rather
        # than allowing a stale response to recreate or mutate a removed row.
        bind_v1_financial_scope(
            context=f"watchlist_reevaluation:{entry.ticker}:missing_current_entry",
            run_as_of_date=as_of_date,
            packets=(),
            scenarios=(),
        )
        raise AssertionError("unreachable")
    current_watermark = (
        current_entry.current_event_watermark if use_current_event_watermark else None
    )
    current_snapshot = detect_new_evidence_snapshot(
        entry.ticker,
        since=since,
        as_of_date=as_of_date,
        db_path=db_path,
        include_current_events=include_current_events,
        current_event_watermark=current_watermark,
    )
    current_evidence = current_snapshot.evidence
    current_prompt = compose_reevaluation_prompt(
        current_entry,
        since=since,
        evidence=current_evidence,
    )
    current_scope = _bind_reevaluation_financial_scope(
        entry=current_entry,
        prompt=current_prompt,
        as_of_date=as_of_date,
        db_path=db_path,
    )
    current_scope.require()
    require_unchanged_financial_integrity_scope(
        FinancialIntegrityScope(
            context=current_scope.context,
            run_as_of_date=current_scope.run_as_of_date,
            packets=current_scope.packets,
            scenarios=current_scope.scenarios,
        ),
        expected_scope_fingerprint=authorized_scope.expected_scope_fingerprint,
    )
    return (
        current_entry,
        current_evidence,
        watchlist_entry_revision_fingerprint(current_entry),
        reevaluation_evidence_fingerprint(
            current_evidence,
            ticker=current_entry.ticker,
            since=since,
            as_of_date=as_of_date,
            include_current_events=include_current_events,
            current_event_watermark=current_watermark,
            proposed_current_event_watermark=current_snapshot.current_event_watermark,
        ),
    )


def _call_reevaluation_llm(
    prompt: str,
    *,
    provider: Any | None = None,
    financial_scope: BoundV1FinancialScope,
) -> tuple[dict[str, Any], dict[str, Any]]:
    provider = provider or get_alpha_llm_provider()
    if not _provider_enabled(provider):
        raise RuntimeError("LLM_PROVIDER_UNAVAILABLE")
    strict_provider_kwargs: dict[str, Any] = {}
    if _provider_name(provider) == "openai":
        # A hard call/cost ceiling cannot authorize the provider's sequential
        # output-expansion requests under one logical reservation.
        strict_provider_kwargs["allow_output_token_retry"] = False
    return _synthesize_provider_json_with_meta(
        provider,
        prompt=prompt,
        schema=REEVALUATION_SCHEMA,
        schema_name="watchlist_reevaluation",
        max_output_tokens=REEVALUATION_MAX_OUTPUT_TOKENS,
        integrity_scope=financial_scope,
        **strict_provider_kwargs,
    )


def _financial_integrity_error_in_chain(
    exc: BaseException,
) -> InvalidFinancialInputError | None:
    """Recover integrity failures wrapped by provider compatibility fallbacks."""

    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, InvalidFinancialInputError):
            return current
        current = current.__cause__ or current.__context__
    return None


def _reevaluation_artifact_payload(
    *,
    run_id: str,
    entry: WatchlistEntry,
    since: str,
    evidence: list[WatchlistEvidenceReference],
    prompt: str | None,
    llm_response: dict[str, Any] | None,
    provider_meta: dict[str, Any] | None,
    result: WatchlistReevaluationResult,
    dry_run: bool,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "ticker": entry.ticker,
        "since": since,
        "dry_run": dry_run,
        "entry": entry.to_dict(),
        "evidence": [item.to_dict() for item in evidence],
        "prompt": prompt,
        "llm_response": llm_response,
        "provider_meta": provider_meta,
        "result": result.to_dict(),
        "created_at": _utc_now_iso(),
    }


def _persist_reevaluation_artifact(
    *,
    run_id: str,
    entry: WatchlistEntry,
    since: str,
    evidence: list[WatchlistEvidenceReference],
    prompt: str | None,
    llm_response: dict[str, Any] | None,
    provider_meta: dict[str, Any] | None,
    result: WatchlistReevaluationResult,
    dry_run: bool,
    publication_artifact: dict[str, Any] | None = None,
) -> str:
    cfg = get_config()
    out_dir = cfg.runs_dir / "watchlist_reevaluation" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "reevaluation.json"
    payload = publication_artifact or _reevaluation_artifact_payload(
        run_id=run_id,
        entry=entry,
        since=since,
        evidence=evidence,
        prompt=prompt,
        llm_response=llm_response,
        provider_meta=provider_meta,
        result=result,
        dry_run=dry_run,
    )
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return str(path)


def _discard_unpublished_reevaluation_artifact(path: str) -> None:
    artifact_path = Path(path)
    try:
        artifact_path.unlink(missing_ok=True)
        artifact_path.parent.rmdir()
    except OSError:
        # The generated artifact is not authoritative without its atomic
        # SQLite publication row. Best-effort cleanup must not hide the CAS
        # or state-publication error that prevented authorization.
        pass


def reevaluate_entry(
    entry: WatchlistEntry,
    *,
    since: str | None = None,
    dry_run: bool = False,
    max_cost_remaining_usd: float | None = None,
    db_path: str | Path | None = None,
    include_current_events: bool = True,
) -> WatchlistReevaluationResult:
    normalized_entry = entry.normalized()
    initial_entry_revision = watchlist_entry_revision_fingerprint(normalized_entry)
    since_value = since or normalized_entry.last_evaluated_at or normalized_entry.added_at
    use_current_event_watermark = since is None
    current_event_watermark = (
        normalized_entry.current_event_watermark if use_current_event_watermark else None
    )
    run_as_of_date = date.today().isoformat()
    run_id = _run_id(f"watchlist_reevaluation_{normalized_entry.ticker.lower()}")
    evidence_snapshot = detect_new_evidence_snapshot(
        normalized_entry.ticker,
        since=since_value,
        as_of_date=run_as_of_date,
        db_path=db_path,
        include_current_events=include_current_events,
        current_event_watermark=current_event_watermark,
    )
    evidence = evidence_snapshot.evidence
    evidence_fingerprint = reevaluation_evidence_fingerprint(
        evidence,
        ticker=normalized_entry.ticker,
        since=str(since_value),
        as_of_date=run_as_of_date,
        include_current_events=include_current_events,
        current_event_watermark=current_event_watermark,
        proposed_current_event_watermark=evidence_snapshot.current_event_watermark,
    )

    if dry_run:
        result = WatchlistReevaluationResult(
            run_id=run_id,
            ticker=normalized_entry.ticker,
            prior_status=normalized_entry.status,
            new_status=normalized_entry.status,
            evaluation="DRY_RUN_NEW_EVIDENCE" if evidence else "DRY_RUN_NO_NEW_EVIDENCE",
            summary=f"Dry run found {len(evidence)} new evidence item(s).",
            evidence_count=len(evidence),
            llm_called=False,
            evidence_references=[item.source_ref for item in evidence],
        )
        artifact_path = _persist_reevaluation_artifact(
            run_id=run_id,
            entry=normalized_entry,
            since=str(since_value),
            evidence=evidence,
            prompt=None,
            llm_response=None,
            provider_meta=None,
            result=result,
            dry_run=True,
        )
        return WatchlistReevaluationResult(**{**result.to_dict(), "artifact_path": artifact_path})

    if not evidence:
        reason = "NO_MATERIAL_CHANGE: no new 10-Q, 8-K, or dated current-event evidence since last refresh."
        result = WatchlistReevaluationResult(
            run_id=run_id,
            ticker=normalized_entry.ticker,
            prior_status=normalized_entry.status,
            new_status=normalized_entry.status,
            evaluation="NO_NEW_EVIDENCE",
            summary="No new evidence was found; watchlist status is unchanged.",
            evidence_count=0,
            llm_called=False,
            status_reason=reason,
        )
        publication_artifact = _reevaluation_artifact_payload(
            run_id=run_id,
            entry=normalized_entry,
            since=str(since_value),
            evidence=evidence,
            prompt=None,
            llm_response=None,
            provider_meta=None,
            result=result,
            dry_run=False,
        )
        artifact_path = _persist_reevaluation_artifact(
            run_id=run_id,
            entry=normalized_entry,
            since=str(since_value),
            evidence=evidence,
            prompt=None,
            llm_response=None,
            provider_meta=None,
            result=result,
            dry_run=False,
            publication_artifact=publication_artifact,
        )
        try:
            record_reevaluation_result(
                normalized_entry.ticker,
                status=normalized_entry.status,
                reason=reason,
                evaluation="NO_NEW_EVIDENCE",
                source_run_id=run_id,
                db_path=db_path,
                expected_entry_revision=initial_entry_revision,
                expected_evidence_fingerprint=evidence_fingerprint,
                evidence_since=str(since_value),
                evidence_as_of_date=run_as_of_date,
                include_current_events=include_current_events,
                use_current_event_watermark=use_current_event_watermark,
                publication_artifact=publication_artifact,
            )
        except BaseException:
            _discard_unpublished_reevaluation_artifact(artifact_path)
            raise
        return WatchlistReevaluationResult(**{**result.to_dict(), "artifact_path": artifact_path})

    prompt = compose_reevaluation_prompt(
        normalized_entry, since=str(since_value), evidence=evidence
    )
    financial_scope = _bind_reevaluation_financial_scope(
        entry=normalized_entry,
        prompt=prompt,
        as_of_date=run_as_of_date,
        db_path=db_path,
    )
    provider = get_alpha_llm_provider()
    if max_cost_remaining_usd is not None:
        preflight_cost = estimate_preflight_cost_usd(
            provider,
            prompt=prompt,
            max_output_tokens=REEVALUATION_MAX_OUTPUT_TOKENS,
        )
        if preflight_cost > max_cost_remaining_usd:
            result = WatchlistReevaluationResult(
                run_id=run_id,
                ticker=normalized_entry.ticker,
                prior_status=normalized_entry.status,
                new_status=normalized_entry.status,
                evaluation="BUDGET_SKIPPED",
                summary=f"Skipped before LLM call: estimated ${preflight_cost:.2f} exceeds remaining budget.",
                evidence_count=len(evidence),
                llm_called=False,
                evidence_references=[item.source_ref for item in evidence],
            )
            publication_artifact = _reevaluation_artifact_payload(
                run_id=run_id,
                entry=normalized_entry,
                since=str(since_value),
                evidence=evidence,
                prompt=prompt,
                llm_response=None,
                provider_meta={"preflight_cost_estimate_usd": preflight_cost},
                result=result,
                dry_run=False,
            )
            artifact_path = _persist_reevaluation_artifact(
                run_id=run_id,
                entry=normalized_entry,
                since=str(since_value),
                evidence=evidence,
                prompt=prompt,
                llm_response=None,
                provider_meta={"preflight_cost_estimate_usd": preflight_cost},
                result=result,
                dry_run=False,
                publication_artifact=publication_artifact,
            )
            try:
                record_reevaluation_attempt_artifact(
                    normalized_entry.ticker,
                    source_run_id=run_id,
                    evaluation="BUDGET_SKIPPED",
                    expected_entry_revision=initial_entry_revision,
                    evidence_fingerprint=evidence_fingerprint,
                    publication_artifact=publication_artifact,
                    db_path=db_path,
                )
            except BaseException:
                _discard_unpublished_reevaluation_artifact(artifact_path)
                raise
            return WatchlistReevaluationResult(
                **{**result.to_dict(), "artifact_path": artifact_path}
            )

    def require_exact_scope(_attempt: dict[str, Any] | None = None) -> None:
        financial_scope.require()

    cost_context = None
    try:
        with llm_cost_budget(
            max_cost_remaining_usd,
            strict_first_call=True,
        ) as cost_context:
            with llm_physical_attempt_guard(require_exact_scope):
                require_exact_scope()
                payload, provider_meta = _call_reevaluation_llm(
                    prompt,
                    provider=provider,
                    financial_scope=financial_scope,
                )
                (
                    publication_entry,
                    publication_evidence,
                    publication_entry_revision,
                    publication_evidence_fingerprint,
                ) = _require_current_reevaluation_authorization(
                    entry=normalized_entry,
                    since=str(since_value),
                    as_of_date=run_as_of_date,
                    db_path=db_path,
                    include_current_events=include_current_events,
                    use_current_event_watermark=use_current_event_watermark,
                    authorized_scope=financial_scope,
                )
    except InvalidFinancialInputError:
        raise
    except Exception as exc:
        integrity_error = _financial_integrity_error_in_chain(exc)
        if integrity_error is not None:
            raise integrity_error from exc
        # Strict mode intentionally suppresses transport retries, so there may
        # be no second physical-attempt guard to observe an in-memory mutation
        # made during the failed provider call. Revalidate the exact paid scope
        # explicitly before publishing even a degraded failure result.
        financial_scope.require()
        (
            publication_entry,
            publication_evidence,
            publication_entry_revision,
            publication_evidence_fingerprint,
        ) = _require_current_reevaluation_authorization(
            entry=normalized_entry,
            since=str(since_value),
            as_of_date=run_as_of_date,
            db_path=db_path,
            include_current_events=include_current_events,
            use_current_event_watermark=use_current_event_watermark,
            authorized_scope=financial_scope,
        )
        failure_state = _provider_failure_state(exc)
        reason = f"REEVALUATION_DEGRADED:{failure_state}: {str(exc)[:240]}"
        cost_summary = cost_context.summary() if cost_context is not None else {}
        failed_cost_usd = float(cost_summary.get("cumulative_cost_usd") or 0.0)
        physical_call_count = int(cost_summary.get("call_count") or 0)
        result = WatchlistReevaluationResult(
            run_id=run_id,
            ticker=publication_entry.ticker,
            prior_status=publication_entry.status,
            new_status=publication_entry.status,
            evaluation="PROVIDER_FAILED",
            summary=reason,
            evidence_count=len(publication_evidence),
            llm_called=physical_call_count > 0,
            cost_estimate_usd=failed_cost_usd,
            status_reason=reason,
            degraded_states=[failure_state],
            evidence_references=[item.source_ref for item in publication_evidence],
        )
        failed_provider_meta = {
            "cost_context": cost_summary,
            "physical_call_count": physical_call_count,
            "failure_state": failure_state,
        }
        publication_artifact = _reevaluation_artifact_payload(
            run_id=run_id,
            entry=publication_entry,
            since=str(since_value),
            evidence=publication_evidence,
            prompt=prompt,
            llm_response=None,
            provider_meta=failed_provider_meta,
            result=result,
            dry_run=False,
        )
        artifact_path = _persist_reevaluation_artifact(
            run_id=run_id,
            entry=publication_entry,
            since=str(since_value),
            evidence=publication_evidence,
            prompt=prompt,
            llm_response=None,
            provider_meta=failed_provider_meta,
            result=result,
            dry_run=False,
            publication_artifact=publication_artifact,
        )
        try:
            record_reevaluation_attempt_artifact(
                publication_entry.ticker,
                source_run_id=run_id,
                evaluation=f"PROVIDER_FAILED:{failure_state}",
                expected_entry_revision=publication_entry_revision,
                evidence_fingerprint=publication_evidence_fingerprint,
                publication_artifact=publication_artifact,
                db_path=db_path,
            )
        except BaseException:
            _discard_unpublished_reevaluation_artifact(artifact_path)
            raise
        return WatchlistReevaluationResult(**{**result.to_dict(), "artifact_path": artifact_path})

    evaluation = str(payload.get("evaluation") or "UNCERTAIN").upper()
    summary = str(payload.get("summary") or "Re-evaluation completed.").strip()
    cost_summary = cost_context.summary() if cost_context is not None else {}
    accounted_cost_usd = float(cost_summary.get("cumulative_cost_usd") or 0.0)
    if int(cost_summary.get("call_count") or 0) == 0:
        accounted_cost_usd = float(provider_meta.get("cost_estimate_usd") or 0.0)
    provider_meta = {
        **provider_meta,
        "cost_context": cost_summary,
        "physical_call_count": int(cost_summary.get("call_count") or 0),
    }
    new_status = _status_for_evaluation(publication_entry.status, evaluation)
    # Deterministic grade promotion: a resolved DATA_INCOMPLETE row whose named
    # data gaps are covered by the new evidence is promoted to the WATCHLIST_ONLY
    # floor (never inline-jumped to ACTIONABLE). This is independent of the LLM's
    # status evaluation and never disturbs the price status ladder.
    resolved_grade = _grade_after_resolution(publication_entry, publication_evidence)
    result = WatchlistReevaluationResult(
        run_id=run_id,
        ticker=publication_entry.ticker,
        prior_status=publication_entry.status,
        new_status=new_status,
        evaluation=evaluation,
        summary=summary,
        evidence_count=len(publication_evidence),
        llm_called=True,
        cost_estimate_usd=accounted_cost_usd,
        status_reason=summary,
        evidence_references=[str(item) for item in payload.get("evidence_references", [])],
    )
    publication_artifact = _reevaluation_artifact_payload(
        run_id=run_id,
        entry=publication_entry,
        since=str(since_value),
        evidence=publication_evidence,
        prompt=prompt,
        llm_response=payload,
        provider_meta=provider_meta,
        result=result,
        dry_run=False,
    )
    artifact_path = _persist_reevaluation_artifact(
        run_id=run_id,
        entry=publication_entry,
        since=str(since_value),
        evidence=publication_evidence,
        prompt=prompt,
        llm_response=payload,
        provider_meta=provider_meta,
        result=result,
        dry_run=False,
        publication_artifact=publication_artifact,
    )
    try:
        record_reevaluation_result(
            publication_entry.ticker,
            status=new_status,
            reason=summary,
            evaluation=evaluation,
            source_run_id=run_id,
            conviction_grade=resolved_grade,
            expected_entry_revision=publication_entry_revision,
            expected_evidence_fingerprint=publication_evidence_fingerprint,
            evidence_since=str(since_value),
            evidence_as_of_date=run_as_of_date,
            include_current_events=include_current_events,
            use_current_event_watermark=use_current_event_watermark,
            publication_artifact=publication_artifact,
            db_path=db_path,
        )
    except BaseException:
        _discard_unpublished_reevaluation_artifact(artifact_path)
        raise
    return WatchlistReevaluationResult(**{**result.to_dict(), "artifact_path": artifact_path})


def refresh_watchlist(
    *,
    ticker: str | None = None,
    since: str | None = None,
    dry_run: bool = False,
    max_cost_usd: float | None = None,
    db_path: str | Path | None = None,
    include_current_events: bool = True,
) -> WatchlistRefreshSummary:
    run_id = _run_id("watchlist_refresh")
    if ticker:
        entry = get_latest(ticker, db_path=db_path)
        entries = [entry] if entry is not None else []
    else:
        entries = [
            entry
            for entry in list_active(db_path=db_path)
            if entry.status.upper() in REFRESHABLE_STATUSES
            or str(entry.conviction_grade or "").upper() in REFRESHABLE_GRADES
        ]

    results: list[WatchlistReevaluationResult] = []
    spent = 0.0
    for entry in entries:
        remaining = None if max_cost_usd is None else max(0.0, float(max_cost_usd) - spent)
        result = reevaluate_entry(
            entry,
            since=since,
            dry_run=dry_run,
            max_cost_remaining_usd=remaining,
            db_path=db_path,
            include_current_events=include_current_events,
        )
        spent += result.cost_estimate_usd
        results.append(result)
    return WatchlistRefreshSummary(run_id=run_id, results=results)
