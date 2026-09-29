from __future__ import annotations

import hashlib
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from sqlite3 import Connection, Row
from typing import Sequence

from app.config import AppConfig, get_config
from app.db import get_db
from app.dossier.filing_cache import find_cached_filing, warm_cached_filing_to_raw
from app.ingest.sec_client import FilingStub, SecClient
from app.parse.document_store import filing_local_path
from app.util.financial_data_access import (
    ANNUAL_CACHED_FILING_FORM_TYPES,
    MATERIAL_EVENT_CACHED_FILING_FORM_TYPES,
    QUARTERLY_CACHED_FILING_FORM_TYPES,
    VALID_CACHED_FILING_STATUSES,
    FilingIssuerScope,
    filing_rows,
    issuer_filing_rows,
)


V2_RECOVERABLE_CACHED_FILING_STATUSES = VALID_CACHED_FILING_STATUSES + (
    "downloaded",
    "download_error",
)
RECOVERABLE_CACHED_FILING_STATUSES = VALID_CACHED_FILING_STATUSES + ("download_error",)


@dataclass(frozen=True)
class FilingDocument:
    ticker: str
    cik: str
    accession: str
    form_type: str
    filing_date: str
    period_end: str | None
    role: str
    local_path: str | None
    primary_doc_url: str | None
    html: str
    materialized_from: str | None = None
    content_revision: str | None = None


@dataclass(frozen=True)
class FilingContext:
    documents: list[FilingDocument] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    recent_filing_status: str = "UNKNOWN"
    recovered_document_count: int = 0
    requested_ticker: str | None = None
    issuer_cik: str | None = None
    resolved_aliases: tuple[str, ...] = ()
    identity_sources: tuple[str, ...] = ()

    @property
    def latest_document(self) -> FilingDocument | None:
        ordered = _sort_documents(self.documents)
        return ordered[0] if ordered else None

    @property
    def ordered_documents(self) -> list[FilingDocument]:
        return _sort_documents(self.documents)


_ROW_COLUMNS: tuple[str, ...] = (
    "id",
    "ticker",
    "cik",
    "accession",
    "form_type",
    "filing_date",
    "period_end",
    "primary_doc_url",
    "local_path",
    "hash",
    "updated_at",
)


def _default_role_for_form(form_type: str | None) -> str:
    form = str(form_type or "").upper()
    if form in MATERIAL_EVENT_CACHED_FILING_FORM_TYPES:
        return "material_event"
    if form in QUARTERLY_CACHED_FILING_FORM_TYPES:
        return "quarterly"
    if form in ANNUAL_CACHED_FILING_FORM_TYPES:
        return "annual"
    return "filing"


def _sort_documents(documents: Sequence[FilingDocument]) -> list[FilingDocument]:
    return sorted(
        list(documents),
        key=lambda doc: (
            str(doc.filing_date or ""),
            str(doc.form_type or ""),
            str(doc.accession or ""),
        ),
        reverse=True,
    )


def _filing_stub_from_row(row: Row) -> FilingStub | None:
    cik = str(row["cik"] or "").strip()
    accession = str(row["accession"] or "").strip()
    primary_doc_url = str(row["primary_doc_url"] or "").strip()
    filing_date = str(row["filing_date"] or "").strip()
    if not cik or not accession or not primary_doc_url or not filing_date:
        return None
    try:
        filed = date.fromisoformat(filing_date)
    except ValueError:
        return None
    primary_document = SecClient.filename_from_url(primary_doc_url)
    accession_nodash = accession.replace("-", "")
    return FilingStub(
        cik=cik,
        accession=accession,
        accession_nodash=accession_nodash,
        form_type=str(row["form_type"] or ""),
        filing_date=filed,
        period_end=row["period_end"],
        primary_document=primary_document,
        primary_doc_url=primary_doc_url,
        filing_index_url=(
            f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_nodash}/index.json"
        ),
    )


def _download_primary_document_to_raw(filing: FilingStub, target_path: Path) -> bool:
    try:
        payload = SecClient().download_bytes(filing.primary_doc_url, use_cache=True)
    except Exception:
        return False
    if not payload or not payload.strip():
        return False
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(payload)
    return True


def _path_within_allowed_roots(
    path: Path,
    allowed_roots: Sequence[Path],
) -> bool:
    if not allowed_roots:
        return True
    try:
        resolved_path = path.expanduser().resolve(strict=False)
    except OSError:
        return False
    for root in allowed_roots:
        try:
            resolved_path.relative_to(root.expanduser().resolve(strict=False))
            return True
        except (OSError, ValueError):
            continue
    return False


def _materialize_local_path(
    row: Row,
    *,
    allow_download: bool = True,
    cfg: AppConfig,
    allowed_roots: Sequence[Path] = (),
) -> tuple[Path | None, str | None, str | None, bytes | None]:
    rejection_reason: str | None = None
    stored_hash = str(row["hash"] or "").strip().lower() or None

    def eligible_bytes(path: Path) -> bytes | None:
        nonlocal rejection_reason
        if not _path_within_allowed_roots(path, allowed_roots):
            rejection_reason = rejection_reason or "path_outside_allowed_roots"
            return None
        try:
            payload = path.read_bytes()
        except OSError:
            return None
        if not payload:
            return None
        if stored_hash and hashlib.sha256(payload).hexdigest() != stored_hash:
            rejection_reason = "hash_mismatch"
            return None
        return payload

    local_path = str(row["local_path"] or "").strip()
    if local_path:
        path = Path(local_path)
        payload = eligible_bytes(path)
        if payload is not None:
            return path, "local_path", None, payload

    filing = _filing_stub_from_row(row)
    if filing is None:
        return None, None, rejection_reason, None
    target_path = filing_local_path(
        filing.cik,
        filing.accession,
        filing.primary_document,
        cfg=cfg,
        create_parent=False,
    )
    payload = eligible_bytes(target_path)
    if payload is not None:
        return target_path, "raw_cache", None, payload

    cached_path = find_cached_filing(filing=filing, cfg=cfg)
    cached_payload = eligible_bytes(cached_path) if cached_path is not None else None
    if cached_payload is not None:
        if not allow_download:
            return cached_path, "filing_cache", None, cached_payload
    elif cached_path is not None and rejection_reason == "hash_mismatch":
        return None, None, rejection_reason, None

    if not allow_download:
        return None, None, rejection_reason, None
    if not _path_within_allowed_roots(target_path, allowed_roots):
        return None, None, rejection_reason or "path_outside_allowed_roots", None
    try:
        warmed = bool(cached_payload) and warm_cached_filing_to_raw(
            filing=filing, target_path=target_path, cfg=cfg
        )
    except Exception:
        warmed = False
    payload = eligible_bytes(target_path) if warmed else None
    if payload is not None:
        return target_path, "cached_sec_primary_document", None, payload
    if _download_primary_document_to_raw(filing, target_path):
        payload = eligible_bytes(target_path)
        if payload is not None:
            return target_path, "sec_primary_document_fetch", None, payload
    return None, None, rejection_reason, None


def _document_from_row(
    row: Row,
    *,
    role: str,
    allow_download: bool = True,
    cfg: AppConfig,
    allowed_roots: Sequence[Path] = (),
) -> tuple[FilingDocument | None, str | None, str | None]:
    path, materialized_from, rejection_reason, payload = _materialize_local_path(
        row,
        allow_download=allow_download,
        cfg=cfg,
        allowed_roots=allowed_roots,
    )
    if path is None or payload is None:
        warning_kind = rejection_reason or "unreadable"
        return None, f"{role}_filing_{warning_kind}:{row['accession']}", None

    content_revision = hashlib.sha256(payload).hexdigest()
    html = payload.decode("utf-8", errors="ignore")
    if not html.strip():
        return None, f"{role}_filing_unreadable:{row['accession']}", None

    return (
        FilingDocument(
            ticker=str(row["ticker"] or "").upper(),
            cik=str(row["cik"] or ""),
            accession=str(row["accession"] or ""),
            form_type=str(row["form_type"] or ""),
            filing_date=str(row["filing_date"] or ""),
            period_end=row["period_end"],
            role=role,
            local_path=str(path),
            primary_doc_url=str(row["primary_doc_url"] or "") or None,
            html=html,
            materialized_from=materialized_from,
            content_revision=content_revision,
        ),
        None,
        materialized_from,
    )


def build_inline_filing_context(
    filing_html: str | None,
    *,
    ticker: str,
    form_type: str | None = None,
    filing_date: str | None = None,
    accession: str | None = None,
    role: str | None = None,
) -> FilingContext:
    if not filing_html:
        return FilingContext()
    return FilingContext(
        documents=[
            FilingDocument(
                ticker=str(ticker or "").upper(),
                cik="",
                accession=str(accession or ""),
                form_type=str(form_type or ""),
                filing_date=str(filing_date or ""),
                period_end=None,
                role=role or _default_role_for_form(form_type),
                local_path=None,
                primary_doc_url=None,
                html=filing_html,
            )
        ]
    )


def load_research_filing_context(
    ticker: str,
    *,
    as_of_date: str,
    issuer_cik: str | None = None,
    aliases: Sequence[str] = (),
    issuer_aware: bool = False,
    connection: Connection | None = None,
    annual_filing_limit: int = 1,
    quarters: int = 0,
    include_material_events: bool = True,
    material_event_window_days: int = 365,
    material_event_limit: int = 24,
    allow_annual_download: bool = True,
    allow_network_materialization: bool = True,
    cfg: AppConfig | None = None,
    allowed_filing_roots: Sequence[str | Path] = (),
) -> FilingContext:
    upper = str(ticker or "").upper()
    resolved_cfg = cfg or get_config()
    resolved_allowed_roots = tuple(Path(root) for root in allowed_filing_roots)
    warnings: list[str] = []
    documents: list[FilingDocument] = []
    material_event_rows: list[Row] = []
    recovered_document_count = 0

    try:
        anchor_date = date.fromisoformat(as_of_date)
    except ValueError:
        anchor_date = None
    material_event_cutoff = (
        (anchor_date - timedelta(days=max(0, int(material_event_window_days)))).isoformat()
        if anchor_date is not None
        else None
    )

    db_context = nullcontext(connection) if connection is not None else get_db(resolved_cfg)
    with db_context as conn:
        if issuer_aware:
            scope, annual_rows = issuer_filing_rows(
                conn,
                upper,
                columns=_ROW_COLUMNS,
                issuer_cik=issuer_cik,
                aliases=aliases,
                form_types=ANNUAL_CACHED_FILING_FORM_TYPES,
                statuses=V2_RECOVERABLE_CACHED_FILING_STATUSES,
                as_of_date=as_of_date,
                require_local_path=False,
                order_by="COALESCE(filing_date, '1900-01-01') DESC, id DESC",
            )
            _quarter_scope, quarter_rows = (
                issuer_filing_rows(
                    conn,
                    upper,
                    columns=_ROW_COLUMNS,
                    issuer_cik=scope.issuer_cik,
                    aliases=scope.aliases,
                    form_types=QUARTERLY_CACHED_FILING_FORM_TYPES,
                    statuses=V2_RECOVERABLE_CACHED_FILING_STATUSES,
                    as_of_date=as_of_date,
                    require_local_path=False,
                    limit=max(0, int(quarters)),
                    order_by="COALESCE(filing_date, '1900-01-01') DESC, id DESC",
                )
                if quarters > 0
                else (scope, [])
            )
            if include_material_events:
                _event_scope, material_event_rows = issuer_filing_rows(
                    conn,
                    upper,
                    columns=_ROW_COLUMNS,
                    issuer_cik=scope.issuer_cik,
                    aliases=scope.aliases,
                    form_types=MATERIAL_EVENT_CACHED_FILING_FORM_TYPES,
                    statuses=V2_RECOVERABLE_CACHED_FILING_STATUSES,
                    as_of_date=as_of_date,
                    require_local_path=False,
                    order_by="COALESCE(filing_date, '1900-01-01') DESC, id DESC",
                )
        else:
            # V1 remains ticker-exact. Issuer aliases, local-first recovery,
            # and amendment fallback are v2 behavior and must not change an
            # existing scheduled cadence while the rollout allowlist is empty.
            scope = FilingIssuerScope(
                requested_ticker=upper,
                issuer_cik=None,
                aliases=(upper,),
                sources=(),
            )
            annual_rows = filing_rows(
                conn,
                upper,
                columns=_ROW_COLUMNS,
                form_types=ANNUAL_CACHED_FILING_FORM_TYPES,
                statuses=RECOVERABLE_CACHED_FILING_STATUSES,
                as_of_date=as_of_date,
                require_local_path=False,
                order_by="COALESCE(filing_date, '1900-01-01') DESC, id DESC",
            )
            quarter_rows = (
                filing_rows(
                    conn,
                    upper,
                    columns=_ROW_COLUMNS,
                    form_types=QUARTERLY_CACHED_FILING_FORM_TYPES,
                    statuses=RECOVERABLE_CACHED_FILING_STATUSES,
                    as_of_date=as_of_date,
                    require_local_path=False,
                    limit=max(0, int(quarters)),
                    order_by="COALESCE(filing_date, '1900-01-01') DESC, id DESC",
                )
                if quarters > 0
                else []
            )
            if include_material_events:
                material_event_rows = filing_rows(
                    conn,
                    upper,
                    columns=_ROW_COLUMNS,
                    form_types=MATERIAL_EVENT_CACHED_FILING_FORM_TYPES,
                    statuses=RECOVERABLE_CACHED_FILING_STATUSES,
                    as_of_date=as_of_date,
                    require_local_path=False,
                    order_by="COALESCE(filing_date, '1900-01-01') DESC, id DESC",
                )

    if not annual_rows:
        warnings.append("annual_filing_missing")
    elif not issuer_aware:
        # Preserve the legacy newest-to-oldest, fetch-as-you-go behavior and
        # stop at the first readable exact-ticker annual filing.
        for row in annual_rows:
            document, warning, materialized_from = _document_from_row(
                row,
                role="annual",
                allow_download=(allow_annual_download and allow_network_materialization),
                cfg=resolved_cfg,
                allowed_roots=resolved_allowed_roots,
            )
            if document is not None:
                documents.append(document)
                if materialized_from in {
                    "filing_cache",
                    "cached_sec_primary_document",
                    "sec_primary_document_fetch",
                }:
                    recovered_document_count += 1
                break
            if warning:
                warnings.append(warning)
    else:
        annual_limit = max(1, int(annual_filing_limit))
        readable_annuals = 0
        materialization_rows: list[Row] = []
        # Tier 1 is strictly local: explicit local_path, deterministic raw
        # cache, or the dossier filing cache. Preserve filing-date ordering
        # from issuer_filing_rows within this tier, and do not let a newer
        # missing-path row trigger SEC retries before an older cached alias or
        # foreign annual can satisfy the request.
        for row in annual_rows:
            try:
                document, _warning, materialized_from = _document_from_row(
                    row,
                    role="annual",
                    allow_download=False,
                    cfg=resolved_cfg,
                    allowed_roots=resolved_allowed_roots,
                )
            except Exception:
                document = None
                materialized_from = None
            if document is not None:
                documents.append(document)
                readable_annuals += 1
                if materialized_from in {"filing_cache", "cached_sec_primary_document"}:
                    recovered_document_count += 1
                if readable_annuals >= annual_limit:
                    break
            else:
                materialization_rows.append(row)

        # Tier 2 may fetch. It is reached only when all existing readable
        # cached candidates have been considered and the caller's annual
        # limit is still unsatisfied. A failed row is diagnostic, not fatal;
        # later candidates (including amendment/prior-full pairs) continue.
        if readable_annuals >= annual_limit:
            warnings.extend(
                f"annual_filing_unreadable:{row['accession']}" for row in materialization_rows
            )
        if (
            readable_annuals < annual_limit
            and allow_annual_download
            and allow_network_materialization
        ):
            for row in materialization_rows:
                try:
                    document, warning, materialized_from = _document_from_row(
                        row,
                        role="annual",
                        allow_download=True,
                        cfg=resolved_cfg,
                        allowed_roots=resolved_allowed_roots,
                    )
                except Exception as exc:
                    document = None
                    materialized_from = None
                    warning = (
                        "annual_filing_materialization_failed:"
                        f"{row['accession']}:{type(exc).__name__}"
                    )
                if document is not None:
                    documents.append(document)
                    readable_annuals += 1
                    if materialized_from in {
                        "filing_cache",
                        "cached_sec_primary_document",
                        "sec_primary_document_fetch",
                    }:
                        recovered_document_count += 1
                    if readable_annuals >= annual_limit:
                        break
                if warning:
                    warnings.append(warning)
        elif readable_annuals < annual_limit:
            warnings.extend(
                f"annual_filing_unreadable:{row['accession']}" for row in materialization_rows
            )

    readable_quarters = 0
    for row in quarter_rows:
        document, warning, materialized_from = _document_from_row(
            row,
            role="quarterly",
            allow_download=allow_network_materialization,
            cfg=resolved_cfg,
            allowed_roots=resolved_allowed_roots,
        )
        if document is not None:
            documents.append(document)
            readable_quarters += 1
            if materialized_from in {
                "filing_cache",
                "cached_sec_primary_document",
                "sec_primary_document_fetch",
            }:
                recovered_document_count += 1
        if warning:
            warnings.append(warning)

    if quarters > 0 and not quarter_rows:
        warnings.append("no_recent_quarterly_filings_cached")
    if quarters > 0 and readable_quarters < quarters:
        warnings.append(f"quarterly_filings_partial:{readable_quarters}/{quarters}")

    recent_candidate_count = len(quarter_rows)
    recent_readable_count = readable_quarters
    if material_event_rows:
        filtered_material_event_rows = [
            row
            for row in material_event_rows
            if material_event_cutoff is None
            or str(row["filing_date"] or "") >= material_event_cutoff
        ]
        available_material_events = len(filtered_material_event_rows)
        capped_material_event_rows = filtered_material_event_rows[
            : max(0, int(material_event_limit))
        ]
        if available_material_events > len(capped_material_event_rows):
            warnings.append(
                f"material_event_filings_truncated:{len(capped_material_event_rows)}/{available_material_events}"
            )
        recent_candidate_count += len(capped_material_event_rows)
        for row in capped_material_event_rows:
            document, warning, materialized_from = _document_from_row(
                row,
                role="material_event",
                allow_download=allow_network_materialization,
                cfg=resolved_cfg,
                allowed_roots=resolved_allowed_roots,
            )
            if document is not None:
                documents.append(document)
                recent_readable_count += 1
                if materialized_from in {
                    "filing_cache",
                    "cached_sec_primary_document",
                    "sec_primary_document_fetch",
                }:
                    recovered_document_count += 1
            if warning:
                warnings.append(warning)

    if recent_candidate_count == 0:
        recent_status = "NO_RECENT_FILINGS_CACHED"
    elif recent_readable_count == 0:
        recent_status = "RECENT_FILINGS_UNREADABLE"
    else:
        recent_status = "RECENT_FILING_CONTEXT_AVAILABLE"

    return FilingContext(
        documents=_sort_documents(documents),
        warnings=list(dict.fromkeys(warnings)),
        recent_filing_status=recent_status,
        recovered_document_count=recovered_document_count,
        requested_ticker=upper,
        issuer_cik=scope.issuer_cik,
        resolved_aliases=scope.aliases,
        identity_sources=scope.sources,
    )
