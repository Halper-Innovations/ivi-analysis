from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from app.config import get_config
from app.db import get_db, utc_now_iso
from app.dossier.filing_cache import warm_cached_filing_to_raw
from app.ingest.filings import _download_filing, _upsert_filing
from app.ingest.sec_client import FilingStub, SecClient
from app.logging import get_logger
from app.parse.document_store import file_hash, filing_local_path
from app.parse.filing_parser import parse_filing_by_id
from app.util.db_lock import DB_WRITE_LOCK
from app.valuation.facts import resolve_cik_for_ticker


logger = get_logger(__name__)

ANNUAL_FORM_TYPES = ("10-K", "20-F", "40-F")
ANNUAL_FORM_TYPES_WITH_AMENDMENTS = ("10-K", "10-K/A", "20-F", "20-F/A", "40-F")
NO_ANNUAL_FILING_ERROR = "No annual filing (10-K/20-F/40-F) available"

# Foreign private issuers report interim periods on 6-K, which is a
# press-release-grade catch-all with unbounded per-year counts — the
# quarterly docket stays domestic 10-Q only.
QUARTERLY_FORM_TYPES = ("10-Q",)
QUARTERLY_FORM_TYPES_WITH_AMENDMENTS = ("10-Q", "10-Q/A")


@dataclass(frozen=True)
class DossierFiling:
    ticker: str
    cik: str
    accession: str
    form_type: str
    filing_date: str
    period_end: str | None
    primary_doc_url: str
    local_path: str | None
    filing_id: int


@dataclass(frozen=True)
class DossierStage1Filing:
    ticker: str
    cik: str
    filing: FilingStub
    local_path: str | None


def _subtract_years(value: date, years: int) -> date:
    target_year = value.year - max(1, int(years))
    try:
        return value.replace(year=target_year)
    except ValueError:
        # Handle leap-year boundary deterministically.
        return value.replace(year=target_year, day=28)


def _company_row(ticker: str) -> dict[str, Any] | None:
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT ticker, cik, name
            FROM companies
            WHERE ticker = ?
            LIMIT 1
            """,
            (ticker.upper(),),
        ).fetchone()
    return dict(row) if row else None


def _period_year(period_end: str | None, filing_date: date) -> int:
    if period_end:
        try:
            return date.fromisoformat(period_end).year
        except Exception:
            pass
    return filing_date.year


def _annual_forms(
    *,
    include_amendments: bool,
    include_foreign: bool = True,
) -> list[str]:
    forms: list[str] = ["10-K"]
    if include_amendments:
        forms.append("10-K/A")
    if include_foreign:
        forms.append("20-F")
        if include_amendments:
            forms.append("20-F/A")
        forms.append("40-F")
    return forms


def _quarterly_forms(*, include_amendments: bool) -> list[str]:
    if include_amendments:
        return list(QUARTERLY_FORM_TYPES_WITH_AMENDMENTS)
    return list(QUARTERLY_FORM_TYPES)


def _default_min_annual_filings(*, years_back: int) -> int:
    years = max(1, int(years_back))
    if years >= 10:
        return 3
    if years >= 6:
        return 2
    return 1


def _min_annual_filings_required(*, years_back: int, min_annual_filings: int | None) -> int:
    if min_annual_filings is None:
        return _default_min_annual_filings(years_back=years_back)
    return max(1, int(min_annual_filings))


def _build_preflight_payload(
    *,
    ticker: str,
    cik: str | None,
    as_of: date,
    years_back: int,
    start_date: date,
    end_date: date,
    forms: list[str],
    include_amendments: bool,
    include_foreign: bool,
    min_annual_filings: int,
    filings: list[FilingStub],
) -> dict[str, Any]:
    counts_by_form: dict[str, int] = {}
    for filing in filings:
        form_type = str(filing.form_type).upper()
        counts_by_form[form_type] = counts_by_form.get(form_type, 0) + 1
    annual_count = int(sum(counts_by_form.values()))
    cik_present = bool(str(cik or "").strip())
    eligible = cik_present and annual_count >= int(min_annual_filings)

    skip_reason: str | None = None
    if not cik_present:
        skip_reason = "CIK_MISSING"
    elif annual_count == 0:
        skip_reason = "NO_ANNUAL_FILING_IN_WINDOW"
    elif annual_count < int(min_annual_filings):
        skip_reason = "INSUFFICIENT_ANNUAL_FILINGS_IN_WINDOW"

    return {
        "ticker": ticker,
        "cik": cik,
        "query": {
            "as_of_date": as_of.isoformat(),
            "years_back": int(years_back),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "forms": list(forms),
            "include_amendments": bool(include_amendments),
            "include_foreign": bool(include_foreign),
        },
        "cik_present": bool(cik_present),
        "annual_forms_found": {k: counts_by_form[k] for k in sorted(counts_by_form.keys())},
        "annual_filing_count_window": int(annual_count),
        "min_annual_filings_required": int(min_annual_filings),
        "eligible": bool(eligible),
        "skip_reason": skip_reason,
    }


def _list_window_with_cached_fallback(
    client: SecClient,
    cik: str,
    *,
    ticker: str,
    start_date: date,
    end_date: date,
    forms: list[str],
) -> list[FilingStub]:
    """Window listing with the cached-submissions fallback; offline
    (net disabled) goes straight to cache."""
    cfg = get_config()
    net_disabled = str(getattr(cfg, "net_provider", "enabled")).strip().lower() == "disabled"
    if net_disabled:
        filings = client.list_cached_filings_window(
            cik,
            start_date=start_date,
            end_date=end_date,
            forms=forms,
        )
        if filings:
            logger.info("dossier_preflight_cached_submissions_fallback ticker=%s cik=%s", ticker, cik)
        return filings
    try:
        return client.list_filings_window(
            cik,
            start_date=start_date,
            end_date=end_date,
            forms=forms,
        )
    except Exception:
        filings = client.list_cached_filings_window(
            cik,
            start_date=start_date,
            end_date=end_date,
            forms=forms,
        )
        if filings:
            logger.info("dossier_preflight_cached_submissions_fallback ticker=%s cik=%s", ticker, cik)
            return filings
        raise


def _annual_preflight_with_filings(
    *,
    ticker: str,
    cik: str | None,
    as_of: date,
    years_back: int,
    include_amendments: bool,
    include_foreign: bool,
    min_annual_filings: int | None,
    client: SecClient | None = None,
    forms_override: list[str] | None = None,
) -> tuple[dict[str, Any], list[FilingStub]]:
    start_date, end_date = _annual_window_bounds(as_of=as_of, years_back=years_back)
    forms = list(forms_override or _annual_forms(include_amendments=include_amendments, include_foreign=include_foreign))
    min_required = _min_annual_filings_required(years_back=years_back, min_annual_filings=min_annual_filings)
    client = client or SecClient()

    filings: list[FilingStub] = []
    cik_norm = str(cik or "").strip()
    if cik_norm:
        filings = _list_window_with_cached_fallback(
            client,
            cik_norm,
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            forms=forms,
        )
    payload = _build_preflight_payload(
        ticker=ticker,
        cik=cik_norm or None,
        as_of=as_of,
        years_back=years_back,
        start_date=start_date,
        end_date=end_date,
        forms=forms,
        include_amendments=include_amendments,
        include_foreign=include_foreign,
        min_annual_filings=min_required,
        filings=filings,
    )
    return payload, filings


def preflight_annual_eligibility(
    *,
    ticker: str,
    as_of_date: str,
    years_back: int = 10,
    include_amendments: bool = True,
    include_foreign: bool = True,
    min_annual_filings: int | None = None,
    forms_override: list[str] | None = None,
    cik_hint: str | None = None,
) -> dict[str, Any]:
    ticker_norm = str(ticker or "").strip().upper()
    company = _company_row(ticker_norm)
    as_of = date.fromisoformat(as_of_date)
    cik = str(cik_hint or "").strip()
    if not cik and company:
        cik = str(company.get("cik") or "").strip()
    if not cik:
        cik = str(resolve_cik_for_ticker(ticker_norm, refresh_if_missing=False) or "").strip()
    if not company and not cik:
        start_date, end_date = _annual_window_bounds(as_of=as_of, years_back=years_back)
        forms = list(forms_override or _annual_forms(include_amendments=include_amendments, include_foreign=include_foreign))
        return {
            "ticker": ticker_norm,
            "cik": None,
            "query": {
                "as_of_date": as_of_date,
                "years_back": int(years_back),
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "forms": forms,
                "include_amendments": bool(include_amendments),
                "include_foreign": bool(include_foreign),
            },
            "cik_present": False,
            "annual_forms_found": {},
            "annual_filing_count_window": 0,
            "min_annual_filings_required": _min_annual_filings_required(
                years_back=years_back,
                min_annual_filings=min_annual_filings,
            ),
            "eligible": False,
            "skip_reason": "COMPANY_NOT_FOUND",
        }
    payload, _ = _annual_preflight_with_filings(
        ticker=ticker_norm,
        cik=cik,
        as_of=as_of,
        years_back=years_back,
        include_amendments=include_amendments,
        include_foreign=include_foreign,
        min_annual_filings=min_annual_filings,
        forms_override=forms_override,
    )
    return payload


def _select_annual_filings(
    filings: list[FilingStub],
    *,
    years_back: int,
    as_of_date: date,
    include_amendments: bool,
    include_foreign: bool = True,
    forms_override: list[str] | None = None,
) -> list[Any]:
    allowed = {
        str(form).upper()
        for form in (
            forms_override
            or _annual_forms(include_amendments=include_amendments, include_foreign=include_foreign)
        )
    }
    annual = [f for f in filings if str(f.form_type).upper() in allowed and f.filing_date <= as_of_date]
    selected_by_year: dict[int, Any] = {}
    for filing in annual:
        year = _period_year(filing.period_end, filing.filing_date)
        existing = selected_by_year.get(year)
        if existing is None:
            selected_by_year[year] = filing
            continue
        filing_key = (filing.filing_date, str(filing.accession), str(filing.form_type).upper())
        existing_key = (existing.filing_date, str(existing.accession), str(existing.form_type).upper())
        if filing_key > existing_key:
            selected_by_year[year] = filing
    selected = sorted(
        selected_by_year.values(),
        key=lambda f: (f.filing_date, _period_year(f.period_end, f.filing_date), str(f.accession)),
        reverse=True,
    )
    return selected[: max(1, int(years_back))]


def _annual_window_bounds(*, as_of: date, years_back: int) -> tuple[date, date]:
    return _subtract_years(as_of, years_back), as_of


def _expected_fiscal_years(*, as_of: date, years_back: int) -> list[int]:
    end = as_of.year - 1
    start = end - max(1, int(years_back)) + 1
    return list(range(end, start - 1, -1))


def _filing_summary(filing: FilingStub) -> dict[str, Any]:
    return {
        "accession": filing.accession,
        "form_type": str(filing.form_type).upper(),
        "filing_date": filing.filing_date.isoformat(),
        "period_end": filing.period_end,
        "fiscal_year": int(_period_year(filing.period_end, filing.filing_date)),
    }


def _build_collection_debug(
    *,
    ticker: str,
    cik: str | None,
    as_of: date,
    years_back: int,
    include_amendments: bool,
    include_foreign: bool,
    start_date: date,
    end_date: date,
    forms: list[str],
    min_annual_filings: int,
    filings: list[FilingStub],
    selected: list[FilingStub],
    preflight: dict[str, Any],
) -> dict[str, Any]:
    counts_by_form: dict[str, int] = {}
    annual_by_year: dict[int, int] = {}
    for filing in filings:
        form_type = str(filing.form_type).upper()
        counts_by_form[form_type] = counts_by_form.get(form_type, 0) + 1
        fiscal_year = _period_year(filing.period_end, filing.filing_date)
        annual_by_year[fiscal_year] = annual_by_year.get(fiscal_year, 0) + 1

    selected_years = {_period_year(f.period_end, f.filing_date) for f in selected}
    missing_years: list[dict[str, Any]] = []
    for fiscal_year in _expected_fiscal_years(as_of=as_of, years_back=years_back):
        if fiscal_year in selected_years:
            continue
        in_window_count = annual_by_year.get(fiscal_year, 0)
        if in_window_count == 0:
            reason = "NO_ANNUAL_FILING_FOR_FISCAL_YEAR"
        else:
            reason = "NOT_SELECTED_AFTER_YEAR_BUCKETING"
        missing_years.append(
            {
                "fiscal_year": int(fiscal_year),
                "reason": reason,
                "filings_in_window_for_year": int(in_window_count),
            }
        )

    payload = {
        "ticker": ticker,
        "cik": cik,
        "query": {
            "as_of_date": as_of.isoformat(),
            "years_back": int(years_back),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "forms": list(forms),
            "include_amendments": bool(include_amendments),
            "include_foreign": bool(include_foreign),
            "min_annual_filings": int(min_annual_filings),
        },
        "returned_count": int(len(filings)),
        "counts_by_form_type": {k: counts_by_form[k] for k in sorted(counts_by_form.keys())},
        "selected_count": int(len(selected)),
        "selected_accessions": [_filing_summary(filing) for filing in selected],
        "missing_years": missing_years,
        "preflight": preflight,
    }
    payload["cik_present"] = bool(preflight.get("cik_present"))
    payload["annual_forms_found"] = preflight.get("annual_forms_found") or {}
    payload["eligible"] = bool(preflight.get("eligible"))
    payload["skip_reason"] = preflight.get("skip_reason")
    return payload


def _download_primary_doc_no_db(client: SecClient, filing: FilingStub) -> str | None:
    doc_name = filing.primary_document or SecClient.filename_from_url(filing.primary_doc_url)
    local_path = filing_local_path(filing.cik, filing.accession, doc_name)
    index_path = local_path.parent / "index.json"
    if local_path.exists() and local_path.is_file():
        return str(local_path)
    if warm_cached_filing_to_raw(filing=filing, target_path=local_path):
        return str(local_path)
    cfg = get_config()
    net_disabled = str(getattr(cfg, "net_provider", "enabled")).strip().lower() == "disabled"
    if net_disabled:
        return None
    try:
        index_raw = client.download_bytes(filing.filing_index_url, use_cache=True)
        index_path.write_bytes(index_raw)
    except Exception as exc:  # noqa: BLE001
        logger.info(
            "dossier_index_download_failed",
            extra={"stage_name": "dossier", "stage_error": str(exc), "stage_accession": filing.accession},
        )
    try:
        payload = client.download_bytes(filing.primary_doc_url, use_cache=True)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "dossier_primary_download_failed",
            extra={"stage_name": "dossier", "stage_error": str(exc), "stage_accession": filing.accession},
        )
        return None
    local_path.write_bytes(payload)
    return str(local_path)


def collect_10k_docket_stage1(
    *,
    ticker: str,
    as_of_date: str,
    years_back: int = 10,
    include_amendments: bool = True,
    include_foreign: bool = True,
    min_annual_filings: int | None = None,
) -> list[DossierStage1Filing]:
    stage1, _ = collect_10k_docket_stage1_with_debug(
        ticker=ticker,
        as_of_date=as_of_date,
        years_back=years_back,
        include_amendments=include_amendments,
        include_foreign=include_foreign,
        min_annual_filings=min_annual_filings,
    )
    return stage1


def collect_10k_docket_stage1_with_debug(
    *,
    ticker: str,
    as_of_date: str,
    years_back: int = 10,
    include_amendments: bool = True,
    include_foreign: bool = True,
    min_annual_filings: int | None = None,
    forms_override: list[str] | None = None,
) -> tuple[list[DossierStage1Filing], dict[str, Any]]:
    ticker_norm = ticker.strip().upper()
    as_of = date.fromisoformat(as_of_date)
    start_date, end_date = _annual_window_bounds(as_of=as_of, years_back=years_back)
    forms = list(forms_override or _annual_forms(include_amendments=include_amendments, include_foreign=include_foreign))
    min_required = _min_annual_filings_required(years_back=years_back, min_annual_filings=min_annual_filings)
    company = _company_row(ticker_norm)
    cik = str((company or {}).get("cik") or "").strip()
    if not cik:
        cik = str(resolve_cik_for_ticker(ticker_norm, refresh_if_missing=False) or "").strip()
    if not cik:
        return [], {
            "ticker": ticker_norm,
            "cik": None,
            "query": {
                "as_of_date": as_of_date,
                "years_back": int(years_back),
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "forms": forms,
                "include_amendments": bool(include_amendments),
                "include_foreign": bool(include_foreign),
                "min_annual_filings": int(min_required),
            },
            "returned_count": 0,
            "counts_by_form_type": {},
            "annual_forms_found": {},
            "cik_present": False,
            "eligible": False,
            "skip_reason": "COMPANY_NOT_FOUND",
            "selected_count": 0,
            "selected_accessions": [],
            "missing_years": [],
            "error": "COMPANY_NOT_FOUND",
            "preflight": {
                "ticker": ticker_norm,
                "cik": None,
                "query": {
                    "as_of_date": as_of_date,
                    "years_back": int(years_back),
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "forms": forms,
                    "include_amendments": bool(include_amendments),
                    "include_foreign": bool(include_foreign),
                },
                "cik_present": False,
                "annual_forms_found": {},
                "annual_filing_count_window": 0,
                "min_annual_filings_required": int(min_required),
                "eligible": False,
                "skip_reason": "COMPANY_NOT_FOUND",
            },
        }

    client = SecClient()
    preflight, filings = _annual_preflight_with_filings(
        ticker=ticker_norm,
        cik=cik,
        as_of=as_of,
        years_back=years_back,
        include_amendments=include_amendments,
        include_foreign=include_foreign,
        min_annual_filings=min_annual_filings,
        client=client,
        forms_override=forms,
    )

    selected: list[FilingStub] = []
    if preflight.get("eligible"):
        selected = _select_annual_filings(
            filings,
            years_back=years_back,
            as_of_date=as_of,
            include_amendments=include_amendments,
            include_foreign=include_foreign,
            forms_override=forms,
        )
        if not selected:
            preflight = dict(preflight)
            preflight["eligible"] = False
            preflight["skip_reason"] = "NO_ANNUAL_FILING_AFTER_YEAR_BUCKETING"

    debug = _build_collection_debug(
        ticker=ticker_norm,
        cik=cik,
        as_of=as_of,
        years_back=years_back,
        include_amendments=include_amendments,
        include_foreign=include_foreign,
        start_date=start_date,
        end_date=end_date,
        forms=forms,
        min_annual_filings=min_required,
        filings=filings,
        selected=selected,
        preflight=preflight,
    )
    if not preflight.get("eligible"):
        debug["downloaded_count"] = 0
        debug["download_failures"] = []
        return [], debug

    stage1: list[DossierStage1Filing] = []
    download_failures: list[dict[str, str]] = []

    def _download_filing(filing: Any) -> tuple[Any, str | None]:
        return filing, _download_primary_doc_no_db(client, filing)

    cfg = get_config()
    n_workers = min(cfg.max_workers, max(1, len(selected)))
    results: list[tuple[Any, str | None]] = []
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_download_filing, f): f for f in selected}
        for future in as_completed(futures):
            results.append(future.result())

    for filing, local_path in results:
        if not local_path:
            download_failures.append(
                {
                    "accession": filing.accession,
                    "reason": "PRIMARY_DOCUMENT_DOWNLOAD_FAILED",
                }
            )
        stage1.append(
            DossierStage1Filing(
                ticker=ticker_norm,
                cik=cik,
                filing=filing,
                local_path=local_path,
            )
        )
    stage1.sort(key=lambda f: f.filing.filing_date, reverse=True)
    debug["downloaded_count"] = len([item for item in stage1 if item.local_path])
    debug["download_failures"] = download_failures
    return stage1, debug


def materialize_and_parse_docket_stage1(
    *,
    stage1: list[DossierStage1Filing],
    as_of_date: str,
) -> list[DossierFiling]:
    if not stage1:
        return []
    client = SecClient()
    docket: list[DossierFiling] = []
    for item in stage1:
        with DB_WRITE_LOCK:
            with get_db() as conn:
                filing_id = _upsert_filing(conn, item.ticker, item.filing, as_of_date)
                local_path = item.local_path
                if local_path and Path(local_path).exists():
                    digest = file_hash(Path(local_path))
                    conn.execute(
                        "UPDATE filings SET local_path=?, hash=?, status='downloaded', updated_at=? WHERE id=?",
                        (local_path, digest, utc_now_iso(), filing_id),
                    )
                else:
                    local_path = _download_filing(conn, client, filing_id, item.filing)
            if local_path and Path(local_path).exists():
                parse_filing_by_id(filing_id, max_retries=3)
            with get_db() as conn:
                row = conn.execute(
                    """
                    SELECT local_path
                    FROM filings
                    WHERE id = ?
                    LIMIT 1
                    """,
                    (filing_id,),
                ).fetchone()
                local_path = row["local_path"] if row else local_path
        docket.append(
            DossierFiling(
                ticker=item.ticker,
                cik=item.cik,
                accession=item.filing.accession,
                form_type=str(item.filing.form_type).upper(),
                filing_date=item.filing.filing_date.isoformat(),
                period_end=item.filing.period_end,
                primary_doc_url=item.filing.primary_doc_url,
                local_path=local_path,
                filing_id=int(filing_id),
            )
        )
    docket.sort(key=lambda f: f.filing_date, reverse=True)
    return docket


def collect_10k_docket(
    *,
    ticker: str,
    as_of_date: str,
    years_back: int = 10,
    include_amendments: bool = True,
    include_foreign: bool = True,
    min_annual_filings: int | None = None,
    parse_downloaded: bool = True,
) -> list[DossierFiling]:
    stage1 = collect_10k_docket_stage1(
        ticker=ticker,
        as_of_date=as_of_date,
        years_back=years_back,
        include_amendments=include_amendments,
        include_foreign=include_foreign,
        min_annual_filings=min_annual_filings,
    )
    if not stage1:
        return []
    if parse_downloaded:
        return materialize_and_parse_docket_stage1(stage1=stage1, as_of_date=as_of_date)
    out: list[DossierFiling] = []
    for item in stage1:
        out.append(
            DossierFiling(
                ticker=item.ticker,
                cik=item.cik,
                accession=item.filing.accession,
                form_type=str(item.filing.form_type).upper(),
                filing_date=item.filing.filing_date.isoformat(),
                period_end=item.filing.period_end,
                primary_doc_url=item.filing.primary_doc_url,
                local_path=item.local_path,
                filing_id=-1,
            )
        )
    out.sort(key=lambda f: f.filing_date, reverse=True)
    return out


def _select_quarterly_filings(
    filings: list[FilingStub],
    *,
    quarters_back: int,
    as_of_date: date,
    include_amendments: bool,
) -> list[Any]:
    allowed = {str(form).upper() for form in _quarterly_forms(include_amendments=include_amendments)}
    quarterly = [f for f in filings if str(f.form_type).upper() in allowed and f.filing_date <= as_of_date]
    # Group by fiscal period; the last ``quarters_back`` periods survive.
    # Within a period the original AND its amendments are all kept (F-FS-5,
    # 2026-08-05): this is a READING docket — SHEN's same-day 39KB partial
    # 10-Q/A hid the 1.2MB substantive original 10-Q from the seat under the
    # old one-filing-per-period supersession. Supersession is the right rule
    # for parsed numbers, not for a surface whose purpose is letting the
    # analyst read filings. Missing period_end falls back to a unique bucket.
    by_period: dict[str, list[Any]] = {}
    for filing in quarterly:
        by_period.setdefault(str(filing.period_end or filing.accession), []).append(filing)
    ranked_periods = sorted(
        by_period.values(),
        key=lambda group: max(
            (f.filing_date, str(f.accession), str(f.form_type).upper()) for f in group
        ),
        reverse=True,
    )
    kept = ranked_periods[: max(1, int(quarters_back))]
    return sorted(
        (filing for group in kept for filing in group),
        key=lambda f: (f.filing_date, str(f.period_end or ""), str(f.accession)),
        reverse=True,
    )


def collect_quarterly_docket(
    *,
    ticker: str,
    as_of_date: str,
    quarters_back: int = 8,
    include_amendments: bool = True,
    parse_downloaded: bool = True,
) -> list[DossierFiling]:
    """The most recent quarterly filings (10-Q plus any 10-Q/A per period)
    filed at or before ``as_of_date``, newest first, primary documents
    downloaded. Same contract as ``collect_10k_docket``; selection keeps the
    last ``quarters_back`` discrete periods instead of annual year-bucketing,
    emitting every filing within each kept period (a reading docket never
    lets an amendment hide its original — F-FS-5, 2026-08-05).
    """
    ticker_norm = str(ticker or "").strip().upper()
    as_of = date.fromisoformat(as_of_date)
    quarters = max(1, int(quarters_back))
    # Enough whole years to hold quarters_back discrete periods plus one
    # year of slack for late filers and amendment lag.
    years_back = (quarters + 3) // 4 + 1
    start_date, end_date = _annual_window_bounds(as_of=as_of, years_back=years_back)
    company = _company_row(ticker_norm)
    cik = str((company or {}).get("cik") or "").strip()
    if not cik:
        cik = str(resolve_cik_for_ticker(ticker_norm, refresh_if_missing=False) or "").strip()
    if not cik:
        return []

    client = SecClient()
    filings = _list_window_with_cached_fallback(
        client,
        cik,
        ticker=ticker_norm,
        start_date=start_date,
        end_date=end_date,
        forms=_quarterly_forms(include_amendments=include_amendments),
    )
    selected = _select_quarterly_filings(
        filings,
        quarters_back=quarters,
        as_of_date=as_of,
        include_amendments=include_amendments,
    )
    if not selected:
        return []

    def _download(filing: Any) -> tuple[Any, str | None]:
        return filing, _download_primary_doc_no_db(client, filing)

    cfg = get_config()
    n_workers = min(cfg.max_workers, max(1, len(selected)))
    results: list[tuple[Any, str | None]] = []
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_download, f): f for f in selected}
        for future in as_completed(futures):
            results.append(future.result())

    stage1 = [
        DossierStage1Filing(ticker=ticker_norm, cik=cik, filing=filing, local_path=local_path)
        for filing, local_path in results
    ]
    stage1.sort(key=lambda item: item.filing.filing_date, reverse=True)
    if parse_downloaded:
        return materialize_and_parse_docket_stage1(stage1=stage1, as_of_date=as_of_date)
    out = [
        DossierFiling(
            ticker=item.ticker,
            cik=item.cik,
            accession=item.filing.accession,
            form_type=str(item.filing.form_type).upper(),
            filing_date=item.filing.filing_date.isoformat(),
            period_end=item.filing.period_end,
            primary_doc_url=item.filing.primary_doc_url,
            local_path=item.local_path,
            filing_id=-1,
        )
        for item in stage1
    ]
    out.sort(key=lambda f: f.filing_date, reverse=True)
    return out


def read_filing_text(filing: DossierFiling) -> str | None:
    if not filing.local_path:
        return None
    path = Path(filing.local_path)
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8", errors="ignore")
