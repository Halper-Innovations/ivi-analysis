"""Company resolution, search, profile and filing index, all from SEC EDGAR.

Ticker -> CIK goes through the repository's existing map
(``app.universe.ticker_cik_map``, SEC ``company_tickers.json``); profiles and
filing lists come from the submissions API via ``app.ingest.sec_client``.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from datetime import date
from typing import Any

import requests

from app.config import get_config
from app.ingest import cik_registry
from app.ingest.sec_client import SecClient
from app.mcp_server.errors import SecToolError
from app.universe import ticker_cik_map
from app.util.http import HttpClient

logger = logging.getLogger("app.mcp_server")

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
EXCHANGE_MAP_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"

TICKER_MAP_MAX_AGE_SECONDS = 7 * 24 * 3600
MAX_LOOKUP_MATCHES = 25
MAX_FILINGS = 200

_CIK_RE = re.compile(r"^(?:CIK)?\s*0*(\d{1,10})$", re.IGNORECASE)
_ACCESSION_RE = re.compile(r"^(\d{10})-?(\d{2})-?(\d{6})$")
_NAME_SUFFIXES = {
    "INC",
    "INCORPORATED",
    "CORP",
    "CORPORATION",
    "CO",
    "COMPANY",
    "LTD",
    "LIMITED",
    "PLC",
    "LLC",
    "LP",
    "NV",
    "SA",
    "AG",
    "THE",
}

_lock = threading.Lock()
_ticker_rows_cache: dict[str, Any] = {}
_exchange_cache: dict[str, Any] = {}


@dataclass(frozen=True)
class Company:
    cik: str  # zero-padded, 10 digits
    ticker: str | None
    name: str | None

    def as_dict(self) -> dict[str, Any]:
        return {"cik": self.cik, "ticker": self.ticker, "name": self.name}


# ---------------------------------------------------------------------------
# small parsers
# ---------------------------------------------------------------------------


def parse_cik(text: str) -> str | None:
    """Return a 10-digit CIK if ``text`` is CIK-shaped (``320193``, ``CIK0000320193``)."""

    match = _CIK_RE.match(str(text or "").strip())
    if not match or int(match.group(1)) == 0:
        return None
    return match.group(1).zfill(10)


def parse_accession(text: str) -> str | None:
    """Return ``0000000000-00-000000`` if ``text`` is an accession number (dashed or not)."""

    match = _ACCESSION_RE.match(str(text or "").strip())
    if not match:
        return None
    return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"


def parse_iso_date(value: str | None, field: str) -> date | None:
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip()
    try:
        if len(text) != 10:
            raise ValueError
        return date.fromisoformat(text)
    except ValueError:
        raise SecToolError(f"{field} must be a date in YYYY-MM-DD form, got {text!r}.") from None


def _normalize_ticker(text: str) -> str:
    return str(text or "").strip().upper().replace(".", "-").replace("/", "-")


def _normalize_name(text: str) -> str:
    # SEC names drop apostrophes ("NATHANS FAMOUS"), so "Nathan's" must become
    # "NATHANS", not "NATHAN S".
    unquoted = re.sub("['\u2019`]", "", str(text or "").upper())
    cleaned = re.sub(r"[^A-Z0-9 ]+", " ", unquoted.replace("&", " AND "))
    tokens = cleaned.split()
    while tokens and tokens[-1] in _NAME_SUFFIXES:
        tokens.pop()
    if tokens and tokens[0] == "THE":
        tokens = tokens[1:]
    return " ".join(tokens)


def _http() -> HttpClient:
    return HttpClient(get_config())


# ---------------------------------------------------------------------------
# ticker map (reuses app.universe.ticker_cik_map)
# ---------------------------------------------------------------------------


def _ensure_ticker_map() -> None:
    """Make sure SEC's ticker file is cached and not older than a week (best effort)."""

    path = ticker_cik_map.cached_mapping_path()
    stale = not path.exists() or (time.time() - path.stat().st_mtime) > TICKER_MAP_MAX_AGE_SECONDS
    if not stale:
        return
    try:
        ticker_cik_map.refresh_ticker_cik_cache()
    except Exception as exc:  # a stale map beats no map
        if not path.exists():
            raise
        logger.warning("Could not refresh the SEC ticker map; using the cached copy: %s", exc)


def _ticker_rows() -> list[dict[str, Any]]:
    """Rows of SEC's ticker file: ``{"ticker", "cik", "name"}``, in SEC order."""

    _ensure_ticker_map()
    path = ticker_cik_map.cached_mapping_path()
    if not path.exists():
        return []
    stamp = path.stat().st_mtime_ns
    with _lock:
        if _ticker_rows_cache.get("stamp") == stamp and _ticker_rows_cache.get("path") == str(path):
            return _ticker_rows_cache["rows"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    rows: list[dict[str, Any]] = []
    for entry in payload.values() if isinstance(payload, dict) else []:
        if not isinstance(entry, dict):
            continue
        ticker = str(entry.get("ticker") or "").strip().upper()
        cik = entry.get("cik_str")
        if not ticker or cik is None:
            continue
        rows.append({"ticker": ticker, "cik": str(cik).zfill(10), "name": str(entry.get("title") or "")})
    with _lock:
        _ticker_rows_cache.update({"stamp": stamp, "path": str(path), "rows": rows})
    return rows


def _exchanges_by_ticker() -> dict[str, str]:
    """Ticker -> exchange from SEC's exchange file. Best effort: ``{}`` on any failure."""

    with _lock:
        cached = _exchange_cache.get("map")
        fetched_at = _exchange_cache.get("at", 0.0)
    if cached is not None and time.time() - fetched_at < 3600:
        return cached
    mapping: dict[str, str] = {}
    try:
        payload = _http().get_json(EXCHANGE_MAP_URL, cache_ttl_seconds=TICKER_MAP_MAX_AGE_SECONDS)
        fields = [str(f) for f in payload.get("fields") or []]
        t_idx, e_idx = fields.index("ticker"), fields.index("exchange")
        for row in payload.get("data") or []:
            ticker, exchange = row[t_idx], row[e_idx]
            if ticker and exchange:
                mapping[str(ticker).upper()] = str(exchange)
    except Exception as exc:
        logger.warning("Exchange listing unavailable: %s", exc)
    with _lock:
        _exchange_cache.update({"map": mapping, "at": time.time()})
    return mapping


# ---------------------------------------------------------------------------
# submissions
# ---------------------------------------------------------------------------


def fetch_submissions(cik10: str) -> dict[str, Any]:
    try:
        payload = SecClient().submissions(cik10)
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            raise SecToolError(f"No SEC filer has CIK {cik10}.") from None
        raise
    if not isinstance(payload, dict) or not payload.get("cik"):
        raise SecToolError(f"SEC returned no filer record for CIK {cik10}.")
    return payload


def resolve_company(ticker_or_cik: str) -> Company:
    """Resolve a ticker (``KO``, ``BRK.B``) or CIK (``21344``, ``CIK0000021344``)."""

    raw = str(ticker_or_cik or "").strip()
    if not raw:
        raise SecToolError("ticker_or_cik is required (a ticker such as KO, or a CIK).")
    cik10 = parse_cik(raw)
    rows = _ticker_rows()
    if cik10:
        for row in rows:
            if row["cik"] == cik10:
                return Company(cik=cik10, ticker=row["ticker"], name=row["name"])
        return Company(cik=cik10, ticker=None, name=None)

    ticker = _normalize_ticker(raw)
    for candidate in dict.fromkeys([raw.upper(), ticker]):
        try:
            cik10 = cik_registry.resolve(candidate)
        except cik_registry.CIKNotFoundError:
            continue
        name = next((r["name"] for r in rows if r["ticker"] == candidate), None)
        return Company(cik=cik10, ticker=candidate, name=name)
    raise SecToolError(
        f"Unknown ticker {raw!r}. Use lookup_company to search by name, or pass the CIK."
    )


# ---------------------------------------------------------------------------
# tool: lookup_company
# ---------------------------------------------------------------------------


def lookup_company(query: str, limit: int = 10) -> dict[str, Any]:
    text = str(query or "").strip()
    if not text:
        raise SecToolError("query is required: a ticker, a CIK, or part of a company name.")
    limit = max(1, min(int(limit), MAX_LOOKUP_MATCHES))
    rows = _ticker_rows()
    exchanges = _exchanges_by_ticker()

    def as_match(row: dict[str, Any], matched_on: str) -> dict[str, Any]:
        return {
            "ticker": row["ticker"],
            "cik": row["cik"],
            "name": row["name"],
            "exchange": exchanges.get(row["ticker"]),
            "matched_on": matched_on,
        }

    matches: list[dict[str, Any]] = []
    cik10 = parse_cik(text)
    if cik10:
        matches = [as_match(r, "cik") for r in rows if r["cik"] == cik10]
        if not matches:
            # Not in SEC's ticker list (delisted, private debt issuer, fund...):
            # the submissions record still names it.
            try:
                sub = fetch_submissions(cik10)
            except SecToolError:
                sub = None
            if sub:
                tickers = [str(t) for t in sub.get("tickers") or []]
                sub_exchanges = [str(e) for e in sub.get("exchanges") or []]
                matches = [
                    {
                        "ticker": tickers[0] if tickers else None,
                        "cik": cik10,
                        "name": sub.get("name"),
                        "exchange": sub_exchanges[0] if sub_exchanges else None,
                        "matched_on": "cik",
                    }
                ]
    else:
        seen: set[tuple[str, str]] = set()
        ticker = _normalize_ticker(text)
        for row in rows:
            if row["ticker"] == ticker:
                matches.append(as_match(row, "ticker"))
                seen.add((row["cik"], row["ticker"]))
        needle = _normalize_name(text)
        if needle:
            tokens = needle.split()
            scored: list[tuple[int, int, str, dict[str, Any]]] = []
            for row in rows:
                if (row["cik"], row["ticker"]) in seen:
                    continue
                name = _normalize_name(row["name"])
                if not name:
                    continue
                if name == needle:
                    score = 0
                elif name.startswith(needle):
                    score = 1
                elif all(token in name.split() for token in tokens):
                    score = 2
                elif needle in name:
                    score = 3
                else:
                    continue
                scored.append((score, len(name), row["ticker"], row))
            scored.sort(key=lambda item: item[:3])
            for _, _, _, row in scored:
                matches.append(as_match(row, "name"))
    total = len(matches)
    result: dict[str, Any] = {"query": text, "match_count": total, "matches": matches[:limit]}
    if total > limit:
        result["truncated"] = True
    if not matches:
        result["hint"] = (
            "No SEC registrant matched. Try a shorter name fragment, the exact ticker, "
            "or the CIK from EDGAR full-text search."
        )
    return result


# ---------------------------------------------------------------------------
# tool: get_company_profile
# ---------------------------------------------------------------------------


def _iso_day(value: Any) -> str | None:
    text = str(value or "").strip()
    return text[:10] if text else None


def _record_rows(records: dict[str, Any], cik10: str) -> list[dict[str, Any]]:
    """Flatten a submissions column block into filing rows."""

    accessions = records.get("accessionNumber") or []
    columns = {
        key: records.get(key) or []
        for key in ("form", "filingDate", "reportDate", "primaryDocument", "primaryDocDescription", "items")
    }
    cik_int = int(cik10)
    rows: list[dict[str, Any]] = []
    for idx, accession in enumerate(accessions):

        def col(key: str, _idx: int = idx) -> str:
            values = columns[key]
            return str(values[_idx] or "").strip() if _idx < len(values) else ""

        filed = col("filingDate")
        try:
            date.fromisoformat(filed)
        except ValueError:
            continue
        nodash = str(accession).replace("-", "")
        primary = col("primaryDocument")
        row: dict[str, Any] = {
            "accession": str(accession),
            "form": col("form").upper(),
            "filed": filed,
            "report_date": col("reportDate") or None,
            "primary_document_url": f"{ARCHIVES_BASE}/{cik_int}/{nodash}/{primary}" if primary else None,
            "index_url": f"{ARCHIVES_BASE}/{cik_int}/{nodash}/{accession}-index.htm",
        }
        description = col("primaryDocDescription")
        if description:
            row["description"] = description
        items = col("items")
        if items:
            row["items"] = items
        rows.append(row)
    return rows


def _older_pages(submissions: dict[str, Any]) -> list[dict[str, Any]]:
    pages = [
        entry
        for entry in (submissions.get("filings") or {}).get("files") or []
        if isinstance(entry, dict) and str(entry.get("name") or "").strip()
    ]
    return sorted(pages, key=lambda e: str(e.get("filingTo") or ""), reverse=True)


def _page_rows(entry: dict[str, Any], cik10: str) -> list[dict[str, Any]]:
    url = f"https://data.sec.gov/submissions/{str(entry['name']).strip()}"
    payload = _http().get_json(url, cache_ttl_seconds=24 * 3600)
    return _record_rows(SecClient().records_from_payload(payload), cik10)


def _page_overlaps(entry: dict[str, Any], since: date | None, until: date | None) -> bool:
    try:
        page_from = date.fromisoformat(str(entry.get("filingFrom")))
        page_to = date.fromisoformat(str(entry.get("filingTo")))
    except ValueError:
        return True
    if since and page_to < since:
        return False
    if until and page_from > until:
        return False
    return True


def _latest(rows: list[dict[str, Any]], forms: set[str]) -> dict[str, Any] | None:
    for row in rows:
        if row["form"] in forms:
            return {k: row[k] for k in ("form", "filed", "report_date", "accession")}
    return None


def get_company_profile(ticker_or_cik: str) -> dict[str, Any]:
    company = resolve_company(ticker_or_cik)
    sub = fetch_submissions(company.cik)
    recent = SecClient().records_from_payload(sub)
    rows = sorted(_record_rows(recent, company.cik), key=lambda r: r["filed"], reverse=True)
    fye = str(sub.get("fiscalYearEnd") or "").strip()
    address = (sub.get("addresses") or {}).get("business") or {}
    address_text = ", ".join(
        part
        for part in (
            str(address.get("street1") or "").strip(),
            str(address.get("street2") or "").strip(),
            str(address.get("city") or "").strip(),
            " ".join(
                p
                for p in (
                    str(address.get("stateOrCountry") or "").strip(),
                    str(address.get("zipCode") or "").strip(),
                )
                if p
            ),
        )
        if part
    )
    profile: dict[str, Any] = {
        "cik": company.cik,
        "name": sub.get("name"),
        "tickers": [str(t) for t in sub.get("tickers") or []],
        "exchanges": [str(e) for e in sub.get("exchanges") or [] if e],
        "sic": str(sub.get("sic") or "") or None,
        "sic_description": sub.get("sicDescription") or None,
        "entity_type": sub.get("entityType") or None,
        "filer_category": sub.get("category") or None,
        "fiscal_year_end": f"{fye[:2]}-{fye[2:]}" if len(fye) == 4 else None,
        "state_of_incorporation": sub.get("stateOfIncorporation") or None,
        "ein": sub.get("ein") or None,
        "business_address": address_text or None,
        "phone": sub.get("phone") or None,
        "website": sub.get("website") or None,
        "former_names": [
            {"name": f.get("name"), "from": _iso_day(f.get("from")), "to": _iso_day(f.get("to"))}
            for f in sub.get("formerNames") or []
            if isinstance(f, dict)
        ],
        "latest_annual_report": _latest(rows, {"10-K", "10-K/A", "20-F", "20-F/A", "40-F", "40-F/A"}),
        "latest_quarterly_report": _latest(rows, {"10-Q", "10-Q/A"}),
        "source": SUBMISSIONS_URL.format(cik=company.cik),
    }
    return {key: value for key, value in profile.items() if value not in (None, "")}


# ---------------------------------------------------------------------------
# tool: list_filings
# ---------------------------------------------------------------------------


def _parse_forms(forms: list[str] | str | None) -> list[str]:
    if forms is None:
        return []
    items = forms.split(",") if isinstance(forms, str) else list(forms)
    return [str(f).strip().upper() for f in items if str(f).strip()]


def list_filings(
    ticker_or_cik: str,
    forms: list[str] | str | None = None,
    since: str | None = None,
    until: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    form_filter = _parse_forms(forms)
    since_d = parse_iso_date(since, "since")
    until_d = parse_iso_date(until, "until")
    if since_d and until_d and since_d > until_d:
        raise SecToolError("since must be on or before until.")
    limit = int(limit)
    if limit < 1:
        raise SecToolError("limit must be at least 1.")
    limit = min(limit, MAX_FILINGS)

    company = resolve_company(ticker_or_cik)
    sub = fetch_submissions(company.cik)

    def keep(row: dict[str, Any]) -> bool:
        if form_filter and row["form"] not in form_filter:
            return False
        filed = date.fromisoformat(row["filed"])
        if since_d and filed < since_d:
            return False
        if until_d and filed > until_d:
            return False
        return True

    matched = [r for r in _record_rows(SecClient().records_from_payload(sub), company.cik) if keep(r)]
    more_available = False
    # Older filings live in extra pages; fetch them only if the recent block
    # cannot fill the request. Pages are newest-first, so once we have
    # ``limit`` rows everything older can only be further down the list.
    for entry in _older_pages(sub):
        if not _page_overlaps(entry, since_d, until_d):
            continue
        if len(matched) >= limit:
            more_available = True
            break
        matched.extend(r for r in _page_rows(entry, company.cik) if keep(r))

    deduped = {row["accession"]: row for row in matched}
    ordered = sorted(deduped.values(), key=lambda r: (r["filed"], r["accession"]), reverse=True)
    if len(ordered) > limit:
        more_available = True
    result: dict[str, Any] = {
        "company": {"cik": company.cik, "ticker": company.ticker, "name": sub.get("name")},
        "filters": {
            "forms": form_filter or None,
            "since": since_d.isoformat() if since_d else None,
            "until": until_d.isoformat() if until_d else None,
            "limit": limit,
        },
        "returned": min(len(ordered), limit),
        "more_available": more_available,
        "filings": ordered[:limit],
    }
    if not ordered:
        result["hint"] = (
            "No filings matched. Form names are exact (amendments are separate, e.g. 10-K/A); "
            "widen the date range or drop the form filter."
        )
    return result


def find_filing(cik10: str, accession: str) -> dict[str, Any]:
    """The filing row for ``accession`` in the company's submissions, or SecToolError."""

    sub = fetch_submissions(cik10)
    for row in _record_rows(SecClient().records_from_payload(sub), cik10):
        if row["accession"] == accession:
            return row
    for entry in _older_pages(sub):
        for row in _page_rows(entry, cik10):
            if row["accession"] == accession:
                return row
    raise SecToolError(
        f"Accession {accession} is not among the filings of CIK {cik10}. "
        "Check the accession with list_filings, or pass the document URL instead."
    )
