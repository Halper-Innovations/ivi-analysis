"""Deterministic per-event-type detectors over parsed EDGAR index rows.

No LLM, no network: history access goes only through injected reader
protocols (submissions/registration/reporting-history), so the same rows +
payloads always produce the same event/evidence/skip rows. Each detector is
two-phase within a day — registrations/anchors first, then qualifiers — so a
same-day pair (e.g. a 10-12B and its CERT) links regardless of row order.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Protocol

from app.events import store
from app.events.index_feed import IndexRow


@dataclass
class DetectorCounters:
    events_created: int = 0
    events_updated: int = 0
    skips: int = 0
    counts: dict[str, int] = field(default_factory=dict)

    def bump(self, key: str, n: int = 1) -> None:
        self.counts[key] = self.counts.get(key, 0) + n


class RegistrationHistoryReader(Protocol):
    def prior_registration(self, cik: str, *, before: str) -> tuple[str, str] | None:
        """(accession, filing_date) of the latest 10-12B/10-12G before `before`."""


class SubmissionsReader(Protocol):
    def items_for(self, cik: str, accession: str) -> str | None:
        """Comma-joined 8-K items string from submissions JSON; None on failure."""

    def prior_item103(self, cik: str, *, before: str) -> tuple[str, str] | None:
        """(accession, filing_date) of the latest 8-K with item 1.03 before `before`."""


class ReportingHistoryReader(Protocol):
    def has_prior_periodic_filings(self, cik: str, *, before: str) -> bool | None:
        """True if any 10-K/10-Q filed before `before`; None on fetch failure."""


_NAME_SUFFIXES = {"CORP", "INC", "LLC", "CO", "NEW", "CORPORATION", "INCORPORATED"}


def normalize_company_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Z0-9 ]+", " ", name.upper())
    tokens = cleaned.split()
    while tokens and tokens[-1] in _NAME_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


def _plus_days(day: str, days: int) -> str:
    return (date.fromisoformat(day) + timedelta(days=days)).isoformat()


# ---------------------------------------------------------------------------
# Spinoff: 10-12B/10-12G registration -> CERT/EFFECT qualification
# ---------------------------------------------------------------------------

_REGISTRATION_FORMS = {"10-12B", "10-12B/A", "10-12G", "10-12G/A"}
_SPINOFF_QUALIFIER_FORMS = {"CERT", "EFFECT"}


def detect_spinoffs(
    conn: sqlite3.Connection,
    rows: list[IndexRow],
    *,
    scan_date: str,
    source_mode: str,
    registrations: RegistrationHistoryReader,
) -> DetectorCounters:
    counters = DetectorCounters()
    registration_rows = [r for r in rows if r.form_type in _REGISTRATION_FORMS]
    qualifier_rows = [r for r in rows if r.form_type in _SPINOFF_QUALIFIER_FORMS]

    for row in registration_rows:
        is_amendment = row.form_type.endswith("/A")
        is_otc = row.form_type.startswith("10-12G")
        event = store.find_active_event(conn, cik=row.cik, event_type="spinoff")
        if event is None:
            detail = {"listing_track": "otc"} if is_otc else None
            event_id = store.upsert_event(
                conn, cik=row.cik, event_type="spinoff",
                anchor_accession=row.accession, company_name=row.company_name,
                detection_date=row.date_filed, source_mode=source_mode, detail=detail,
            )
            counters.events_created += 1
        else:
            event_id = int(event["id"])
        role = "AMENDMENT" if is_amendment else "REGISTRATION"
        attached = store.attach_filing(
            conn, event_id=event_id, cik=row.cik, accession=row.accession,
            form_type=row.form_type, filing_date=row.date_filed, role=role,
        )
        if attached and event is not None:
            counters.events_updated += 1

    for row in qualifier_rows:
        event = store.find_active_event(conn, cik=row.cik, event_type="spinoff")
        if event is not None:
            role = "CERTIFICATION" if row.form_type == "CERT" else "EFFECTIVENESS"
            attached = store.attach_filing(
                conn, event_id=int(event["id"]), cik=row.cik, accession=row.accession,
                form_type=row.form_type, filing_date=row.date_filed, role=role,
            )
            qualified = store.mark_qualified(
                conn, event_id=int(event["id"]), qualification_date=row.date_filed,
            )
            if attached or qualified:
                counters.events_updated += 1
            continue
        if row.form_type != "CERT":
            # Unmatched EFFECT gets no backstop (most EFFECTs are unrelated S-filings).
            counters.bump("n_effect_unmatched")
            continue
        # CERT backstop: the registration may pre-date the scan window.
        prior = registrations.prior_registration(row.cik, before=row.date_filed)
        if prior is None:
            counters.bump("n_cert_backstop_miss")
            continue
        anchor_accession, anchor_date = prior
        event_id = store.upsert_event(
            conn, cik=row.cik, event_type="spinoff",
            anchor_accession=anchor_accession, company_name=row.company_name,
            detection_date=row.date_filed, source_mode=source_mode,
            detail={"registration_filing_date": anchor_date},
        )
        counters.events_created += 1
        store.attach_filing(
            conn, event_id=event_id, cik=row.cik, accession=anchor_accession,
            form_type="10-12B", filing_date=anchor_date, role="REGISTRATION",
        )
        store.attach_filing(
            conn, event_id=event_id, cik=row.cik, accession=row.accession,
            form_type=row.form_type, filing_date=row.date_filed, role="CERTIFICATION",
        )
        store.mark_qualified(conn, event_id=event_id, qualification_date=row.date_filed)

    return counters


def apply_spinoff_auto_effect(conn: sqlite3.Connection, *, scan_date: str) -> int:
    """Exchange Act §12(g): 10-12G registrations go effective 60 days after filing."""
    rows = conn.execute(
        """
        SELECT id, detection_date FROM corporate_events
        WHERE event_type = 'spinoff' AND status = 'DETECTED'
          AND json_extract(detail_json, '$.listing_track') = 'otc'
        """
    ).fetchall()
    n = 0
    for row in rows:
        effective = _plus_days(row["detection_date"], 60)
        if effective <= scan_date:
            if store.mark_qualified(conn, event_id=int(row["id"]), qualification_date=effective):
                n += 1
    return n


# ---------------------------------------------------------------------------
# Ch11 emergence: 8-K item 1.03 -> Form 25 -> 8-A12B/CERT relisting
# ---------------------------------------------------------------------------

_RELISTING_FORMS = {"8-A12B", "8-A12B/A"}

# 1.03 backstop age bound: a petition anchor older than this cannot create a
# fresh emergence from an unlinked 8-A12B. Real window-edge cases are
# 2022-2023 petitions relisting in 2024 (1-2 years); the false-positive class
# is a long-emerged company re-registering paper (e.g. a rights-plan class)
# that matches its decade-old bankruptcy 8-K.
_CH11_BACKSTOP_MAX_AGE_DAYS = 1095


def detect_ch11(
    conn: sqlite3.Connection,
    rows: list[IndexRow],
    *,
    scan_date: str,
    source_mode: str,
    submissions: SubmissionsReader,
) -> DetectorCounters:
    counters = DetectorCounters()
    anchor_rows = [r for r in rows if r.form_type == "8-K"]
    delisting_rows = [r for r in rows if r.form_type in {"25", "25-NSE"}]
    relisting_rows = [r for r in rows if r.form_type in _RELISTING_FORMS or r.form_type == "CERT"]

    for row in anchor_rows:
        counters.bump("n_8k_scanned")
        items = submissions.items_for(row.cik, row.accession)
        if items is None:
            if store.record_skip(
                conn, scan_date=scan_date, cik=row.cik, accession=row.accession,
                form_type=row.form_type, detector="ch11_emergence",
                reason_code="SUBMISSIONS_FETCH_FAILED",
            ):
                counters.skips += 1
            continue
        item_codes = {item.strip() for item in items.split(",") if item.strip()}
        if "1.03" not in item_codes:
            continue
        counters.bump("n_8k_item103")
        event = store.find_active_event(conn, cik=row.cik, event_type="ch11_emergence")
        if event is None:
            event_id = store.upsert_event(
                conn, cik=row.cik, event_type="ch11_emergence",
                anchor_accession=row.accession, company_name=row.company_name,
                detection_date=row.date_filed, source_mode=source_mode,
            )
            counters.events_created += 1
        else:
            event_id = int(event["id"])
        attached = store.attach_filing(
            conn, event_id=event_id, cik=row.cik, accession=row.accession,
            form_type=row.form_type, filing_date=row.date_filed, role="BANKRUPTCY_8K",
        )
        if attached and event is not None:
            counters.events_updated += 1

    for row in delisting_rows:
        event = store.find_active_event(conn, cik=row.cik, event_type="ch11_emergence")
        if event is None:
            if row.form_type == "25-NSE":
                # Dual-indexed under exchange CIKs; ~1,700+/yr junk — counted, not skipped.
                counters.bump("n_25nse_unlinked")
            else:
                counters.bump("n_25_unlinked")
            continue
        attached = store.attach_filing(
            conn, event_id=int(event["id"]), cik=row.cik, accession=row.accession,
            form_type=row.form_type, filing_date=row.date_filed, role="DELISTING",
        )
        if attached:
            counters.events_updated += 1

    open_events = conn.execute(
        """
        SELECT * FROM corporate_events
        WHERE event_type = 'ch11_emergence' AND status NOT IN ('DECIDED', 'EXPIRED')
        """
    ).fetchall()
    names_to_events: dict[str, list[sqlite3.Row]] = {}
    for event in open_events:
        names_to_events.setdefault(normalize_company_name(event["company_name"]), []).append(event)

    for row in relisting_rows:
        event = store.find_active_event(conn, cik=row.cik, event_type="ch11_emergence")
        if event is not None:
            attached = store.attach_filing(
                conn, event_id=int(event["id"]), cik=row.cik, accession=row.accession,
                form_type=row.form_type, filing_date=row.date_filed, role="RELISTING",
            )
            qualified = store.mark_qualified(
                conn, event_id=int(event["id"]), qualification_date=row.date_filed,
            )
            if attached or qualified:
                counters.events_updated += 1
            continue
        if row.form_type == "CERT":
            # CERT with no open ch11 event belongs to the spinoff detector.
            continue
        if row.form_type.endswith("/A"):
            # An 8-A12B AMENDMENT means the registration already exists (e.g.
            # a rights-plan class being amended) — never a fresh relisting.
            # Without a CIK or name link it is counted, not created.
            counters.bump("n_8a12b_amendment_unlinked")
            continue
        matches = names_to_events.get(normalize_company_name(row.company_name), [])
        matches = [m for m in matches if m["cik"] != row.cik]
        if len(matches) > 1:
            if store.record_skip(
                conn, scan_date=scan_date, cik=row.cik, accession=row.accession,
                form_type=row.form_type, detector="ch11_emergence",
                reason_code="AMBIGUOUS_NAME_MATCH",
            ):
                counters.skips += 1
            continue
        if len(matches) == 1:
            matched = matches[0]
            store.attach_filing(
                conn, event_id=int(matched["id"]), cik=row.cik, accession=row.accession,
                form_type=row.form_type, filing_date=row.date_filed, role="RELISTING",
                detail={"cross_cik_link": matched["cik"]},
            )
            store.merge_event_detail(
                conn, event_id=int(matched["id"]), detail={"relisting_cik": row.cik},
            )
            store.reset_ticker(conn, event_id=int(matched["id"]))
            store.mark_qualified(
                conn, event_id=int(matched["id"]), qualification_date=row.date_filed,
            )
            counters.events_updated += 1
            continue
        # 1.03 backstop: the petition routinely pre-dates the scan window.
        prior = submissions.prior_item103(row.cik, before=row.date_filed)
        if prior is not None:
            anchor_age = (
                date.fromisoformat(row.date_filed) - date.fromisoformat(prior[1])
            ).days
            if anchor_age > _CH11_BACKSTOP_MAX_AGE_DAYS:
                prior = None
        if prior is not None:
            anchor_accession, anchor_date = prior
            event_id = store.upsert_event(
                conn, cik=row.cik, event_type="ch11_emergence",
                anchor_accession=anchor_accession, company_name=row.company_name,
                detection_date=row.date_filed, source_mode=source_mode,
            )
            counters.events_created += 1
            store.attach_filing(
                conn, event_id=event_id, cik=row.cik, accession=anchor_accession,
                form_type="8-K", filing_date=anchor_date, role="BANKRUPTCY_8K",
            )
            store.attach_filing(
                conn, event_id=event_id, cik=row.cik, accession=row.accession,
                form_type=row.form_type, filing_date=row.date_filed, role="RELISTING",
            )
            store.mark_qualified(conn, event_id=event_id, qualification_date=row.date_filed)
            continue
        if store.record_skip(
            conn, scan_date=scan_date, cik=row.cik, accession=row.accession,
            form_type=row.form_type, detector="ch11_emergence",
            reason_code="NO_CH11_LINK_8A",
        ):
            counters.skips += 1

    return counters


# ---------------------------------------------------------------------------
# Queue protection: watchlist-scoped event flags (EVENT_PENDING:<TYPE>)
# ---------------------------------------------------------------------------

# 8-K items that mean a watchlist name is carrying a known event.
_QUEUE_PROTECTION_8K_ITEMS = {
    "1.01": "material_agreement",
    "1.03": "bankruptcy",
    "2.04": "obligation_acceleration",
    "2.05": "restructuring",
    "3.01": "delisting_notice",
    "4.02": "restatement",
}

_QUEUE_PROTECTION_ROLES = {
    "merger": "MERGER_FILING",
    "activist": "ACTIVIST_FILING",
    "dilution": "DILUTION_FILING",
    "material_agreement": "ITEM_8K",
    "bankruptcy": "ITEM_8K",
    "obligation_acceleration": "ITEM_8K",
    "restructuring": "ITEM_8K",
    "delisting_notice": "ITEM_8K",
    "restatement": "ITEM_8K",
}


def queue_protection_type_for_form(form_type: str) -> str | None:
    """Map an EDGAR form type to a queue-protection event type (non-8-K forms)."""
    form = form_type.strip().upper()
    # A Form 25/25-NSE by a WATCHLIST CIK is a live delisting — it must
    # open a queue-protection event (EVENT_PENDING:DELISTING_NOTICE blocks
    # DEPLOY_READY presentation), not merely be counted. The 25-NSE junk
    # concern doesn't apply here: this map only runs against watchlist CIKs.
    if form in {"25", "25-NSE"}:
        return "delisting_notice"
    if form == "425":
        return "merger"
    if form.startswith("DEFM14") or form.startswith("PREM14"):
        return "merger"
    if form.startswith("SC TO"):
        return "merger"
    if form.startswith("SC 13E3") or form.startswith("SC 13E-3"):
        return "merger"
    if form.startswith("SC 13D"):
        return "activist"
    if form in {"S-1", "S-1/A", "S-1MEF"} or form.startswith("424B"):
        return "dilution"
    return None


def detect_queue_protection(
    conn: sqlite3.Connection,
    rows: list[IndexRow],
    *,
    scan_date: str,
    source_mode: str,
    watchlist_ciks: dict[str, str],
    submissions: SubmissionsReader,
) -> DetectorCounters:
    """Watchlist-scoped queue-protection events.

    Any filing in the protected families (M&A paper, activist stakes,
    dilution registrations, flagged 8-K items) by a CIK currently on the
    watchlist opens — or extends — an event of the mapped type. Open events
    render as EVENT_PENDING:<TYPE> on every surface and block DEPLOY_READY
    presentation until an analyst disposal (store.mark_decided).
    """
    counters = DetectorCounters()

    def _open_event(row: IndexRow, event_type: str, *, role: str, detail: dict | None = None) -> None:
        ticker = watchlist_ciks[row.cik]
        event = store.find_active_event(conn, cik=row.cik, event_type=event_type)
        if event is None:
            event_id = store.upsert_event(
                conn, cik=row.cik, event_type=event_type,
                anchor_accession=row.accession, company_name=row.company_name,
                detection_date=row.date_filed, source_mode=source_mode,
                detail={"watchlist_ticker": ticker},
            )
            store.set_ticker(conn, event_id=event_id, ticker=ticker)
            counters.events_created += 1
            created = True
        else:
            event_id = int(event["id"])
            created = False
        attached = store.attach_filing(
            conn, event_id=event_id, cik=row.cik, accession=row.accession,
            form_type=row.form_type, filing_date=row.date_filed, role=role,
            detail=detail,
        )
        if attached and not created:
            counters.events_updated += 1

    for row in rows:
        if row.cik not in watchlist_ciks:
            continue
        counters.bump("n_watchlist_candidate_rows")
        if row.form_type == "8-K":
            items = submissions.items_for(row.cik, row.accession)
            if items is None:
                if store.record_skip(
                    conn, scan_date=scan_date, cik=row.cik, accession=row.accession,
                    form_type=row.form_type, detector="queue_protection",
                    reason_code="SUBMISSIONS_FETCH_FAILED",
                ):
                    counters.skips += 1
                continue
            item_codes = [item.strip() for item in items.split(",") if item.strip()]
            for code in item_codes:
                event_type = _QUEUE_PROTECTION_8K_ITEMS.get(code)
                if event_type is None:
                    continue
                _open_event(
                    row, event_type,
                    role=_QUEUE_PROTECTION_ROLES[event_type],
                    detail={"item": code, "items": item_codes},
                )
            continue
        event_type = queue_protection_type_for_form(row.form_type)
        if event_type is None:
            continue
        _open_event(row, event_type, role=_QUEUE_PROTECTION_ROLES[event_type])

    return counters


# ---------------------------------------------------------------------------
# 8-K freshness: at-target names may not carry an unread fresh 8-K
# ---------------------------------------------------------------------------


class FilingsWindowReader(Protocol):
    def filings_window(self, cik: str, *, start: str, end: str) -> list[dict[str, str]] | None:
        """All filings in [start, end] with item codes, newest first; None on fetch failure."""


def detect_atarget_8k_freshness(
    conn: sqlite3.Connection,
    atarget_ciks: dict[str, str],
    *,
    scan_date: str,
    lookback_days: int,
    source_mode: str,
    filings: FilingsWindowReader,
) -> DetectorCounters:
    """Every 8-K filed within the lookback by an at-target name opens an
    unreviewed_8k event unless the accession already anchors or is attached to
    another event (the item-mapped detectors keep precedence). Item 8.01
    lawsuits/scandals land here — too noisy to gate the whole watchlist, but a
    name at its buy trigger must not be bought with an unread fresh 8-K.
    """
    counters = DetectorCounters()
    start = _plus_days(scan_date, -lookback_days)
    for cik, ticker in sorted(atarget_ciks.items()):
        window = filings.filings_window(cik, start=start, end=scan_date)
        if window is None:
            counters.bump("n_8k_freshness_fetch_failed")
            continue
        for filing in window:
            if str(filing.get("form", "")).upper() != "8-K":
                continue
            accession = str(filing.get("accession", ""))
            if not accession:
                continue
            known = conn.execute(
                "SELECT 1 FROM corporate_events WHERE anchor_accession = ? "
                "UNION SELECT 1 FROM corporate_event_filings WHERE accession = ? LIMIT 1",
                (accession, accession),
            ).fetchone()
            if known is not None:
                continue
            item_codes = [
                item.strip() for item in str(filing.get("items", "")).split(",") if item.strip()
            ]
            event_id = store.upsert_event(
                conn, cik=cik, event_type="unreviewed_8k",
                anchor_accession=accession, company_name=ticker,
                detection_date=str(filing.get("filing_date") or scan_date),
                source_mode=source_mode,
                detail={"watchlist_ticker": ticker, "items": item_codes},
            )
            store.set_ticker(conn, event_id=event_id, ticker=ticker)
            store.attach_filing(
                conn, event_id=event_id, cik=cik, accession=accession,
                form_type="8-K", filing_date=str(filing.get("filing_date") or scan_date),
                role="FRESH_8K", detail={"items": item_codes},
            )
            counters.events_created += 1
    return counters


# ---------------------------------------------------------------------------
# Litigation dockets: recent CourtListener/RECAP suits against queue names
# ---------------------------------------------------------------------------


class DocketSearcher(Protocol):
    def search_recent_dockets(self, company_name: str, *, filed_after: str) -> list | None:
        """DocketHit list for recent suits naming the company; None on fetch failure."""


def detect_litigation_dockets(
    conn: sqlite3.Connection,
    targets: list[dict[str, str]],
    *,
    scan_date: str,
    lookback_days: int,
    nos_codes: set[int],
    source_mode: str,
    searcher: DocketSearcher,
) -> DetectorCounters:
    """A recent docket naming a target company opens a litigation_docket event
    when its nature-of-suit code is in the severity whitelist and the company
    name matches the case caption conservatively (full token sequence). Anchor
    is the synthetic "CL-<docket_id>" — no EDGAR accession exists."""
    from app.events.courtlistener import (
        MIN_NORMALIZED_NAME_LEN,
        case_name_matches,
    )

    counters = DetectorCounters()
    filed_after = _plus_days(scan_date, -lookback_days)
    for target in targets:
        company_name = target["company_name"]
        if len(normalize_company_name(company_name)) < MIN_NORMALIZED_NAME_LEN:
            counters.bump("n_docket_name_too_generic")
            continue
        hits = searcher.search_recent_dockets(company_name, filed_after=filed_after)
        if hits is None:
            counters.bump("n_docket_fetch_failed")
            continue
        for hit in hits:
            if hit.nos_code not in nos_codes:
                counters.bump("n_docket_filtered_nos")
                continue
            if hit.date_filed and hit.date_filed < filed_after:
                continue
            if not case_name_matches(company_name, hit.case_name):
                counters.bump("n_docket_filtered_name")
                continue
            anchor = f"CL-{hit.docket_id}"
            known = conn.execute(
                "SELECT 1 FROM corporate_events WHERE anchor_accession = ? LIMIT 1",
                (anchor,),
            ).fetchone()
            if known is not None:
                continue
            event_id = store.upsert_event(
                conn, cik=target["cik"], event_type="litigation_docket",
                anchor_accession=anchor, company_name=company_name,
                detection_date=hit.date_filed or scan_date, source_mode=source_mode,
                detail={
                    "watchlist_ticker": target["ticker"],
                    "case_name": hit.case_name,
                    "court": hit.court,
                    "nature_of_suit": hit.nature_of_suit,
                    "nos_code": hit.nos_code,
                    "docket_url": hit.url,
                },
            )
            store.set_ticker(conn, event_id=event_id, ticker=target["ticker"])
            counters.events_created += 1
    return counters


# ---------------------------------------------------------------------------
# Busted IPO: non-reporting S-1 -> 424B4 pricing (qualification stubbed)
# ---------------------------------------------------------------------------


def detect_busted_ipos(
    conn: sqlite3.Connection,
    rows: list[IndexRow],
    *,
    scan_date: str,
    source_mode: str,
    history: ReportingHistoryReader,
) -> DetectorCounters:
    counters = DetectorCounters()
    s1_rows = [r for r in rows if r.form_type in {"S-1", "S-1/A"}]
    pricing_rows = [r for r in rows if r.form_type == "424B4"]

    def _skip(row: IndexRow, reason: str) -> None:
        if store.record_skip(
            conn, scan_date=scan_date, cik=row.cik, accession=row.accession,
            form_type=row.form_type, detector="busted_ipo", reason_code=reason,
        ):
            counters.skips += 1

    for row in s1_rows:
        event = store.find_active_event(conn, cik=row.cik, event_type="busted_ipo")
        if event is not None:
            role = "AMENDMENT" if row.form_type.endswith("/A") else "IPO_S1"
            attached = store.attach_filing(
                conn, event_id=int(event["id"]), cik=row.cik, accession=row.accession,
                form_type=row.form_type, filing_date=row.date_filed, role=role,
            )
            if attached:
                counters.events_updated += 1
            continue
        has_history = history.has_prior_periodic_filings(row.cik, before=row.date_filed)
        if has_history is None:
            _skip(row, "SUBMISSIONS_FETCH_FAILED")
            continue
        if has_history:
            _skip(row, "PRIOR_REPORTING_S1")
            continue
        event_id = store.upsert_event(
            conn, cik=row.cik, event_type="busted_ipo",
            anchor_accession=row.accession, company_name=row.company_name,
            detection_date=row.date_filed, source_mode=source_mode,
        )
        counters.events_created += 1
        store.attach_filing(
            conn, event_id=event_id, cik=row.cik, accession=row.accession,
            form_type=row.form_type, filing_date=row.date_filed, role="IPO_S1",
        )

    for row in pricing_rows:
        event = store.find_active_event(conn, cik=row.cik, event_type="busted_ipo")
        if event is not None:
            attached = store.attach_filing(
                conn, event_id=int(event["id"]), cik=row.cik, accession=row.accession,
                form_type=row.form_type, filing_date=row.date_filed, role="IPO_PRICING",
            )
            if attached:
                store.merge_event_detail(
                    conn, event_id=int(event["id"]), detail={"priced_date": row.date_filed},
                )
                counters.events_updated += 1
            continue
        has_history = history.has_prior_periodic_filings(row.cik, before=row.date_filed)
        if has_history is None:
            _skip(row, "SUBMISSIONS_FETCH_FAILED")
            continue
        if has_history:
            _skip(row, "FOLLOWON_424B4")
            continue
        # The S-1 may pre-date the scan window: create from the 424B4 alone.
        event_id = store.upsert_event(
            conn, cik=row.cik, event_type="busted_ipo",
            anchor_accession=row.accession, company_name=row.company_name,
            detection_date=row.date_filed, source_mode=source_mode,
            detail={"priced_date": row.date_filed},
        )
        counters.events_created += 1
        store.attach_filing(
            conn, event_id=event_id, cik=row.cik, accession=row.accession,
            form_type=row.form_type, filing_date=row.date_filed, role="IPO_PRICING",
        )

    return counters
