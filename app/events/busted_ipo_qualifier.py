"""Busted-IPO price-decline qualifier.

Qualification rule (the contract this module stubbed until the price provider
was wired in): a non-reporting IPO is BUSTED when

    price(priced_date + seasoning_days) <= baseline * (1 - decline_threshold)

The baseline is the provider close at/just after ``priced_date`` (searched a
few days forward for the first print) — a debut-close proxy for the offer
price, chosen so qualification needs no prospectus parsing. Both prices and
the decline are merged into the event detail so the proxy is auditable.

Outcomes per priced DETECTED candidate:
  * QUALIFIED — seasoned price cleared the decline threshold (intake-eligible
    via ``ivi events surface``).
  * EXPIRED / DECLINE_THRESHOLD_NOT_MET — the seasoned-date price is a fixed
    historical fact, so a failed threshold test is terminal, not retried.
  * skip TICKER_UNRESOLVED / NOT_SEASONED / NO_PRICE_HISTORY — transient; the
    event stays DETECTED and re-evaluates on the next poll. Skips are recorded
    under detector ``busted_ipo_qualifier`` (the detection-time detector token
    already consumed the (cik, accession) dedup key for these accessions).
"""

from __future__ import annotations

import sqlite3
from datetime import date, timedelta

from app.events import store

QUALIFIER_DETECTOR = "busted_ipo_qualifier"
#: Days scanned forward from priced_date for the first available close.
BASELINE_SEARCH_DAYS = 5
EXPIRY_DECLINE_NOT_MET = "DECLINE_THRESHOLD_NOT_MET"


def _parse_date(value: object) -> date | None:
    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def _close_asof(provider, ticker: str, target: date) -> float | None:
    snapshot = provider.get_price_asof(ticker, target.isoformat())
    price = getattr(snapshot, "price", None)
    if isinstance(price, (int, float)) and float(price) > 0:
        return float(price)
    return None


def _baseline_close(provider, ticker: str, priced: date) -> tuple[float | None, str | None]:
    """First available close at/just after pricing (debut-close offer proxy)."""
    for offset in (0, BASELINE_SEARCH_DAYS):
        probe = priced + timedelta(days=offset)
        price = _close_asof(provider, ticker, probe)
        if price is not None:
            return price, probe.isoformat()
    return None, None


def qualify_busted_ipos(
    conn: sqlite3.Connection,
    *,
    as_of: str,
    provider=None,
    decline_threshold: float = 0.5,
    seasoning_days: int = 90,
) -> dict[str, int]:
    rows = conn.execute(
        """
        SELECT id, cik, ticker, anchor_accession,
               json_extract(detail_json, '$.priced_date') AS priced_date
        FROM corporate_events
        WHERE event_type = 'busted_ipo' AND status = 'DETECTED'
          AND json_extract(detail_json, '$.priced_date') IS NOT NULL
        """
    ).fetchall()
    counts = {"candidates": len(rows), "qualified": 0, "expired_not_met": 0, "skipped": 0}
    if not rows:
        return counts

    as_of_date = _parse_date(as_of)

    def _skip(row: sqlite3.Row, reason: str) -> None:
        store.record_skip(
            conn, scan_date=as_of, cik=row["cik"], accession=row["anchor_accession"],
            form_type="424B4", detector=QUALIFIER_DETECTOR, reason_code=reason,
        )
        counts["skipped"] += 1

    for row in rows:
        priced = _parse_date(row["priced_date"])
        if priced is None or as_of_date is None:
            _skip(row, "INDEX_ROW_MALFORMED")
            continue
        seasoned_date = priced + timedelta(days=seasoning_days)
        # Cheap checks first: no provider call before ticker + seasoning clear.
        if not row["ticker"]:
            _skip(row, "TICKER_UNRESOLVED")
            continue
        if seasoned_date > as_of_date:
            _skip(row, "NOT_SEASONED")
            continue
        if provider is None:
            from app.market.price_provider import get_default_provider

            provider = get_default_provider()
        ticker = str(row["ticker"]).upper()
        baseline, baseline_date = _baseline_close(provider, ticker, priced)
        seasoned = _close_asof(provider, ticker, seasoned_date)
        if baseline is None or seasoned is None:
            _skip(row, "NO_PRICE_HISTORY")
            continue
        decline = (baseline - seasoned) / baseline
        store.merge_event_detail(
            conn, event_id=int(row["id"]),
            detail={
                "qualifier": "PRICE_DECLINE",
                "baseline_price": round(baseline, 4),
                "baseline_date": baseline_date,
                "seasoned_price": round(seasoned, 4),
                "seasoned_date": seasoned_date.isoformat(),
                "decline_pct": round(decline, 4),
                "decline_threshold": decline_threshold,
            },
        )
        if seasoned <= baseline * (1.0 - decline_threshold):
            store.mark_qualified(conn, event_id=int(row["id"]), qualification_date=as_of)
            counts["qualified"] += 1
        else:
            # The seasoned-date close is historical fact — the test can never
            # flip later, so a miss is terminal rather than re-polled forever.
            store.mark_expired(
                conn, event_id=int(row["id"]), expiry_reason=EXPIRY_DECLINE_NOT_MET
            )
            counts["expired_not_met"] += 1
    return counts
