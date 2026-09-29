"""Catalyst context builder wiring Form-4 + buyback.

Assembles the catalyst context dict consumed by
``app.catalyst.overlay.compute_catalyst_signal``. It mirrors the I/O boundary of
``app/score/ranker.py:_change_context`` (a ``conn``-first reader that returns a
plain context dict) so the catalyst overlay stays consistent with the rest of
the deterministic scoring layer.

The two v1 EDGAR-derived catalysts are merged here:
  * Form-4 OPEN-MARKET insider-buy CLUSTERS (>= 2 distinct open-market buyers
    within a 90-day lookback) -- the primary signal.
  * Buyback acceleration -- a secondary, WEAK-only signal.

Distinct-buyer counting dedupes by reporting-owner identity (CIK preferred, else
name) parsed from the Form-4 XML. When the network/parse is disabled the
transaction codes cannot be confirmed: ``insider_open_market_purchase_count``
stays 0, ``codes_unverified`` is True, and the overlay can only reach WEAK -- it
never reports CONFIRMED without confirmed open-market buys.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import Any

from app.catalyst.buyback import detect_buyback_signal
from app.catalyst.form4 import enrich_events, read_form4_stubs
from app.catalyst.overlay import compute_catalyst_signal
from app.db import upsert_catalyst_event


def catalyst_context_for_ticker(
    conn,
    ticker: str,
    as_of_date: str | date,
    lookback_days: int = 90,
    *,
    fetch: Callable[[str], bytes] | None = None,
    persist: bool = True,
) -> dict[str, Any]:
    """Build the catalyst context dict for ``ticker`` as of ``as_of_date``.

    Reads in-window Form-4 stubs, enriches their transaction codes when
    the network is enabled, counts distinct reporting owners with a
    confirmed open-market purchase, and merges the buyback-acceleration signal.
    The returned dict is the input to ``compute_catalyst_signal``.

    Keys returned:
      * ``insider_open_market_purchase_count`` -- confirmed P-coded Form-4
        open-market purchases in the window (0 when codes are unverified).
      * ``insider_distinct_buyers`` -- distinct reporting owners with a confirmed
        open-market purchase; when codes are unverified this counts in-window
        filers instead, but the overlay can only reach WEAK.
      * ``insider_filing_count`` -- count of in-window Form-4 filings.
      * ``codes_unverified`` -- True when filings exist but transaction codes
        could not be confirmed (net/parse disabled).
      * ``buyback_label`` / ``buyback_latest_fy`` / ``buyback_prior_fy`` /
        ``buyback_fiscal_year`` -- the buyback-acceleration signal.
      * ``buyback_status`` / ``buyback_reason_codes`` /
        ``buyback_source_lineage`` -- its exact filed-as-of provenance. A
        ``NEEDS_DATA`` buyback always carries label ``NONE``.
    """
    events = read_form4_stubs(
        ticker=ticker,
        conn=conn,
        as_of_date=as_of_date,
        lookback_days=lookback_days,
    )
    enrich_events(events, fetch=fetch)

    filing_count = len(events)

    # Codes are unverified when in-window filings exist but none carries a
    # confirmed transaction code (net/parse disabled left is_purchase=None).
    codes_unverified = filing_count > 0 and all(e.is_purchase is None for e in events)

    if codes_unverified:
        # Cannot confirm open-market buys; count filers but the overlay stays
        # WEAK because there are zero confirmed open-market purchases.
        open_market_count = 0
        distinct_buyers = filing_count
    else:
        purchases = [e for e in events if e.is_purchase is True]
        open_market_count = len(purchases)
        distinct_owners = {e.reporting_owner for e in purchases if e.reporting_owner}
        # Only purchases with an identified reporting owner count toward the
        # distinct-buyer cluster. Owner-less purchases (malformed / partial XML
        # with no parseable rptOwnerCik/Name) still count toward open_market_count
        # but must NOT manufacture a false CONFIRMED >=2-buyer cluster — which
        # would defeat the dedupe that distinguishes a genuine cluster from
        # unattributable filings.
        distinct_buyers = len(distinct_owners)

    buyback = detect_buyback_signal(conn, ticker, as_of_date)

    ctx = {
        "insider_open_market_purchase_count": open_market_count,
        "insider_distinct_buyers": distinct_buyers,
        "insider_filing_count": filing_count,
        "codes_unverified": codes_unverified,
        "buyback_label": buyback["label"],
        "buyback_latest_fy": buyback["latest_fy"],
        "buyback_prior_fy": buyback["prior_fy"],
        "buyback_fiscal_year": buyback["fiscal_year"],
        "buyback_unit": buyback["unit"],
        "buyback_status": buyback["status"],
        "buyback_reason_codes": buyback["reason_codes"],
        "buyback_source_lineage": buyback["source_lineage"],
    }

    if persist:
        _persist_catalyst_events(conn, ticker, as_of_date, events, ctx, buyback)

    return ctx


def _persist_catalyst_events(
    conn,
    ticker: str,
    as_of_date: str | date,
    events,
    ctx: dict[str, Any],
    buyback: dict[str, Any],
) -> None:
    """Memoize the parsed Form-4 cluster + buyback signal to catalyst_events.

    Mirrors the outcome-persistence boundary: the context builder is the one
    place that fetches/parses form4.xml, so it is where the parsed result is
    cached. The table is idempotent on UNIQUE(ticker, as_of_date, catalyst_type),
    so re-running for the same as-of date overwrites rather than re-fetching, and
    the parsed events stay auditable / available to the outcome loop.
    """
    as_of_str = as_of_date.isoformat() if isinstance(as_of_date, date) else str(as_of_date)

    overall = compute_catalyst_signal(ctx)
    upsert_catalyst_event(
        conn,
        ticker=ticker,
        as_of_date=as_of_str,
        catalyst_type="INSIDER_BUY_CLUSTER",
        signal_label=overall.label,
        score=overall.score,
        detail={
            "insider_open_market_purchase_count": ctx["insider_open_market_purchase_count"],
            "insider_distinct_buyers": ctx["insider_distinct_buyers"],
            "insider_filing_count": ctx["insider_filing_count"],
            "codes_unverified": ctx["codes_unverified"],
            "accessions": [e.accession for e in events],
        },
        source_url=events[0].primary_doc_url if events else None,
    )

    upsert_catalyst_event(
        conn,
        ticker=ticker,
        as_of_date=as_of_str,
        catalyst_type="BUYBACK_ACCELERATION",
        signal_label=buyback["label"],
        score=0.0,
        detail={
            "latest_fy": buyback["latest_fy"],
            "prior_fy": buyback["prior_fy"],
            "fiscal_year": buyback["fiscal_year"],
            "unit": buyback["unit"],
            "status": buyback["status"],
            "reason_codes": buyback["reason_codes"],
            "source_lineage": buyback["source_lineage"],
        },
        source_url=(
            buyback["source_lineage"][0]["source_url"] if buyback["source_lineage"] else None
        ),
    )
