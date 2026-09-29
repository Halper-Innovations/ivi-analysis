from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

from app.catalyst.context import catalyst_context_for_ticker
from app.catalyst.overlay import compute_catalyst_signal
from app.config import get_config
from app.db import connect as db_connect
from app.market.price_provider import PriceSnapshot, build_price_provider
from app.watchlist.anchor_sanity import evaluate_anchor
from app.watchlist.schema import resolve_db_path
from app.watchlist.contract import WatchlistEntry, is_price_trigger_eligible
from app.watchlist.store import (
    add_price_snapshot,
    get_latest,
    list_active,
    record_trigger_status_change,
)


PRICE_DATA_SUSPECT = "PRICE_DATA_SUSPECT"
QUARANTINE = "QUARANTINE"
PRICE_SANITY_LOW_MULTIPLE = 0.25
PRICE_SANITY_HIGH_MULTIPLE = 4.0
# Retired 2026-09-02: the price-sanity reference no longer switches to the
# valuation anchor. Kept as a named constant only so the removed threshold is
# findable from the reason strings written before that date.
PRICE_SANITY_ANCHOR_SCALE_MULTIPLE = 4.0
PRICE_SANITY_HISTORY_YEARS = 5

# Price integrity at trigger time. A snapshot whose as-of date is more
# than this many calendar days behind the evaluation date is PRICE_STALE:
# trigger evaluation refuses to act on it (no transition, no snapshot row),
# and an already-DEPLOY_READY row demotes to PRICE_DATA_SUSPECT rather than
# keep quoting a phantom trigger. 5 calendar days tolerates a long weekend +
# holiday; override with VOE_PRICE_TRIGGER_MAX_AGE_DAYS.
PRICE_TRIGGER_MAX_AGE_DAYS_DEFAULT = 5
# A fresh at-target crossing must persist across two consecutive reprice
# heartbeats before the status flips (illiquid micro-cap prints make a
# single-heartbeat flip too twitchy; two-source checks are impractical in
# this band). The confirming prior snapshot must itself be recent.
AT_TARGET_CONFIRMATION_MAX_AGE_DAYS = 7
AT_TARGET_STATUSES = {"DEPLOY_READY", "BUY_CONFIRMED"}


def _price_trigger_max_age_days() -> int:
    import os

    raw = os.getenv("VOE_PRICE_TRIGGER_MAX_AGE_DAYS", "")
    try:
        return int(raw) if raw.strip() else PRICE_TRIGGER_MAX_AGE_DAYS_DEFAULT
    except ValueError:
        return PRICE_TRIGGER_MAX_AGE_DAYS_DEFAULT


def _snapshot_age_days(snapshot: PriceSnapshot, *, reference: date) -> int | None:
    """Calendar days between the snapshot's as-of date and the reference."""
    try:
        snap_date = date.fromisoformat(str(snapshot.as_of_date or "").strip()[:10])
    except ValueError:
        return None
    return (reference - snap_date).days


def _previous_price_snapshot(
    watchlist_id: int, *, db_path: str | Path | None = None
) -> dict[str, Any] | None:
    """Latest persisted price snapshot for the row (before this check writes)."""
    path = resolve_db_path(db_path)
    if not path.exists():
        return None
    try:
        with db_connect(path) as conn:
            row = conn.execute(
                """
                SELECT price, checked_at, source
                FROM watchlist_price_snapshots
                WHERE watchlist_id = ?
                ORDER BY checked_at DESC, id DESC
                LIMIT 1
                """,
                (int(watchlist_id),),
            ).fetchone()
    except sqlite3.Error:
        return None
    if row is None:
        return None
    return {"price": row[0], "checked_at": row[1], "source": row[2]}


def _iso_age_days(value: str | None, *, reference: datetime) -> float | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return (reference - parsed).total_seconds() / 86400.0


TRIGGER_ELIGIBLE_STATUSES = {
    "ACTIVE",
    "DEPLOY_READY",
    "BUY_CONFIRMED",
    "UNCERTAIN",
    PRICE_DATA_SUSPECT,
}
# QUARANTINE is force-checkable but NOT trigger-eligible: a quarantined anchor is
# not a price-checkable buy candidate, yet a forced recheck must be able to
# RE-EVALUATE and lift the quarantine once the underlying data is repaired.
FORCE_CHECKABLE_STATUSES = TRIGGER_ELIGIBLE_STATUSES | {"CONTRADICTED", "RESOLVED", QUARANTINE}


@dataclass(frozen=True)
class WatchlistTriggerResult:
    ticker: str
    prior_status: str
    new_status: str
    latest_price: float | None
    buy_price_target: float | None
    transition: str
    checked_at: str | None = None
    source: str | None = None
    warning: str | None = None
    dry_run: bool = False
    mutated: bool = False
    price_age_days: int | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _price_source(snapshot: PriceSnapshot) -> str:
    return str(snapshot.source or "").strip() or "unknown"


def _fetch_latest_price(ticker: str, *, as_of_date: str | None = None) -> PriceSnapshot | None:
    cfg = get_config()
    provider = build_price_provider(
        cfg=cfg,
        with_prices=True,
        fallback_days=getattr(cfg, "price_fallback_days", None),
    )
    return provider.get_price_asof(ticker.upper(), as_of_date or date.today().isoformat())


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()}


def _fetch_catalyst_label(
    ticker: str,
    *,
    db_path: str | Path | None = None,
    as_of_date: str | None = None,
    persist: bool = True,
) -> str:
    """Default catalyst lookup: build the context and score it into a label.

    Net/parse-disabled or missing cache degrades to WEAK/NONE via the overlay's
    own rules; it never returns CONFIRMED without confirmed open-market buys.

    ``persist`` is threaded to ``catalyst_context_for_ticker`` so a dry-run
    trigger check (``persist=False``) does not memoize catalyst_events rows.
    """
    path = resolve_db_path(db_path)
    if not path.exists():
        return "NONE"
    try:
        with db_connect(path) as conn:
            ctx = catalyst_context_for_ticker(
                conn,
                ticker.upper(),
                as_of_date or date.today().isoformat(),
                persist=persist,
            )
    except sqlite3.Error:
        return "NONE"
    return compute_catalyst_signal(ctx).label


def _companyfacts_fiscal_year_end_dates(
    ticker: str,
    *,
    db_path: str | Path | None = None,
    as_of_date: str | None = None,
    limit: int = PRICE_SANITY_HISTORY_YEARS,
) -> list[str]:
    path = resolve_db_path(db_path)
    if not path.exists():
        return []
    raw_as_of = str(as_of_date or date.today().isoformat()).strip()
    try:
        effective_as_of = date.fromisoformat(raw_as_of[:10]).isoformat()
    except ValueError:
        return []
    try:
        with db_connect(path) as conn:
            required = {
                "ticker",
                "fiscal_year",
                "period_type",
                "period_end",
                "filed_date",
                "source_url",
            }
            if not required <= _table_columns(conn, "companyfacts_facts"):
                return []
            params: list[Any] = [str(ticker).upper()]
            clauses = [
                "ticker = ?",
                "period_type = 'FY'",
                "date(period_end) IS NOT NULL",
                "date(filed_date) IS NOT NULL",
                "date(period_end) <= date(filed_date)",
                "source_url IS NOT NULL",
                "TRIM(source_url) != ''",
                "date(period_end) <= date(?)",
                "date(filed_date) <= date(?)",
            ]
            params.extend((effective_as_of, effective_as_of))
            rows = conn.execute(
                f"""
                SELECT fiscal_year, MAX(date(period_end)) AS period_end
                FROM companyfacts_facts
                WHERE {" AND ".join(clauses)}
                GROUP BY fiscal_year
                ORDER BY fiscal_year DESC
                LIMIT ?
                """,
                (*params, max(1, int(limit))),
            ).fetchall()
    except sqlite3.Error:
        return []
    return [str(row[1]) for row in rows if str(row[1] or "").strip()]


def _historical_fiscal_year_median_price(
    ticker: str,
    *,
    db_path: str | Path | None = None,
    as_of_date: str | None = None,
    price_lookup_asof: Callable[[str, str], PriceSnapshot | None] | None = None,
) -> float | None:
    period_ends = _companyfacts_fiscal_year_end_dates(
        ticker,
        db_path=db_path,
        as_of_date=as_of_date,
        limit=PRICE_SANITY_HISTORY_YEARS,
    )
    if not period_ends:
        return None
    provider = (
        None
        if price_lookup_asof is not None
        else build_price_provider(
            cfg=get_config(),
            with_prices=True,
            fallback_days=getattr(get_config(), "price_fallback_days", None),
            max_retries=1,
        )
    )
    prices: list[float] = []
    for period_end in period_ends:
        try:
            snapshot = (
                price_lookup_asof(ticker.upper(), period_end)
                if price_lookup_asof is not None
                else provider.get_price_asof(ticker.upper(), period_end)  # type: ignore[union-attr]
            )
        except Exception:
            snapshot = None
        price = getattr(snapshot, "price", None)
        if isinstance(price, (int, float)) and not isinstance(price, bool) and float(price) > 0:
            prices.append(float(price))
    return float(median(prices)) if prices else None


def _price_sanity_reference(
    *,
    historical_median_price: float | None,
    valuation_anchor_value: float | None,
) -> tuple[float | None, str | None]:
    # The reference is the traded price history, always. It used to switch to
    # the valuation ANCHOR whenever the anchor was more than 4x the median —
    # inside the [0.2x, 5.0x] band where anchor_sanity has already accepted the
    # anchor — and then called any quote below a quarter of the anchor suspect.
    # A quote sitting exactly ON the multi-year median was written
    # PRICE_DATA_SUSPECT, the row dropped out of the default review queue with a
    # reason blaming the quote, and every heartbeat re-flagged it. Those are the
    # deepest-discount names, which is the population this system exists to
    # find. A units artefact in the anchor is the anchor band's job, not this
    # check's.
    if historical_median_price is None or historical_median_price <= 0:
        return None, None
    return float(historical_median_price), "historical_fiscal_year_median"


def _price_sanity_warning(
    *,
    current_price: float,
    historical_median_price: float | None,
    valuation_anchor_value: float | None,
) -> str | None:
    reference_price, reference_source = _price_sanity_reference(
        historical_median_price=historical_median_price,
        valuation_anchor_value=valuation_anchor_value,
    )
    if reference_price is None or reference_source is None:
        return None
    if current_price < PRICE_SANITY_LOW_MULTIPLE * reference_price:
        return (
            f"{PRICE_DATA_SUSPECT}:latest_price={current_price:.2f}:"
            f"{reference_source}={reference_price:.2f}:historical_median={historical_median_price:.2f}:"
            f"threshold=below_{PRICE_SANITY_LOW_MULTIPLE:.2f}x"
        )
    if current_price > PRICE_SANITY_HIGH_MULTIPLE * reference_price:
        return (
            f"{PRICE_DATA_SUSPECT}:latest_price={current_price:.2f}:"
            f"{reference_source}={reference_price:.2f}:historical_median={historical_median_price:.2f}:"
            f"threshold=above_{PRICE_SANITY_HIGH_MULTIPLE:.2f}x"
        )
    return None


def _new_status_for_price(
    price: float,
    buy_price_target: float,
    catalyst_label: str | None = None,
) -> str:
    # The price gate is necessary: only a within-reach (price <= target) name can
    # move off ACTIVE. A CONFIRMED catalyst is the orthogonal timing axis that
    # upgrades DEPLOY_READY -> BUY_CONFIRMED; WEAK/NONE/None stay DEPLOY_READY.
    if price > buy_price_target:
        return "ACTIVE"
    if catalyst_label == "CONFIRMED":
        return "BUY_CONFIRMED"
    return "DEPLOY_READY"


def _transition_label(prior_status: str, new_status: str) -> str:
    if prior_status == new_status:
        return "none"
    if new_status == "BUY_CONFIRMED":
        return "catalyst-confirmed below buy-target"
    if new_status == "DEPLOY_READY":
        return "crossed below buy-target"
    if prior_status in {"DEPLOY_READY", "BUY_CONFIRMED"} and new_status == "ACTIVE":
        return "rose above buy-target"
    return "price above buy-target"


def _status_reason(*, transition: str, price: float, buy_price_target: float) -> str:
    if transition == "catalyst-confirmed below buy-target":
        return (
            f"catalyst-confirmed below buy-target: "
            f"price ${price:.2f} <= buy-target ${buy_price_target:.2f}"
        )
    if transition == "rose above buy-target":
        direction = "rose above"
    elif transition == "price above buy-target":
        direction = "is above"
    else:
        direction = "crossed below"
    return f"price ${price:.2f} {direction} buy-target ${buy_price_target:.2f}"


def check_entry_trigger(
    entry: WatchlistEntry,
    *,
    dry_run: bool = False,
    force: bool = False,
    db_path: str | Path | None = None,
    price_lookup: Callable[[str], PriceSnapshot | None] | None = None,
    historical_median_lookup: Callable[[str], float | None] | None = None,
    catalyst_lookup: Callable[[str], str] | None = None,
    checked_at: str | None = None,
) -> WatchlistTriggerResult:
    normalized = entry.normalized()
    if normalized.status == "REMOVED" or (
        normalized.status not in TRIGGER_ELIGIBLE_STATUSES and not force
    ):
        return WatchlistTriggerResult(
            ticker=normalized.ticker,
            prior_status=normalized.status,
            new_status=normalized.status,
            latest_price=None,
            buy_price_target=normalized.buy_price_target,
            transition="skipped",
            checked_at=checked_at,
            warning=f"STATUS_SKIPPED:{normalized.status}",
            dry_run=dry_run,
            mutated=False,
        )
    if force and normalized.status not in FORCE_CHECKABLE_STATUSES:
        return WatchlistTriggerResult(
            ticker=normalized.ticker,
            prior_status=normalized.status,
            new_status=normalized.status,
            latest_price=None,
            buy_price_target=normalized.buy_price_target,
            transition="skipped",
            checked_at=checked_at,
            warning=f"STATUS_SKIPPED:{normalized.status}",
            dry_run=dry_run,
            mutated=False,
        )
    price_trigger_eligible = is_price_trigger_eligible(normalized)
    if not price_trigger_eligible:
        baseline_status = (
            "UNCERTAIN"
            if str(normalized.conviction_grade or "").upper() == "DATA_INCOMPLETE"
            else "ACTIVE"
        )
        reason = (
            "V2_INVESTMENT_GATE_BLOCKED: requires validated underwriting "
            "(UNDERWRITTEN + VALIDATED_UNDERWRITING + VALIDATED + ACTIONABLE)"
        )
        mutated = False
        needs_demotion = normalized.status in AT_TARGET_STATUSES
        if needs_demotion and not dry_run:
            record_trigger_status_change(
                normalized.ticker,
                status=baseline_status,
                reason=reason,
                db_path=db_path,
            )
            mutated = True
        return WatchlistTriggerResult(
            ticker=normalized.ticker,
            prior_status=normalized.status,
            new_status=baseline_status if needs_demotion else normalized.status,
            latest_price=None,
            buy_price_target=None,
            transition=(
                "v2-investment-gate-demotion" if needs_demotion else "v2-investment-gate-refused"
            ),
            checked_at=checked_at,
            warning=reason,
            dry_run=dry_run,
            mutated=mutated,
        )
    if normalized.buy_price_target is None or normalized.buy_price_target <= 0:
        return WatchlistTriggerResult(
            ticker=normalized.ticker,
            prior_status=normalized.status,
            new_status=normalized.status,
            latest_price=None,
            buy_price_target=normalized.buy_price_target,
            transition="skipped",
            checked_at=checked_at,
            warning="BUY_PRICE_TARGET_MISSING",
            dry_run=dry_run,
            mutated=False,
        )
    lookup = price_lookup or (lambda ticker: _fetch_latest_price(ticker))
    snapshot = lookup(normalized.ticker)
    if snapshot is None or snapshot.price <= 0:
        return WatchlistTriggerResult(
            ticker=normalized.ticker,
            prior_status=normalized.status,
            new_status=normalized.status,
            latest_price=None,
            buy_price_target=normalized.buy_price_target,
            transition="skipped",
            checked_at=checked_at,
            warning="PRICE_UNAVAILABLE",
            dry_run=dry_run,
            mutated=False,
        )

    # An over-age snapshot is PRICE_STALE — never act on it. No price
    # snapshot row is written (a stale price must not poison the two-heartbeat
    # confirmation or render as "latest"), and a row already presenting
    # at-target demotes to PRICE_DATA_SUSPECT instead of quoting a phantom.
    evaluation_date = date.fromisoformat(str(checked_at)[:10]) if checked_at else date.today()
    price_age_days = _snapshot_age_days(snapshot, reference=evaluation_date)
    max_age_days = _price_trigger_max_age_days()
    if price_age_days is not None and price_age_days > max_age_days:
        stale_reason = (
            f"PRICE_STALE:asof={snapshot.as_of_date}:age_days={price_age_days}:"
            f"ceiling={max_age_days}:source={_price_source(snapshot)}"
        )
        demote = normalized.status in AT_TARGET_STATUSES
        mutated = False
        if demote and not dry_run:
            record_trigger_status_change(
                normalized.ticker,
                status=PRICE_DATA_SUSPECT,
                reason=stale_reason,
                db_path=db_path,
            )
            mutated = True
        return WatchlistTriggerResult(
            ticker=normalized.ticker,
            prior_status=normalized.status,
            new_status=PRICE_DATA_SUSPECT if demote else normalized.status,
            latest_price=float(snapshot.price),
            buy_price_target=normalized.buy_price_target,
            transition="price-stale-demotion" if demote else "skipped",
            checked_at=checked_at or _utc_now_iso(),
            source=_price_source(snapshot),
            warning=stale_reason,
            dry_run=dry_run,
            mutated=mutated,
            price_age_days=price_age_days,
        )

    latest_price = float(snapshot.price)
    buy_price_target = float(normalized.buy_price_target)
    historical_median_price = (
        historical_median_lookup(normalized.ticker)
        if historical_median_lookup is not None
        else _historical_fiscal_year_median_price(
            normalized.ticker,
            db_path=db_path,
            as_of_date=snapshot.as_of_date,
        )
    )
    effective_checked_at = checked_at or _utc_now_iso()
    source = _price_source(snapshot)

    # Anchor-sanity quarantine takes precedence over the price-sanity check so an
    # out-of-band per-share anchor is QUARANTINEd rather than mislabeled
    # PRICE_DATA_SUSPECT (which would wrongly blame the latest price). When the
    # anchor is in-band against the trailing median, fall through to the normal
    # price logic so a previously-QUARANTINEd row lifts once its data is repaired.
    # When the trailing FY-median reference is missing, fall back to the
    # latest price (mirroring the add-time gate in store._entry_from_candidate)
    # so an out-of-band per-share anchor is still QUARANTINEd rather than silently
    # passed to DEPLOY_READY. evaluate_anchor against None always returns
    # UNASSESSABLE, which previously skipped the quarantine block entirely.
    reference_for_anchor = (
        historical_median_price if historical_median_price is not None else latest_price
    )
    anchor_sanity = evaluate_anchor(
        normalized.valuation_anchor_value,
        reference_for_anchor,
    )
    if anchor_sanity.verdict.startswith("QUARANTINE"):
        ref_str = (
            f"{float(reference_for_anchor):.2f}" if reference_for_anchor is not None else "n/a"
        )
        quarantine_reason = (
            f"{anchor_sanity.verdict}:"
            f"anchor={float(normalized.valuation_anchor_value):.2f}:"
            f"ref={ref_str}:"
            f"ratio={anchor_sanity.ratio}"
        )
        mutated = False
        if not dry_run:
            record_trigger_status_change(
                normalized.ticker,
                status=QUARANTINE,
                reason=quarantine_reason,
                db_path=db_path,
            )
            mutated = True
        return WatchlistTriggerResult(
            ticker=normalized.ticker,
            prior_status=normalized.status,
            new_status=QUARANTINE,
            latest_price=latest_price,
            buy_price_target=buy_price_target,
            transition="anchor-quarantine",
            checked_at=effective_checked_at,
            source=source,
            warning=quarantine_reason,
            dry_run=dry_run,
            mutated=mutated,
        )
    if anchor_sanity.verdict == "UNASSESSABLE_NO_REFERENCE":
        # A missing/non-positive anchor (no usable reference at all) must
        # never be silently promoted to a buy status. Surface it explicitly and
        # skip the DEPLOY_READY/BUY_CONFIRMED promotion (never auto-DEPLOY_READY
        # an unassessable anchor).
        anchor_value = normalized.valuation_anchor_value
        anchor_str = f"{float(anchor_value):.2f}" if anchor_value is not None else "n/a"
        unassessable_reason = f"UNASSESSABLE_NO_REFERENCE:anchor={anchor_str}"
        mutated = False
        if not dry_run:
            record_trigger_status_change(
                normalized.ticker,
                status=PRICE_DATA_SUSPECT,
                reason=unassessable_reason,
                db_path=db_path,
            )
            mutated = True
        return WatchlistTriggerResult(
            ticker=normalized.ticker,
            prior_status=normalized.status,
            new_status=PRICE_DATA_SUSPECT,
            latest_price=latest_price,
            buy_price_target=buy_price_target,
            transition="unassessable-anchor",
            checked_at=effective_checked_at,
            source=source,
            warning=unassessable_reason,
            dry_run=dry_run,
            mutated=mutated,
        )

    sanity_warning = _price_sanity_warning(
        current_price=latest_price,
        historical_median_price=historical_median_price,
        valuation_anchor_value=normalized.valuation_anchor_value,
    )
    if sanity_warning:
        mutated = False
        if not dry_run:
            record_trigger_status_change(
                normalized.ticker,
                status=PRICE_DATA_SUSPECT,
                reason=sanity_warning,
                db_path=db_path,
            )
            mutated = True
        return WatchlistTriggerResult(
            ticker=normalized.ticker,
            prior_status=normalized.status,
            new_status=PRICE_DATA_SUSPECT,
            latest_price=latest_price,
            buy_price_target=buy_price_target,
            transition="price-data-suspect",
            checked_at=effective_checked_at,
            source=source,
            warning=PRICE_DATA_SUSPECT,
            dry_run=dry_run,
            mutated=mutated,
        )

    # The catalyst is an orthogonal timing axis that only matters once a name is
    # within reach (price <= target), so resolve the (potentially Form-4-fetching)
    # catalyst label ONLY then -- never for above-target names.
    catalyst_label: str | None = None
    if price_trigger_eligible and latest_price <= buy_price_target:
        # A dry_run check must be side-effect-free, so the default lookup
        # must NOT memoize parsed Form-4 / buyback rows into catalyst_events.
        lookup_catalyst = catalyst_lookup or (
            lambda ticker: _fetch_catalyst_label(
                ticker,
                db_path=db_path,
                as_of_date=snapshot.as_of_date,
                persist=not dry_run,
            )
        )
        catalyst_label = lookup_catalyst(normalized.ticker)

    if price_trigger_eligible:
        new_status = _new_status_for_price(latest_price, buy_price_target, catalyst_label)
        warning: str | None = None
    else:
        new_status = (
            "UNCERTAIN"
            if str(normalized.conviction_grade or "").upper() == "DATA_INCOMPLETE"
            else "ACTIVE"
        )
        warning = (
            "V2_INVESTMENT_GATE_BLOCKED:price refreshed but at-target "
            "promotion requires validated underwriting"
        )

    # A FRESH at-target crossing must persist across two consecutive
    # reprice heartbeats before the status flips. The confirmation memory is
    # the prior persisted price snapshot: this check's snapshot is always
    # written, so if the name is still at target on the next heartbeat, the
    # flip proceeds. Rows already presenting at-target (DEPLOY_READY ->
    # BUY_CONFIRMED catalyst upgrades) are not fresh crossings.
    if (
        price_trigger_eligible
        and new_status in AT_TARGET_STATUSES
        and normalized.status not in AT_TARGET_STATUSES
    ):
        prior_snapshot = (
            _previous_price_snapshot(int(normalized.id), db_path=db_path)
            if normalized.id is not None
            else None
        )
        prior_price = prior_snapshot.get("price") if prior_snapshot else None
        prior_age = (
            _iso_age_days(
                str(prior_snapshot.get("checked_at") or ""),
                reference=datetime.fromisoformat(effective_checked_at.replace("Z", "+00:00")),
            )
            if prior_snapshot
            else None
        )
        confirmed = (
            isinstance(prior_price, (int, float))
            and float(prior_price) <= buy_price_target
            and prior_age is not None
            and 0 <= prior_age <= AT_TARGET_CONFIRMATION_MAX_AGE_DAYS
        )
        if not confirmed:
            warning = (
                f"AT_TARGET_PENDING_CONFIRMATION:price={latest_price:.2f}:"
                f"target={buy_price_target:.2f}:confirms_next_heartbeat"
            )
            new_status = normalized.status

    transition = _transition_label(normalized.status, new_status)
    if warning is not None and price_trigger_eligible:
        transition = "at-target-pending-confirmation"

    mutated = False
    if not dry_run:
        if normalized.id is None:
            raise ValueError(
                f"Watchlist entry {normalized.ticker} must be persisted before trigger checks"
            )
        add_price_snapshot(
            int(normalized.id),
            price=latest_price,
            checked_at=effective_checked_at,
            source=source,
            db_path=db_path,
        )
        # Refresh the row's dollar-ADV/capacity fields on every reprice
        # pass (cheap aggregates over price_quotes; ADV_UNKNOWN when the
        # volume history is too thin). Best-effort — never blocks the check.
        try:
            from app.market.adv import persist_watchlist_adv

            persist_watchlist_adv(
                normalized.ticker,
                as_of_date=snapshot.as_of_date,
                db_path=db_path,
            )
        except Exception:  # noqa: BLE001 - liquidity annotation is advisory
            pass
        mutated = True
        if new_status != normalized.status:
            record_trigger_status_change(
                normalized.ticker,
                status=new_status,
                reason=_status_reason(
                    transition=transition,
                    price=latest_price,
                    buy_price_target=buy_price_target,
                ),
                db_path=db_path,
            )

    return WatchlistTriggerResult(
        ticker=normalized.ticker,
        prior_status=normalized.status,
        new_status=new_status,
        latest_price=latest_price,
        buy_price_target=buy_price_target,
        transition=transition,
        checked_at=effective_checked_at,
        source=source,
        warning=warning,
        dry_run=dry_run,
        mutated=mutated,
        price_age_days=price_age_days,
    )


def check_watchlist_triggers(
    ticker: str | None = None,
    *,
    dry_run: bool = False,
    force: bool = False,
    db_path: str | Path | None = None,
) -> list[WatchlistTriggerResult]:
    if ticker:
        entry = get_latest(ticker, db_path=db_path)
        if entry is None:
            return [
                WatchlistTriggerResult(
                    ticker=ticker.upper(),
                    prior_status="n/a",
                    new_status="n/a",
                    latest_price=None,
                    buy_price_target=None,
                    transition="skipped",
                    warning="ENTRY_NOT_FOUND",
                    dry_run=dry_run,
                    mutated=False,
                )
            ]
        return [check_entry_trigger(entry, dry_run=dry_run, force=force, db_path=db_path)]

    entries = [
        entry
        for entry in list_active(db_path=db_path)
        if entry.status.upper() in TRIGGER_ELIGIBLE_STATUSES
    ]
    return [
        check_entry_trigger(entry, dry_run=dry_run, force=False, db_path=db_path)
        for entry in entries
    ]
