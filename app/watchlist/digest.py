from __future__ import annotations

import base64
import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.autonomous.artifact_financial_audit import (
    DIGEST_DECISION_STATE_FIELDS,
    DIGEST_LINEAGE_SCHEMA_VERSION,
    DIGEST_RENDERED_STATE_SCHEMA_VERSION,
    active_financial_integrity_manifest_path,
    financial_integrity_manifest_is_usable,
)
from app.config import get_config
from app.decision.decision_block import buy_now_imperative_line
from app.ops.data_health import DataHealth, compute_data_health
from app.watchlist.schema import ensure_watchlist_schema, resolve_db_path
from app.watchlist.store import current_watchlist_cte, watchlist_queue
from app.watchlist.triggers import PRICE_DATA_SUSPECT, check_watchlist_triggers
from app.watchlist.contract import is_price_trigger_eligible
from app.watchlist.lineage import (
    authorized_watchlist_decision_binding,
    watchlist_row_is_decision_eligible,
)


NO_WINDOW_ACTIVITY = "(none in this window)"
NO_ACTIONABLE_AT_TARGET = "No ACTIONABLE-grade names at target today."
FINANCIAL_INTEGRITY_BLOCKED = (
    "**BLOCKED: the canonical financial-integrity audit manifest is missing, "
    "malformed, or unreadable. Current decision rows are suppressed.**"
)
_DIGEST_RENDERED_STATE_MARKER = "IVI_DIGEST_RENDERED_STATE_V2:"

# ACTIONABLE rows carry a full conviction memo; the rest are price triggers
# without one and must never outrank a memo-backed name however deep their
# discount runs.
_GRADE_ORDER = {"ACTIONABLE": 0, "WATCHLIST_ONLY": 1, "DATA_INCOMPLETE": 2}

ACTIVE_SUMMARY_STATUSES = [
    "ACTIVE",
    "DEPLOY_READY",
    "UNCERTAIN",
    PRICE_DATA_SUSPECT,
    "CONTRADICTED",
    "RESOLVED",
    "REMOVED",
]


@dataclass(frozen=True)
class DigestWriteResult:
    markdown: str
    path: Path
    trigger_checked: bool = False
    lineage_path: Path | None = None


@dataclass(frozen=True)
class _DigestSnapshot:
    watchlist_rows: list[dict[str, Any]]
    latest_prices: dict[int, dict[str, Any]]
    history_rows: list[dict[str, Any]]
    review_rows: list[dict[str, Any]]
    cheapness: dict[str, Any]
    open_dispositions: list[dict[str, Any]]
    held_exit_rows: list[dict[str, Any]]
    data_health: DataHealth


@dataclass(frozen=True)
class _DigestManifestDecision:
    path: Path | None
    sha256: str | None
    usable: bool


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    from app.db import connect

    ensure_watchlist_schema(db_path)
    return connect(resolve_db_path(db_path))


def _capture_manifest_decision() -> _DigestManifestDecision:
    """Capture one immutable manifest choice for a complete render/write call."""

    path = active_financial_integrity_manifest_path()
    if path is None:
        return _DigestManifestDecision(path=None, sha256=None, usable=False)
    try:
        before = path.read_bytes()
    except OSError:
        return _DigestManifestDecision(path=path, sha256=None, usable=False)
    sha256 = hashlib.sha256(before).hexdigest()
    usable = financial_integrity_manifest_is_usable(path)
    try:
        unchanged = hashlib.sha256(path.read_bytes()).hexdigest() == sha256
    except OSError:
        unchanged = False
    return _DigestManifestDecision(
        path=path,
        sha256=sha256 if usable and unchanged else None,
        usable=bool(usable and unchanged),
    )


def _manifest_decision_is_current(decision: _DigestManifestDecision) -> bool:
    if not decision.usable or decision.path is None or decision.sha256 is None:
        return False
    active = active_financial_integrity_manifest_path()
    if active is None or active.resolve() != decision.path.resolve():
        return False
    try:
        return hashlib.sha256(decision.path.read_bytes()).hexdigest() == decision.sha256
    except OSError:
        return False


def _blocked_digest(effective_now: datetime) -> str:
    return "\n".join(
        (
            "# IVI Watchlist Daily Digest",
            "",
            f"- Generated: {effective_now.isoformat()}",
            "",
            FINANCIAL_INTEGRITY_BLOCKED,
            "",
        )
    )


def _parse_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _in_window(value: Any, *, cutoff: datetime) -> bool:
    parsed = _parse_datetime(value)
    return parsed is not None and parsed >= cutoff


def _fmt_money(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"${float(value):.2f}"
    return "n/a"


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}%"


def _distance_from_buy(price: float | None, buy_price_target: float | None) -> float | None:
    if price is None or buy_price_target is None or buy_price_target == 0:
        return None
    return ((float(price) - float(buy_price_target)) / float(buy_price_target)) * 100.0


def _grade_rank(row: dict[str, Any]) -> int:
    return _GRADE_ORDER.get(str(row.get("conviction_grade") or "").upper(), 3)


def _band_label(row: dict[str, Any]) -> str:
    """Canonical band token, or the explicit UNKNOWN_CAP label.

    A name without a computable cap must never present as in-band output;
    the label keeps the gap loud on every rendered surface.
    """
    return str(row.get("cap_band") or "UNKNOWN_CAP")


def _latest_price_for_rows(conn: sqlite3.Connection) -> dict[int, dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT id, watchlist_id, price, checked_at, source
        FROM watchlist_price_snapshots
        ORDER BY watchlist_id, checked_at DESC, id DESC
        """
    ).fetchall()
    latest: dict[int, dict[str, Any]] = {}
    for row in rows:
        watchlist_id = int(row["watchlist_id"])
        if watchlist_id not in latest:
            latest[watchlist_id] = dict(row)
    return latest


def _history_rows(
    conn: sqlite3.Connection,
    *,
    manifest_path: Path,
) -> list[dict[str, Any]]:
    rows = conn.execute(
        f"""
        WITH latest_watchlist AS ({current_watchlist_cte()})
        SELECT
            h.*,
            w.ticker,
            w.status,
            w.conviction_grade,
            w.confidence,
            w.conviction_source,
            w.scan_family,
            w.valuation_anchor_method,
            w.valuation_anchor_value,
            w.buy_price_target,
            w.current_price_at_addition,
            w.thesis_text,
            w.key_risks_json,
            w.falsifiers_json,
            w.open_questions_json,
            w.source_sector,
            w.status_reason,
            w.market_cap_mm,
            w.cap_source,
            w.cap_band,
            w.cap_asof,
            w.pipeline_version,
            w.candidate_disposition,
            w.decision_basis,
            w.selection_validation_status,
            w.source_run_id AS watchlist_source_run_id
        FROM watchlist_history h
        JOIN watchlist w ON w.id = h.watchlist_id
        JOIN latest_watchlist ON latest_watchlist.latest_id = w.id
        ORDER BY h.changed_at DESC, h.id DESC
        """
    ).fetchall()
    return [
        dict(row)
        for row in rows
        if watchlist_row_is_decision_eligible(
            row,
            manifest_path,
            source_run_field="watchlist_source_run_id",
        )
    ]


def _watchlist_rows(
    conn: sqlite3.Connection,
    *,
    manifest_path: Path,
) -> list[dict[str, Any]]:
    rows = conn.execute(
        f"""
        WITH latest_watchlist AS ({current_watchlist_cte()})
        SELECT w.*
        FROM watchlist w
        JOIN latest_watchlist ON latest_watchlist.latest_id = w.id
        ORDER BY w.status, w.ticker
        """
    ).fetchall()
    return [dict(row) for row in rows if watchlist_row_is_decision_eligible(row, manifest_path)]


def _append_section(lines: list[str], title: str, rows: list[str]) -> None:
    lines.extend(["", f"## {title}"])
    if rows:
        lines.extend(rows)
    else:
        lines.append(NO_WINDOW_ACTIVITY)


def _render_transition_rows(rows: list[dict[str, Any]], *, new_value: str) -> list[str]:
    matched = [
        row
        for row in rows
        if row["field_name"] == "status"
        and row["new_value"] == new_value
        and str(row.get("status") or "").upper() != "REMOVED"
        and (new_value != "DEPLOY_READY" or is_price_trigger_eligible(row))
    ]
    if not matched:
        return []
    lines = [
        "| Ticker | Changed At | Prior Status | New Status | Conviction | Confidence | Conviction Source | Reason |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in matched:
        lines.append(
            f"| {row['ticker']} | {row['changed_at']} | {row['old_value'] or 'n/a'} | "
            f"{row['new_value']} | {row['conviction_grade'] or 'n/a'} | {row['confidence'] or 'n/a'} | "
            f"{row['conviction_source'] or 'n/a'} | {row['status_reason'] or 'n/a'} |"
        )
    return lines


def _render_new_entries(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    lines = [
        "| Ticker | Status | Conviction | Confidence | Conviction Source | Buy Target | Band | Source Sector | Added |",
        "| --- | --- | --- | --- | --- | ---: | --- | --- | --- |",
    ]
    for row in rows:
        price_eligible = is_price_trigger_eligible(row)
        lines.append(
            f"| {row['ticker']} | {row['status']} | {row['conviction_grade'] or 'n/a'} | "
            f"{row['confidence'] or 'n/a'} | {row['conviction_source'] or 'n/a'} | "
            f"{_fmt_money(row['buy_price_target'] if price_eligible else None)} | "
            f"{_band_label(row)} | "
            f"{row['source_sector'] or 'n/a'} | {row['added_at']} |"
        )
    return lines


def _render_review_today(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    lines = [
        "| Ticker | Status | Conviction | Confidence | Conviction Source | Latest/Add Price | Buy Target | Distance From Buy | Band | Events | Source Sector | Reason |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |",
    ]
    for row in rows:
        price_eligible = is_price_trigger_eligible(row)
        lines.append(
            f"| {row['ticker']} | {row.get('presented_status') or row['status'] or 'n/a'} | "
            f"{row['conviction_grade'] or 'n/a'} | "
            f"{row['confidence'] or 'n/a'} | {row['conviction_source'] or 'n/a'} | "
            f"{_fmt_money(row['latest_price'])} | "
            f"{_fmt_money(row['buy_price_target'] if price_eligible else None)} | "
            f"{_fmt_pct(row['distance_from_buy_pct'] if price_eligible else None)} | "
            f"{_band_label(row)} | "
            f"{row.get('event_pending') or '—'} | "
            f"{row['source_sector'] or 'n/a'} | "
            f"{str(row['status_reason'] or 'n/a').replace(chr(10), ' ')} |"
        )
    return lines


def _render_held_book_exits(rows: list[dict[str, Any]]) -> list[str]:
    """Open exit signals for the held book — capital at risk leads."""
    if not rows:
        return []
    lines = [
        "| Ticker | Signal | As Of | Evidence |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        evidence = row.get("evidence") or {}
        evidence_text = (
            ", ".join(f"{key}={value}" for key, value in list(evidence.items())[:4]) or "n/a"
        )
        lines.append(
            f"| {row['ticker']} | {row['signal_type']} | {row['as_of_date']} | {evidence_text} |"
        )
    return lines


def _render_open_dispositions(
    db_path: str | Path | None,
    *,
    now: datetime,
    rows: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Open at-target dispositions render until the operator closes them —
    an at-target surfacing without a recorded decision is unfinished work."""
    if rows is None:
        try:
            from app.watchlist.dispositions import open_dispositions

            rows = open_dispositions(db_path)
        except Exception:  # noqa: BLE001 - digest renders even if ledger is unreachable
            return []
    # A disposition is decision-bearing output.  Render only the disposition
    # attached to the normally selected current watchlist row after that row's
    # exact source run passes authorization.  Stale, manual, and integrity-
    # blocked rows remain in the ledger for explicit closure but cannot leak
    # an unbound ticker claim into a current digest.
    rows = [
        row
        for row in rows
        if row.get("source_state") == "CURRENT"
        and row.get("financial_integrity_eligible") is True
        and str(row.get("current_source_run_id") or "").strip()
    ]
    if not rows:
        return []
    lines = [
        "Every at-target surfacing must terminate in a recorded decision — "
        "close with `ivi investor journal <TICKER> --action acted|passed|deferred ...`.",
        "",
        "| Ticker | Kind | Source State | Opened | Age (d) | Trigger Snapshot | Resolution |",
        "| --- | --- | --- | --- | ---: | --- | --- |",
    ]
    for row in rows:
        opened = _parse_datetime(row.get("opened_at"))
        age = f"{(now - opened).days}" if opened is not None else "n/a"
        snapshot = row.get("trigger_snapshot") or {}
        snap_text = (
            ", ".join(f"{key}={value}" for key, value in snapshot.items() if value is not None)
            or "n/a"
        )
        stale = row.get("source_state") == "SUPERSEDED_STALE_SOURCE"
        resolution = (
            f"Superseded/stale-source; explicit PASSED or DEFERRED required: "
            f"`ivi investor journal {row['ticker']} --disposition-id {row['id']} "
            "--action passed|deferred ...`"
            if stale
            else f"`ivi investor journal {row['ticker']} --disposition-id {row['id']} ...`"
        )
        lines.append(
            f"| {row['ticker']} | {row['kind']} | {row.get('source_state') or 'n/a'} | "
            f"{str(row['opened_at'])[:10]} | {age} | {snap_text} | {resolution} |"
        )
    return lines


def _render_universe_exits(rows: list[dict[str, Any]]) -> list[str]:
    """Names transitioned to REMOVED in the window (delistings/registry
    exits) — a removal must surface once, not silently vanish from lists."""
    matched = [
        row for row in rows if row["field_name"] == "status" and row["new_value"] == "REMOVED"
    ]
    if not matched:
        return []
    lines = [
        "| Ticker | Changed At | Prior Status | Reason |",
        "| --- | --- | --- | --- |",
    ]
    for row in matched:
        lines.append(
            f"| {row['ticker']} | {row['changed_at']} | {row['old_value'] or 'n/a'} | "
            f"{str(row['status_reason'] or 'n/a').replace(chr(10), ' ')} |"
        )
    return lines


def _render_reevaluations(rows: list[dict[str, Any]]) -> list[str]:
    matched = [
        row
        for row in rows
        if row["field_name"] == "reevaluation" and str(row["new_value"] or "") != "NO_NEW_EVIDENCE"
    ]
    if not matched:
        return []
    lines = [
        "| Ticker | Changed At | Evaluation | Current Status | Reason |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in matched:
        lines.append(
            f"| {row['ticker']} | {row['changed_at']} | {row['new_value'] or 'n/a'} | "
            f"{row['status']} | {row['status_reason'] or 'n/a'} |"
        )
    return lines


def _render_price_quality_issues(rows: list[dict[str, Any]]) -> list[str]:
    matched = [row for row in rows if str(row.get("status") or "").upper() == PRICE_DATA_SUSPECT]
    if not matched:
        return []
    lines = [
        "| Ticker | Status | Buy Target | Source Sector | Reason |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for row in matched:
        price_eligible = is_price_trigger_eligible(row)
        lines.append(
            f"| {row['ticker']} | {row['status']} | "
            f"{_fmt_money(row['buy_price_target'] if price_eligible else None)} | "
            f"{row['source_sector'] or 'n/a'} | {row['status_reason'] or 'n/a'} |"
        )
    return lines


def _render_status_counts(rows: list[dict[str, Any]]) -> list[str]:
    counts = {status: 0 for status in ACTIVE_SUMMARY_STATUSES}
    for row in rows:
        status = str(row["status"] or "UNKNOWN")
        counts[status] = counts.get(status, 0) + 1
    lines = ["| Status | Count |", "| --- | ---: |"]
    for status in sorted(counts):
        count = counts[status]
        if count or status in {
            "ACTIVE",
            "DEPLOY_READY",
            "UNCERTAIN",
            PRICE_DATA_SUSPECT,
            "CONTRADICTED",
        }:
            lines.append(f"| {status} | {count} |")
    return lines


def _latest_price_for_row(
    row: dict[str, Any], latest_prices: dict[int, dict[str, Any]]
) -> tuple[float | None, dict[str, Any] | None]:
    snapshot = latest_prices.get(int(row["id"]))
    if snapshot is not None and isinstance(snapshot.get("price"), (int, float)):
        return float(snapshot["price"]), snapshot
    return row["current_price_at_addition"], snapshot


def _capacity_note(row: dict[str, Any]) -> str:
    """Tradeability annotation for an at-target imperative line.

    A DEPLOY_READY trigger on a name below the dollar-ADV floor renders a
    CAPACITY_LIMITED banner; names without volume history stay explicitly
    ADV_UNKNOWN rather than silently unannotated.
    """
    from app.market.adv import adv_dollar_floor

    adv20 = row.get("adv_dollar_20d")
    capacity = str(row.get("capacity_class") or "ADV_UNKNOWN")
    if not isinstance(adv20, (int, float)):
        return "  (capacity: ADV_UNKNOWN — no volume history; tradeability unassessed)"
    floor = adv_dollar_floor()
    if float(adv20) < floor:
        return (
            f"  (**CAPACITY_LIMITED**: 20d dollar-ADV ${float(adv20):,.0f} is below the "
            f"${floor:,.0f} floor — real size cannot enter or exit at this trigger)"
        )
    return f"  (capacity: {capacity}, 20d dollar-ADV ${float(adv20):,.0f})"


def _price_basis_note(
    row: dict[str, Any],
    snapshot: dict[str, Any] | None,
    *,
    now: datetime | None = None,
) -> str:
    """One-line price provenance for an imperative line.

    Snapshot-backed prices render date + source + age; the
    addition-time-price fallback is labeled explicitly instead of passing
    as a live quote.
    """
    if snapshot is None or not isinstance(snapshot.get("price"), (int, float)):
        added = _parse_datetime(row.get("added_at"))
        age = ""
        if added is not None and now is not None:
            age = f", {int((now - added).total_seconds() // 86400)}d old"
        return (
            "  (price basis: ADDITION-TIME price — no snapshot recorded"
            f"{age}; treat the trigger as unverified)"
        )
    checked_at = str(snapshot.get("checked_at") or "")
    source = str(snapshot.get("source") or "unknown")
    checked = _parse_datetime(checked_at)
    age = ""
    if checked is not None and now is not None:
        age = f", {int((now - checked).total_seconds() // 86400)}d old"
    return f"  (price basis: snapshot {checked_at[:10]} via {source}{age})"


def _cheapness_cell(row: dict[str, Any], cheapness: dict[str, Any] | None) -> str:
    from app.events.cheapness import cheapness_headline

    if cheapness is None:
        return "n/a"
    return cheapness_headline(cheapness.get(str(row.get("ticker") or "").upper()))


def _render_buy_now(
    rows: list[dict[str, Any]],
    latest_prices: dict[int, dict[str, Any]],
    cheapness: dict[str, Any] | None = None,
    *,
    now: datetime | None = None,
) -> list[str]:
    """Render the At Buy Target (Review) section: DEPLOY_READY, non-AVOID rows.

    Sorted conviction grade first (ACTIONABLE > WATCHLIST_ONLY >
    DATA_INCOMPLETE), then distance-through-target ascending (deepest below
    target first), then ticker — a deep discount on a memo-less name must not
    outrank a memo-backed one. Each row carries an explicit at-target review
    line so the names needing attention lead the digest. The deploy flag routes
    attention; it is not a buy signal and the rendered lines must not phrase it
    as one. AVOID-grade rows are never surfaced (grade==AVOID => Pass).
    """

    candidates: list[tuple[float, dict[str, Any], float | None, dict[str, Any] | None]] = []
    for row in rows:
        if str(row["status"]) != "DEPLOY_READY":
            continue
        if not is_price_trigger_eligible(row):
            continue
        if str(row.get("conviction_grade") or "").upper() == "AVOID":
            continue
        if row.get("event_pending"):
            # Open corporate event: blocked from at-target presentation until
            # the analyst pass disposes it (rendered in its own section).
            continue
        latest_price, snapshot = _latest_price_for_row(row, latest_prices)
        distance = _distance_from_buy(latest_price, row["buy_price_target"])
        if distance is None:
            continue
        candidates.append((distance, row, latest_price, snapshot))
    if not candidates:
        return []
    candidates.sort(key=lambda item: (_grade_rank(item[1]), item[0], str(item[1]["ticker"])))
    lines = [
        "| Ticker | Conviction | Latest Price | Buy Target | Distance From Buy | Band | Why Cheap | Source Sector |",
        "| --- | --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    imperatives: list[str] = []
    for distance, row, latest_price, snapshot in candidates:
        lines.append(
            f"| {row['ticker']} | {row['conviction_grade'] or 'n/a'} | "
            f"{_fmt_money(latest_price)} | {_fmt_money(row['buy_price_target'])} | "
            f"{_fmt_pct(distance)} | {_band_label(row)} | "
            f"{_cheapness_cell(row, cheapness)} | {row['source_sector'] or 'n/a'} |"
        )
        imperatives.append(
            "- "
            + buy_now_imperative_line(
                str(row["ticker"]),
                latest_price,
                row["buy_price_target"],
                None,
                row["conviction_grade"],
            )
        )
        # Every imperative line carries its price basis — snapshot date +
        # source, or an explicit addition-time-price label. A months-old
        # addition price must never render indistinguishable from a live quote.
        imperatives.append(f"  - {_price_basis_note(row, snapshot, now=now).strip()}")
        # And its tradeability basis.
        imperatives.append(f"  - {_capacity_note(row).strip()}")
    lines.append("")
    lines.extend(imperatives)
    return lines


def _render_event_blocked(
    rows: list[dict[str, Any]],
    latest_prices: dict[int, dict[str, Any]],
    cheapness: dict[str, Any] | None = None,
) -> list[str]:
    """At-target names carrying an open EVENT_PENDING flag.

    These rows hit their price trigger but a known corporate event (merger
    paper, activist stake, dilution shelf, flagged 8-K) is undisposed — they
    must never render as at-target review candidates until `ivi events
    dispose` records the analyst pass.
    """
    blocked = [
        row
        for row in rows
        if row.get("event_pending")
        and str(row["status"]) in {"DEPLOY_READY", "BUY_CONFIRMED"}
        and is_price_trigger_eligible(row)
        and str(row.get("conviction_grade") or "").upper() != "AVOID"
    ]
    if not blocked:
        return []
    lines = [
        "These names are at/below target but carry an OPEN corporate event — "
        "blocked from at-target presentation until an analyst pass disposes "
        "the event (`ivi events dispose <id>`).",
        "",
        "| Ticker | Events Pending | Conviction | Latest Price | Buy Target | Band | Why Cheap | Source Sector |",
        "| --- | --- | --- | ---: | ---: | --- | --- | --- |",
    ]
    for row in sorted(blocked, key=lambda item: str(item["ticker"])):
        latest_price, _ = _latest_price_for_row(row, latest_prices)
        lines.append(
            f"| {row['ticker']} | {row['event_pending']} | {row['conviction_grade'] or 'n/a'} | "
            f"{_fmt_money(latest_price)} | {_fmt_money(row['buy_price_target'])} | "
            f"{_band_label(row)} | {_cheapness_cell(row, cheapness)} | "
            f"{row['source_sector'] or 'n/a'} |"
        )
    return lines


def _render_buy_confirmed(
    rows: list[dict[str, Any]], latest_prices: dict[int, dict[str, Any]]
) -> list[str]:
    """Render the At Target + Catalyst Confirmed section: within-reach names
    with a confirmed catalyst (status == BUY_CONFIRMED).

    These are the WAIT-vs-BUY split's BUY side: cheap, within-reach, AND carrying a
    real catalyst. The catalyst reason is read from status_reason (written by the
    price trigger); no new query columns are needed.
    """

    candidates = [
        row
        for row in rows
        if str(row["status"]) == "BUY_CONFIRMED"
        and not row.get("event_pending")
        and is_price_trigger_eligible(row)
    ]
    if not candidates:
        return []
    lines = [
        "| Ticker | Latest Price | Buy Target | Catalyst Reason |",
        "| --- | ---: | ---: | --- |",
    ]
    for row in candidates:
        latest_price, _ = _latest_price_for_row(row, latest_prices)
        lines.append(
            f"| {row['ticker']} | {_fmt_money(latest_price)} | "
            f"{_fmt_money(row['buy_price_target'])} | "
            f"{str(row['status_reason'] or 'n/a').replace(chr(10), ' ')} |"
        )
    return lines


def _render_top_active_entries(
    rows: list[dict[str, Any]], latest_prices: dict[int, dict[str, Any]]
) -> list[str]:
    candidates: list[tuple[float, dict[str, Any], float | None, dict[str, Any] | None]] = []
    for row in rows:
        if row["status"] not in {"ACTIVE", "UNCERTAIN"}:
            continue
        if not is_price_trigger_eligible(row):
            continue
        snapshot = latest_prices.get(int(row["id"]))
        latest_price = (
            float(snapshot["price"])
            if snapshot is not None and isinstance(snapshot.get("price"), (int, float))
            else row["current_price_at_addition"]
        )
        distance = _distance_from_buy(latest_price, row["buy_price_target"])
        if distance is None:
            continue
        candidates.append((distance, row, latest_price, snapshot))
    if not candidates:
        return []
    candidates.sort(key=lambda item: (item[0], str(item[1]["ticker"])))
    lines = [
        "| Ticker | Status | Latest Price | Buy Target | Distance From Buy | Source Sector | Price Source |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for distance, row, latest_price, snapshot in candidates[:10]:
        source = str((snapshot or {}).get("source") or "addition_snapshot")
        lines.append(
            f"| {row['ticker']} | {row['status']} | {_fmt_money(latest_price)} | "
            f"{_fmt_money(row['buy_price_target'])} | {_fmt_pct(distance)} | "
            f"{row['source_sector'] or 'n/a'} | {source} |"
        )
    return lines


# Signal labels distinguish an at-target review trigger from a waiting name.
# A price crossing is never rendered as a capital decision.
_SIGNAL_BY_STATUS = {
    "BUY_CONFIRMED": "REVIEW AT TARGET",
    "DEPLOY_READY": "REVIEW AT TARGET",
    "EVENT_PENDING": "EVENT HOLD",
    "ACTIVE": "WAIT",
    "UNCERTAIN": "UNCERTAIN",
    "PRICE_DATA_SUSPECT": "SUSPECT",
    "QUARANTINE": "QUARANTINE",
    "CONTRADICTED": "CONTRADICTED",
    "RESOLVED": "RESOLVED",
}
_SIGNAL_ORDER = {
    "REVIEW AT TARGET": 0,
    "EVENT HOLD": 1,
    "WAIT": 2,
    "UNCERTAIN": 3,
    "SUSPECT": 4,
}


def signal_label(presented_status: str | None) -> str:
    status = str(presented_status or "").upper()
    return _SIGNAL_BY_STATUS.get(status, status or "n/a")


def render_compact_signal_table(queue_rows: list[dict[str, Any]]) -> list[str]:
    """Compact signal board over watchlist_queue() rows.

    One row per name with literal research provenance. Only rows that pass
    ``is_price_trigger_eligible`` expose a target, distance, or buy/wait
    signal; screen-sourced v2 rows remain visibly non-investable research.
    """
    if not queue_rows:
        return []
    lines = [
        "| Signal | Ticker | Conviction | Confidence | Price | Buy Target | Distance From Buy | Band | Source Sector | Status | Disposition | Decision Basis | Validation |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- |",
    ]
    _GRADE_SORT = {"ACTIONABLE": 0, "WATCHLIST_ONLY": 1, "DATA_INCOMPLETE": 2, "AVOID": 3}
    _CONF_SORT = {"HIGH": 0, "MODERATE": 1, "LOW": 2}

    def sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
        distance = row.get("distance_from_buy_pct")
        price_eligible = is_price_trigger_eligible(row)
        return (
            0 if price_eligible else 1,
            _SIGNAL_ORDER.get(signal_label(row.get("presented_status")), 9)
            if price_eligible
            else 9,
            _GRADE_SORT.get(str(row.get("conviction_grade") or "").upper(), 9),
            _CONF_SORT.get(str(row.get("confidence") or "").upper(), 9),
            distance if distance is not None else float("inf"),
            str(row.get("ticker") or ""),
        )

    for row in sorted(queue_rows, key=sort_key):
        price_eligible = is_price_trigger_eligible(row)
        displayed_signal = (
            signal_label(row.get("presented_status"))
            if price_eligible
            else "SCREEN RESEARCH"
            if str(row.get("decision_basis") or "").upper() == "SCREEN"
            else "RESEARCH ONLY"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    displayed_signal,
                    str(row.get("ticker") or "n/a"),
                    str(row.get("conviction_grade") or "n/a"),
                    str(row.get("confidence") or "n/a"),
                    _fmt_money(row.get("latest_price")),
                    _fmt_money(row.get("buy_price_target") if price_eligible else None),
                    _fmt_pct(row.get("distance_from_buy_pct") if price_eligible else None),
                    _band_label(row),
                    str(row.get("source_sector") or "n/a"),
                    str(row.get("status") or "n/a"),
                    str(row.get("candidate_disposition") or "n/a"),
                    str(row.get("decision_basis") or "n/a"),
                    str(row.get("selection_validation_status") or "n/a"),
                ]
            )
            + " |"
        )
    return lines


def _render_actionable_queue(
    rows: list[dict[str, Any]], latest_prices: dict[int, dict[str, Any]]
) -> list[str]:
    """Digest closer: the top ACTIONABLE-grade at-target names, three lines max.

    The daily artifact must end with the answer to "what is buyable today"
    instead of burying it mid-document. Only memo-backed (ACTIONABLE) rows
    qualify; event-blocked rows are counted, never listed. Reuses the
    imperative line, so the closing lines keep the "review
    trigger, not a buy signal" phrasing.
    """
    presentable: list[tuple[float, dict[str, Any], float | None]] = []
    blocked_count = 0
    for row in rows:
        if str(row["status"]) not in {"DEPLOY_READY", "BUY_CONFIRMED"}:
            continue
        if not is_price_trigger_eligible(row):
            continue
        if str(row.get("conviction_grade") or "").upper() != "ACTIONABLE":
            continue
        latest_price, _ = _latest_price_for_row(row, latest_prices)
        distance = _distance_from_buy(latest_price, row["buy_price_target"])
        if distance is None:
            continue
        if row.get("event_pending"):
            blocked_count += 1
            continue
        presentable.append((distance, row, latest_price))
    presentable.sort(key=lambda item: (item[0], str(item[1]["ticker"])))

    lines: list[str] = []
    for _, row, latest_price in presentable[:3]:
        lines.append(
            buy_now_imperative_line(
                str(row["ticker"]),
                latest_price,
                row["buy_price_target"],
                None,
                row["conviction_grade"],
            )
        )
    if not lines:
        lines.append(NO_ACTIONABLE_AT_TARGET)
    extras: list[str] = []
    overflow = len(presentable) - 3
    if overflow > 0:
        extras.append(f"+{overflow} more at target")
    if blocked_count:
        extras.append(f"{blocked_count} blocked pending event review")
    if extras:
        lines.append(f"({'; '.join(extras)} — see sections above)")
    return lines


def _render_research_pipeline_state(rows: list[dict[str, Any]]) -> list[str]:
    """Literal v2 disposition/provenance without changing legacy tables."""

    v2_rows = [
        row for row in rows if str(row.get("pipeline_version") or "").strip().lower() == "v2"
    ]
    if not v2_rows:
        return []
    lines = [
        "| Ticker | Disposition | Decision Basis | Validation | Investable |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in sorted(v2_rows, key=lambda item: str(item.get("ticker") or "")):
        lines.append(
            f"| {row['ticker']} | {row.get('candidate_disposition') or 'n/a'} | "
            f"{row.get('decision_basis') or 'n/a'} | "
            f"{row.get('selection_validation_status') or 'n/a'} | "
            f"{'yes' if is_price_trigger_eligible(row) else 'no'} |"
        )
    return lines


def _materialize_digest_snapshot(
    *,
    cutoff: datetime,
    now: datetime,
    db_path: str | Path | None,
    manifest_path: Path,
) -> _DigestSnapshot:
    """Read every watchlist decision input from one SQLite snapshot."""

    from app.events.cheapness import latest_cheapness_by_ticker
    from app.watchlist.dispositions import open_dispositions

    conn = _connect(db_path)
    try:
        conn.execute("BEGIN")
        watchlist_rows = _watchlist_rows(conn, manifest_path=manifest_path)
        authorized_watchlist_ids = {
            int(row["id"]) for row in watchlist_rows if row.get("id") is not None
        }
        latest_prices = {
            watchlist_id: row
            for watchlist_id, row in _latest_price_for_rows(conn).items()
            if watchlist_id in authorized_watchlist_ids
        }
        # watchlist_history carries no immutable row fingerprint or exact
        # source-artifact subrecord. A self-hashed digest snapshot cannot turn
        # that mutable auxiliary table into provenance, so current product
        # digests suppress transition/reevaluation claims until the writer
        # supplies an independently verifiable source binding.
        history_rows: list[dict[str, Any]] = []
        review_rows = watchlist_queue(
            limit=25,
            db_path=db_path,
            conn=conn,
            manifest_path=manifest_path,
        )
        authorized_tickers = sorted(
            {
                str(row.get("ticker") or "").strip().upper()
                for row in watchlist_rows
                if str(row.get("ticker") or "").strip()
            }
        )
        cheapness = (
            latest_cheapness_by_ticker(conn, authorized_tickers) if authorized_tickers else {}
        )
        try:
            open_disposition_rows = open_dispositions(
                conn=conn,
                manifest_path=manifest_path,
            )
        except sqlite3.Error:
            # Dispositions are an optional ledger on older/read-only database
            # snapshots. Preserve the legacy empty-section behavior without
            # opening a second connection outside the pinned snapshot.
            open_disposition_rows = []
        # exit_signals has the same provenance gap: its mutable evidence row is
        # not emitted by the autonomous source artifact and has no controlled
        # integrity fingerprint. Keep the ledger inspectable elsewhere, but do
        # not render it as an authorized current-decision row.
        held_exit_rows: list[dict[str, Any]] = []
        data_health = compute_data_health(
            db_path,
            now=now,
            conn=conn,
        )
        return _DigestSnapshot(
            watchlist_rows=watchlist_rows,
            latest_prices=latest_prices,
            history_rows=history_rows,
            review_rows=review_rows,
            cheapness=cheapness,
            open_dispositions=open_disposition_rows,
            held_exit_rows=held_exit_rows,
            data_health=data_health,
        )
    finally:
        conn.rollback()
        conn.close()


def render_digest(
    *,
    days_back: int = 1,
    now: datetime | None = None,
    db_path: str | Path | None = None,
    _snapshot: _DigestSnapshot | None = None,
    _manifest_decision: _DigestManifestDecision | None = None,
) -> str:
    manifest_decision_was_supplied = _manifest_decision is not None
    manifest_decision = _manifest_decision or _capture_manifest_decision()
    effective_now = now or _utc_now()
    if effective_now.tzinfo is None:
        effective_now = effective_now.replace(tzinfo=timezone.utc)
    effective_now = effective_now.astimezone(timezone.utc)
    cutoff = effective_now - timedelta(days=max(0, int(days_back)))

    if not manifest_decision.usable or manifest_decision.path is None:
        return _blocked_digest(effective_now)

    snapshot = _snapshot or _materialize_digest_snapshot(
        cutoff=cutoff,
        now=effective_now,
        db_path=db_path,
        manifest_path=manifest_decision.path,
    )
    watchlist_rows = snapshot.watchlist_rows
    latest_prices = snapshot.latest_prices
    history_rows = snapshot.history_rows
    cheapness = snapshot.cheapness
    review_rows = snapshot.review_rows

    new_entries = [
        row
        for row in watchlist_rows
        if str(row.get("status") or "").upper() != "REMOVED"
        and _in_window(row["added_at"], cutoff=cutoff)
    ]

    # Deterministic data health is captured inside the same SQLite
    # transaction as every other decision-bearing digest input.
    health = snapshot.data_health

    lines = [
        "# IVI Watchlist Daily Digest",
        "",
        f"- Generated: {effective_now.isoformat()}",
        f"- Window start: {cutoff.isoformat()}",
        f"- Days back: {max(0, int(days_back))}",
    ]
    _append_section(lines, "Data Health", health.lines())
    _append_section(
        lines,
        "Signal Board",
        render_compact_signal_table(review_rows),
    )
    blocked_note = [
        "**BLOCKED: data health failed at the engine level — at-target rows"
        " are suppressed rather than rendered from a suspect DB.**"
    ]
    _append_section(
        lines,
        "At Buy Target (Review)",
        blocked_note
        if health.blocking
        else _render_buy_now(watchlist_rows, latest_prices, cheapness, now=effective_now),
    )
    _append_section(
        lines,
        "Blocked Pending Event Review",
        blocked_note
        if health.blocking
        else _render_event_blocked(watchlist_rows, latest_prices, cheapness),
    )
    held_exit_rows = _render_held_book_exits(snapshot.held_exit_rows)
    if held_exit_rows:
        _append_section(lines, "Exits (Held Book)", held_exit_rows)
    open_disposition_rows = _render_open_dispositions(
        db_path,
        now=effective_now,
        rows=snapshot.open_dispositions,
    )
    if open_disposition_rows:
        _append_section(lines, "Open Dispositions (decision required)", open_disposition_rows)
    _append_section(lines, "Review Today", _render_review_today(review_rows))
    research_pipeline_rows = _render_research_pipeline_state(watchlist_rows)
    if research_pipeline_rows:
        _append_section(lines, "Research Pipeline State", research_pipeline_rows)
    buy_confirmed_rows = (
        [] if health.blocking else _render_buy_confirmed(watchlist_rows, latest_prices)
    )
    if buy_confirmed_rows:
        lines.extend(["", "## At Target + Catalyst Confirmed", *buy_confirmed_rows])
    _append_section(
        lines, "Newly Deploy-Ready", _render_transition_rows(history_rows, new_value="DEPLOY_READY")
    )
    _append_section(
        lines, "Newly Contradicted", _render_transition_rows(history_rows, new_value="CONTRADICTED")
    )
    _append_section(
        lines,
        "Newly Removed (Universe Exits)",
        _render_universe_exits(history_rows),
    )
    _append_section(lines, "Newly Added", _render_new_entries(new_entries))
    _append_section(lines, "Recent Re-Evaluations", _render_reevaluations(history_rows))
    _append_section(
        lines, "Price Data Quality Issues", _render_price_quality_issues(watchlist_rows)
    )
    _append_section(lines, "Active Watchlist Summary", _render_status_counts(watchlist_rows))
    _append_section(
        lines,
        "Within Reach (above target)",
        _render_top_active_entries(watchlist_rows, latest_prices),
    )
    _append_section(
        lines,
        "Current Actionable Queue",
        blocked_note
        if health.blocking
        else _render_actionable_queue(watchlist_rows, latest_prices),
    )
    visible_markdown = "\n".join(lines).rstrip() + "\n\n"
    rendered_state = _digest_rendered_state(snapshot)
    marker = _digest_rendered_state_marker(
        rendered_state_sha256=_digest_payload_sha256(rendered_state),
        rendered_markdown_sha256=hashlib.sha256(visible_markdown.encode("utf-8")).hexdigest(),
    )
    markdown = visible_markdown + marker + "\n"
    if not _manifest_decision_is_current(manifest_decision):
        if manifest_decision_was_supplied:
            raise RuntimeError(
                "Refusing to render a digest after the pinned financial-integrity manifest changed"
            )
        return _blocked_digest(effective_now)
    return markdown


def default_digest_path(*, now: datetime | None = None) -> Path:
    effective_now = now or _utc_now()
    return get_config().outputs_dir / "digests" / f"digest_{effective_now.date().isoformat()}.md"


def _digest_lineage_path(path: Path) -> Path:
    return path.with_suffix(".lineage.json")


def _digest_decision_state(row: dict[str, Any]) -> dict[str, Any]:
    return {field: row.get(field) for field in DIGEST_DECISION_STATE_FIELDS}


def _digest_payload_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _digest_decision_state_sha256(state: dict[str, Any]) -> str:
    return _digest_payload_sha256(state)


def _digest_rendered_state(snapshot: _DigestSnapshot) -> dict[str, Any]:
    """Canonical values that can affect a decision-bearing digest render."""

    latest_prices = [
        dict(row)
        for _, row in sorted(
            snapshot.latest_prices.items(),
            key=lambda item: (int(item[0]), int(item[1].get("id") or 0)),
        )
    ]
    return {
        "watchlist_rows": [dict(row) for row in snapshot.watchlist_rows],
        "latest_prices": latest_prices,
        "history_rows": [dict(row) for row in snapshot.history_rows],
        "review_rows": [dict(row) for row in snapshot.review_rows],
        "cheapness": {
            str(ticker): dict(report) for ticker, report in sorted(snapshot.cheapness.items())
        },
        "open_dispositions": [dict(row) for row in snapshot.open_dispositions],
        "held_exit_rows": [dict(row) for row in snapshot.held_exit_rows],
        "data_health": {
            "checks": [dict(check) for check in snapshot.data_health.checks],
            "blocking": bool(snapshot.data_health.blocking),
        },
    }


def _digest_rendered_state_marker(
    *,
    rendered_state_sha256: str,
    rendered_markdown_sha256: str,
) -> str:
    payload = {
        "schema_version": DIGEST_RENDERED_STATE_SCHEMA_VERSION,
        "rendered_state_sha256": rendered_state_sha256,
        "rendered_markdown_sha256": rendered_markdown_sha256,
    }
    encoded = base64.b64encode(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).decode("ascii")
    return f"<!-- {_DIGEST_RENDERED_STATE_MARKER}{encoded} -->"


def _digest_source_lineage(
    *,
    rows: list[dict[str, Any]],
    manifest_path: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    lineage: list[dict[str, Any]] = []
    missing: list[str] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        ticker = str(row.get("ticker") or "").strip().upper()
        run_id = str(row.get("source_run_id") or "").strip()
        if not ticker or not run_id or (ticker, run_id) in seen:
            continue
        binding = authorized_watchlist_decision_binding(row, manifest_path)
        if binding is None:
            missing.append(f"{ticker}:{run_id}")
            continue
        decision_state = _digest_decision_state(row)
        try:
            decision_state_sha256 = _digest_decision_state_sha256(decision_state)
        except (TypeError, ValueError):
            missing.append(f"{ticker}:{run_id}:noncanonical_decision_state")
            continue
        seen.add((ticker, run_id))
        lineage.append(
            {
                "ticker": ticker,
                "source_run_id": run_id,
                "source_artifact_path": binding["source_artifact_path"],
                "source_artifact_sha256": binding["source_artifact_sha256"],
                "source_decision_fingerprint": binding["source_decision_fingerprint"],
                "decision_state": decision_state,
                "decision_state_sha256": decision_state_sha256,
            }
        )
    lineage.sort(key=lambda item: (item["ticker"], item["source_run_id"]))
    return lineage, sorted(missing)


def write_digest(
    *,
    days_back: int = 1,
    output_path: str | Path | None = None,
    check_prices_first: bool = False,
    db_path: str | Path | None = None,
    now: datetime | None = None,
) -> DigestWriteResult:
    manifest_decision = _capture_manifest_decision()
    trigger_checked = False
    if check_prices_first and manifest_decision.usable:
        if not _manifest_decision_is_current(manifest_decision):
            raise RuntimeError(
                "Refusing to check prices after the pinned financial-integrity manifest changed"
            )
        check_watchlist_triggers(db_path=db_path)
        trigger_checked = True
    effective_now = now or _utc_now()
    if effective_now.tzinfo is None:
        effective_now = effective_now.replace(tzinfo=timezone.utc)
    effective_now = effective_now.astimezone(timezone.utc)
    snapshot: _DigestSnapshot | None = None
    if manifest_decision.usable:
        if manifest_decision.path is None:
            raise RuntimeError("usable digest manifest decision has no path")
        snapshot = _materialize_digest_snapshot(
            cutoff=effective_now - timedelta(days=max(0, int(days_back))),
            now=effective_now,
            db_path=db_path,
            manifest_path=manifest_decision.path,
        )
    markdown = render_digest(
        days_back=days_back,
        now=effective_now,
        db_path=db_path,
        _snapshot=snapshot,
        _manifest_decision=manifest_decision,
    )
    path = Path(output_path) if output_path is not None else default_digest_path(now=now)
    path = path.expanduser().resolve()
    lineage: list[dict[str, Any]] = []
    rendered_state: dict[str, Any] | None = None
    rendered_state_sha256: str | None = None
    rendered_markdown_sha256: str | None = None
    if manifest_decision.usable:
        if snapshot is None:
            raise RuntimeError("digest snapshot was not materialized")
        if manifest_decision.path is None or manifest_decision.sha256 is None:
            raise RuntimeError("usable digest manifest decision is incomplete")
        lineage, missing = _digest_source_lineage(
            rows=snapshot.watchlist_rows,
            manifest_path=manifest_decision.path,
        )
        if missing:
            raise RuntimeError(
                "Refusing to publish a digest with missing exact source lineage: "
                + ", ".join(missing)
            )
        rendered_state = _digest_rendered_state(snapshot)
        rendered_state_sha256 = _digest_payload_sha256(rendered_state)
        marker_prefix = f"<!-- {_DIGEST_RENDERED_STATE_MARKER}"
        marker_start = markdown.rfind(marker_prefix)
        if marker_start < 0:
            raise RuntimeError("digest rendered-state marker is missing")
        rendered_markdown_sha256 = hashlib.sha256(
            markdown[:marker_start].encode("utf-8")
        ).hexdigest()
        if not _manifest_decision_is_current(manifest_decision):
            raise RuntimeError(
                "Refusing to publish a digest after the pinned financial-integrity manifest changed"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(markdown, encoding="utf-8")
    lineage_path: Path | None = None
    lineage_temporary: Path | None = None
    if manifest_decision.usable:
        assert manifest_decision.path is not None
        assert manifest_decision.sha256 is not None
        assert rendered_state is not None
        assert rendered_state_sha256 is not None
        assert rendered_markdown_sha256 is not None
        lineage_path = _digest_lineage_path(path)
        lineage_payload = {
            "schema_version": DIGEST_LINEAGE_SCHEMA_VERSION,
            "digest_path": str(path),
            "digest_sha256": hashlib.sha256(markdown.encode("utf-8")).hexdigest(),
            "rendered_state": rendered_state,
            "rendered_state_sha256": rendered_state_sha256,
            "rendered_markdown_sha256": rendered_markdown_sha256,
            "manifest_path": str(manifest_decision.path),
            "manifest_sha256": manifest_decision.sha256,
            "rows": lineage,
        }
        lineage_temporary = lineage_path.with_name(f".{lineage_path.name}.{uuid4().hex}.tmp")
        lineage_temporary.write_text(
            json.dumps(lineage_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if not _manifest_decision_is_current(manifest_decision):
            temporary.unlink(missing_ok=True)
            lineage_temporary.unlink(missing_ok=True)
            raise RuntimeError(
                "Refusing to publish a digest after the pinned financial-integrity manifest changed"
            )
        lineage_temporary.replace(lineage_path)
    temporary.replace(path)
    return DigestWriteResult(
        markdown=markdown,
        path=path,
        trigger_checked=trigger_checked,
        lineage_path=lineage_path,
    )
