"""Deterministic data-health block.

One computation feeding three surfaces: the top of the daily digest, the
top of ``ivi investor buy-now``, and the 16:30 dead-man check. Zero LLM,
read-only.

Two modes:

- **Production** (no ``db_path`` argument AND the configured DB is the
  canonical books-of-record location ``<repo>/data/engine.db``): full
  checks — preflight (absolute path, engine present, row floors, cache
  symlinks), previous-business-day event scan, price-snapshot age, backup
  freshness. An engine-level failure (missing/empty/shadow DB) is
  ``blocking``: at-target rendering must be suppressed, because "no names
  at target" read from a shadow DB is a lie, not a quiet day.
- **Explicit/overridden db_path** (tests, backtests, ad-hoc copies, a
  VOE_DB_PATH override): cheap mode — the caller chose the DB
  deliberately, so only its basic readability is checked and row floors /
  cron-cadence checks are skipped.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from app.config import get_config

PRICE_SNAPSHOT_MAX_AGE_HOURS = 96
PRICE_SNAPSHOT_MIN_CANONICAL_COVERAGE = 0.95
BACKUP_MAX_AGE_DAYS = 4


@dataclass
class DataHealth:
    checks: list[dict[str, Any]] = field(default_factory=list)
    blocking: bool = False

    @property
    def red(self) -> bool:
        return any(not c["ok"] for c in self.checks)

    @property
    def red_lines(self) -> list[str]:
        return [f"{c['name']}: {c['detail']}" for c in self.checks if not c["ok"]]

    @property
    def state(self) -> str:
        if self.blocking:
            return "RED"
        if self.red:
            return "AMBER"
        return "GREEN"

    def lines(self) -> list[str]:
        if not self.red:
            return ["- data health: OK (all checks green)"]
        count = len(self.red_lines)
        noun = "failing" if self.blocking else "warning"
        if count != 1:
            noun += "s"
        out = [f"- **DATA HEALTH {self.state}** ({count} {noun}):"]
        out.extend(f"  - {line}" for line in self.red_lines)
        if self.blocking:
            out.append(
                "  - **BLOCKING**: engine DB failed basic integrity — at-target"
                " sections are suppressed until this is fixed"
            )
        else:
            out.append(
                "  - **NONBLOCKING**: operational freshness warning; "
                "candidate-level gates remain authoritative"
            )
        return out


def _previous_business_day(anchor: date) -> date:
    day = anchor - timedelta(days=1)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day


def _readonly_conn(db_path: Path) -> sqlite3.Connection | None:
    try:
        return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10.0)
    except sqlite3.Error:
        if db_path.is_file():
            try:
                return sqlite3.connect(str(db_path), timeout=10.0)
            except sqlite3.Error:
                return None
        return None


def check_event_scan_fresh(
    db_path: Path,
    *,
    today: date | None = None,
    conn: sqlite3.Connection | None = None,
) -> tuple[bool, str]:
    """Previous business day's corporate-event scan exists with status OK."""
    prev_bd = _previous_business_day(today or date.today()).isoformat()
    selected_conn = conn or _readonly_conn(db_path)
    if selected_conn is None:
        return False, f"event_scan: engine DB unreadable at {db_path}"
    try:
        row = selected_conn.execute(
            "SELECT status FROM corporate_event_scans WHERE scan_date = ?",
            (prev_bd,),
        ).fetchone()
    except sqlite3.Error as exc:
        return False, f"event_scan: {exc}"
    finally:
        if conn is None:
            selected_conn.close()
    status = str(row[0]) if row else None
    return status == "OK", f"event scan {prev_bd} -> {status or 'MISSING'}"


def check_price_snapshots_fresh(
    db_path: Path,
    *,
    now: datetime | None = None,
    conn: sqlite3.Connection | None = None,
) -> tuple[bool, str]:
    """Newest watchlist price snapshot within the staleness ceiling."""
    effective_now = now or datetime.now(timezone.utc)
    selected_conn = conn or _readonly_conn(db_path)
    if selected_conn is None:
        return False, f"price_snapshots: engine DB unreadable at {db_path}"
    try:
        row = selected_conn.execute(
            "SELECT MAX(checked_at) FROM watchlist_price_snapshots"
        ).fetchone()
    except sqlite3.Error as exc:
        return False, f"price_snapshots: {exc}"
    finally:
        if conn is None:
            selected_conn.close()
    latest_raw = row[0] if row else None
    if not latest_raw:
        return False, "price snapshots: no rows"
    try:
        latest = datetime.fromisoformat(str(latest_raw).replace("Z", "+00:00"))
    except ValueError:
        return False, f"price snapshots: unparseable checked_at {latest_raw!r}"
    if latest.tzinfo is None:
        latest = latest.replace(tzinfo=timezone.utc)
    age_h = (effective_now - latest).total_seconds() / 3600.0
    return (
        age_h <= PRICE_SNAPSHOT_MAX_AGE_HOURS,
        f"newest price snapshot {age_h:.1f}h old (ceiling {PRICE_SNAPSHOT_MAX_AGE_HOURS}h)",
    )


def check_price_snapshot_coverage(
    db_path: Path,
    *,
    now: datetime | None = None,
    conn: sqlite3.Connection | None = None,
) -> tuple[bool, str]:
    """Require fresh snapshots on the authoritative row for most current tickers."""

    from app.watchlist.store import current_watchlist_cte

    effective_now = now or datetime.now(timezone.utc)
    selected_conn = conn or _readonly_conn(db_path)
    if selected_conn is None:
        return False, f"price snapshot coverage: engine DB unreadable at {db_path}"
    try:
        columns = {
            str(row[1])
            for row in selected_conn.execute(
                "PRAGMA table_info(watchlist_price_snapshots)"
            ).fetchall()
        }
        required = {"quote_as_of_date", "quote_snapshot_id"}
        if not required <= columns:
            # The provenance migration is owner-applied and gates activation.
            # Until it lands the refresh feature is inert, so the dead-man
            # channel must stay green rather than alarm on an unactivated check.
            return (
                True,
                "price snapshot coverage: not activated"
                " (provenance migration unapplied)",
            )
        rows = selected_conn.execute(
            f"""
            WITH latest_watchlist AS ({current_watchlist_cte()}),
            current_rows AS (
                SELECT w.id, w.ticker
                FROM watchlist w
                JOIN latest_watchlist ON latest_watchlist.latest_id = w.id
                WHERE w.status != 'REMOVED'
            ),
            latest_snapshot AS (
                SELECT *
                FROM (
                    SELECT
                        s.*,
                        ROW_NUMBER() OVER (
                            PARTITION BY s.watchlist_id
                            ORDER BY s.checked_at DESC, s.id DESC
                        ) AS snapshot_rank
                    FROM watchlist_price_snapshots s
                )
                WHERE snapshot_rank = 1
            )
            SELECT c.ticker, s.checked_at, s.quote_as_of_date, s.quote_snapshot_id
            FROM current_rows c
            LEFT JOIN latest_snapshot s ON s.watchlist_id = c.id
            ORDER BY c.ticker
            """
        ).fetchall()
    except sqlite3.Error as exc:
        return False, f"price snapshot coverage: {exc}"
    finally:
        if conn is None:
            selected_conn.close()
    candidates = len(rows)
    if candidates == 0:
        return False, "price snapshot coverage: no canonical current candidates"
    covered = 0
    for row in rows:
        checked_raw = row[1]
        quote_as_of_raw = row[2]
        quote_snapshot_id = row[3]
        if not checked_raw or not quote_as_of_raw or not quote_snapshot_id:
            continue
        try:
            checked = datetime.fromisoformat(str(checked_raw).replace("Z", "+00:00"))
            if checked.tzinfo is None:
                checked = checked.replace(tzinfo=timezone.utc)
            quote_as_of = date.fromisoformat(str(quote_as_of_raw)[:10])
        except ValueError:
            continue
        checked_age_h = (effective_now - checked).total_seconds() / 3600.0
        quote_age_days = (effective_now.date() - quote_as_of).days
        if (
            checked_age_h <= PRICE_SNAPSHOT_MAX_AGE_HOURS
            and 0 <= quote_age_days <= 5
        ):
            covered += 1
    ratio = covered / candidates
    unavailable = candidates - covered
    ok = ratio >= PRICE_SNAPSHOT_MIN_CANONICAL_COVERAGE
    return (
        ok,
        f"canonical fresh coverage {covered}/{candidates} ({ratio:.1%}); "
        f"unavailable/stale={unavailable} "
        f"(floor {PRICE_SNAPSHOT_MIN_CANONICAL_COVERAGE:.0%})",
    )


def check_backup_fresh(*, today: date | None = None) -> tuple[bool, str]:
    """Latest backup status file says OK and is recent."""
    cfg = get_config()
    effective_today = today or date.today()
    status_path = Path(cfg.outputs_dir) / "cron" / "backup_status_latest.json"
    if not status_path.is_file():
        return False, f"backup: no status file at {status_path}"
    try:
        payload = json.loads(status_path.read_text())
        backup_date = datetime.strptime(str(payload.get("date")), "%Y%m%d").date()
    except (ValueError, TypeError, OSError) as exc:
        return False, f"backup: unparseable status file: {exc}"
    age_days = (effective_today - backup_date).days
    ok = str(payload.get("status")) == "OK" and age_days <= BACKUP_MAX_AGE_DAYS
    return ok, f"backup status={payload.get('status')} ({age_days}d old)"


def compute_data_health(
    db_path: str | Path | None = None,
    *,
    now: datetime | None = None,
    conn: sqlite3.Connection | None = None,
) -> DataHealth:
    health = DataHealth()

    def add(name: str, ok: bool, detail: str) -> None:
        health.checks.append({"name": name, "ok": bool(ok), "detail": detail})

    if db_path is None:
        # Full production checks only apply to the canonical books-of-record
        # DB; a VOE_DB_PATH override (tests, ad-hoc copies) gets cheap mode.
        cfg = get_config()
        canonical = cfg.project_root / "data" / "engine.db"
        if Path(cfg.db_path) != canonical:
            db_path = Path(cfg.db_path)

    if db_path is not None:
        # Cheap mode: an explicitly chosen DB only needs to be readable.
        path = Path(db_path)
        if not path.is_file():
            add("db_readable", False, f"missing: {path}")
            health.blocking = True
            return health
        selected_conn = conn or _readonly_conn(path)
        if selected_conn is None:
            add("db_readable", False, f"unreadable: {path}")
            health.blocking = True
            return health
        try:
            selected_conn.execute("SELECT 1 FROM watchlist LIMIT 1")
            add("db_readable", True, str(path))
        except sqlite3.Error as exc:
            add("db_readable", False, f"watchlist table unreadable: {exc}")
            health.blocking = True
        finally:
            if conn is None:
                selected_conn.close()
        return health

    return compute_data_health_full(now=now, conn=conn)


def compute_data_health_full(
    *,
    now: datetime | None = None,
    conn: sqlite3.Connection | None = None,
) -> DataHealth:
    """Full production checks against the configured engine DB."""
    from app.ops.preflight import run_preflight

    health = DataHealth()

    def add(name: str, ok: bool, detail: str) -> None:
        health.checks.append({"name": name, "ok": bool(ok), "detail": detail})

    preflight = run_preflight(db_conn=conn)
    engine_check_names = {
        "db_path_absolute",
        "engine_db_exists",
        "engine_db_row_floor_watchlist",
        "engine_db_row_floor_companyfacts",
    }
    for check in preflight.checks:
        add(check.name, check.ok, check.detail)
        if not check.ok and check.name in engine_check_names:
            health.blocking = True

    cfg = get_config()
    engine_path = Path(cfg.db_path)
    if not health.blocking:
        # Local calendar date, not UTC: after ~17:00 PT the UTC date has
        # already rolled over, which made the evening ops deck demand a scan
        # that only exists after the next morning's heartbeat.
        today = (now or datetime.now(timezone.utc)).astimezone().date()
        ok, detail = check_event_scan_fresh(engine_path, today=today, conn=conn)
        add("event_scan_fresh", ok, detail)
        ok, detail = check_price_snapshots_fresh(engine_path, now=now, conn=conn)
        add("price_snapshots_fresh", ok, detail)
        ok, detail = check_price_snapshot_coverage(engine_path, now=now, conn=conn)
        add("price_snapshot_coverage", ok, detail)
        ok, detail = check_backup_fresh(today=today)
        add("backup_fresh", ok, detail)
    return health


__all__ = [
    "BACKUP_MAX_AGE_DAYS",
    "PRICE_SNAPSHOT_MAX_AGE_HOURS",
    "DataHealth",
    "check_backup_fresh",
    "check_event_scan_fresh",
    "check_price_snapshots_fresh",
    "compute_data_health",
]
