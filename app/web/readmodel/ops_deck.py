"""Ops-deck read model: heartbeats, backups, costs, cron logs, consoles.

The deadman wall itself is :func:`app.ops.data_health.compute_data_health`
serialized verbatim (the route reuses ``today.serialize_health``); this
module covers the rest of the morning check:

- **heartbeat ledger** — ``heartbeat_runs`` grouped heartbeat × date, each
  cell derived through the production
  :func:`app.ops.heartbeat_ledger.derive_status` (never reimplemented), with
  the day's cron log attached when one exists on disk.
- **backups** — ``check_backup_fresh`` verbatim plus the raw status payload
  and the nightly log history.
- **cost ledger** — weekly spend from the run index (v2 artifacts carry
  ``lane_usage``; v1 runs have no cost fields and are counted honestly as
  uncosted) plus ``synthesis_packets.cost_estimate_usd`` by week and model.
- **cron logs** — a strict directory listing and tail reader for
  ``data/outputs/cron`` only; names are validated, paths are resolved, and
  nothing outside that directory is ever served.
- **consoles** — the legacy /ops tables (dead letters, research gaps,
  pipeline runs) carried into the new shell as read-only lists.
"""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from app.config import get_config
from app.ops.data_health import BACKUP_MAX_AGE_DAYS, check_backup_fresh
from app.ops.heartbeat_ledger import derive_status

_LOG_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.(log|json|err)$")
_DATED_LOG_PATTERN = re.compile(r"^(?P<prefix>[a-z0-9_]+)_(?P<yyyymmdd>\d{8})\.log$")

DEFAULT_TAIL_LINES = 200
MAX_TAIL_LINES = 2000
_TAIL_READ_BYTES = 512 * 1024


def cron_dir() -> Path:
    return Path(get_config().outputs_dir) / "cron"


# --- heartbeat ledger --------------------------------------------------------


def _dated_logs(directory: Path) -> dict[tuple[str, str], str]:
    """(first-token, yyyymmdd) → filename for every dated cron log."""
    if not directory.is_dir():
        return {}
    mapping: dict[tuple[str, str], str] = {}
    for path in directory.iterdir():
        match = _DATED_LOG_PATTERN.match(path.name)
        if not match:
            continue
        token = match.group("prefix").split("_", 1)[0]
        mapping[(token, match.group("yyyymmdd"))] = path.name
    return mapping


def heartbeat_ledger(
    conn: sqlite3.Connection,
    *,
    days: int = 14,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Heartbeat × date grid over the last N calendar days, newest first.

    Cell status comes from the shared ``derive_status`` derivation; MISSING
    cells are real cells (a heartbeat that didn't run is the signal, not an
    absence of data). Log names attach by the heartbeat's first name token +
    the date, matching the shell heartbeats' ``<prefix>_YYYYMMDD.log``
    convention.
    """
    if days <= 0:
        raise ValueError("days must be positive")
    today = (now or datetime.now(timezone.utc)).date()
    dates = [(today - timedelta(days=offset)).isoformat() for offset in range(days)]

    rows = conn.execute(
        """
        SELECT heartbeat, run_date, step, status, exit_code, detail, recorded_at
        FROM heartbeat_runs
        WHERE run_date >= ?
        ORDER BY heartbeat, run_date, recorded_at ASC, id ASC
        """,
        (dates[-1],),
    ).fetchall()

    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        grouped.setdefault(str(row["heartbeat"]), {}).setdefault(
            str(row["run_date"]), []
        ).append(
            {
                "step": row["step"],
                "status": row["status"],
                "exit_code": row["exit_code"],
                "detail": row["detail"],
                "recorded_at": row["recorded_at"],
            }
        )

    logs = _dated_logs(cron_dir())
    heartbeats: list[dict[str, Any]] = []
    for heartbeat in sorted(grouped):
        token = heartbeat.split("_", 1)[0]
        cells: list[dict[str, Any]] = []
        for day in dates:
            day_rows = grouped[heartbeat].get(day, [])
            derived = derive_status(day_rows)
            cells.append(
                {
                    "run_date": day,
                    "status": derived["status"],
                    "failed_steps": derived["failed_steps"],
                    "steps": day_rows,
                    "log_name": logs.get((token, day.replace("-", ""))),
                }
            )
        heartbeats.append({"heartbeat": heartbeat, "days": cells})

    return {"dates": dates, "heartbeats": heartbeats}


# --- backups -----------------------------------------------------------------


def backups(*, now: datetime | None = None) -> dict[str, Any]:
    """The dead-man backup check verbatim, plus payload and log history."""
    today = (now or datetime.now(timezone.utc)).date()
    ok, detail = check_backup_fresh(today=today)

    payload: dict[str, Any] = {}
    status_path = cron_dir() / "backup_status_latest.json"
    if status_path.is_file():
        try:
            loaded = json.loads(status_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                payload = loaded
        except (ValueError, OSError):
            payload = {}

    age_days: int | None = None
    try:
        backup_date = datetime.strptime(str(payload.get("date")), "%Y%m%d").date()
        age_days = (today - backup_date).days
    except (ValueError, TypeError):
        age_days = None

    history: list[dict[str, Any]] = []
    directory = cron_dir()
    if directory.is_dir():
        for path in sorted(directory.glob("nightly_backup_*.log"), reverse=True):
            match = _DATED_LOG_PATTERN.match(path.name)
            history.append(
                {
                    "log_name": path.name,
                    "date": match.group("yyyymmdd") if match else None,
                    "size": path.stat().st_size,
                }
            )

    return {
        "ok": ok,
        "detail": detail,
        "ceiling_days": BACKUP_MAX_AGE_DAYS,
        "age_days": age_days,
        "status": payload.get("status"),
        "status_detail": payload.get("detail"),
        "date": payload.get("date"),
        "duration_s": payload.get("duration_s"),
        "finished_at": payload.get("finished_at"),
        "history": history,
    }


# --- cost ledger -------------------------------------------------------------


def _iso_week(stamp: str | None) -> str | None:
    if not stamp:
        return None
    text = str(stamp).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        try:
            parsed = datetime.combine(date.fromisoformat(text[:10]), datetime.min.time())
        except ValueError:
            return None
    iso = parsed.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


def cost_ledger(
    ui_conn: sqlite3.Connection,
    engine_conn: sqlite3.Connection | None,
) -> dict[str, Any]:
    """Spend over time: scan runs (run index) + research synthesis calls.

    Run costs exist only on v2 artifacts (``lane_usage`` microdollars); v1
    runs are tallied as ``runs_without_cost`` rather than pretending they
    were free. Research costs come from ``synthesis_packets`` with a model
    breakdown — cached replays carry cost 0 and are excluded from spend but
    counted as calls.
    """
    weeks: dict[str, dict[str, Any]] = {}
    by_sector: dict[str, dict[str, Any]] = {}
    by_band: dict[str, dict[str, Any]] = {}
    runs_with_cost = 0
    runs_without_cost = 0

    for row in ui_conn.execute(
        """
        SELECT created_at, as_of_date, sector, market_cap_focus, cost_microdollars
        FROM run_index
        WHERE parse_error IS NULL
        """
    ):
        week = _iso_week(row["created_at"] or row["as_of_date"])
        cost_micro = row["cost_microdollars"]
        if not isinstance(cost_micro, int):
            runs_without_cost += 1
            continue
        runs_with_cost += 1
        cost_usd = cost_micro / 1_000_000
        if week is not None:
            bucket = weeks.setdefault(
                week,
                {"week": week, "scan_cost_usd": 0.0, "scan_runs": 0,
                 "research_cost_usd": 0.0, "research_calls": 0},
            )
            bucket["scan_cost_usd"] += cost_usd
            bucket["scan_runs"] += 1
        sector = str(row["sector"] or "unknown")
        sector_bucket = by_sector.setdefault(sector, {"key": sector, "cost_usd": 0.0, "runs": 0})
        sector_bucket["cost_usd"] += cost_usd
        sector_bucket["runs"] += 1
        band = str(row["market_cap_focus"] or "unknown")
        band_bucket = by_band.setdefault(band, {"key": band, "cost_usd": 0.0, "runs": 0})
        band_bucket["cost_usd"] += cost_usd
        band_bucket["runs"] += 1

    by_model: dict[str, dict[str, Any]] = {}
    if engine_conn is not None:
        for row in engine_conn.execute(
            """
            SELECT created_at, as_of_date, provider, model,
                   COALESCE(cost_estimate_usd, 0.0) AS cost_usd,
                   COALESCE(from_cache, 0) AS from_cache
            FROM synthesis_packets
            """
        ):
            week = _iso_week(row["created_at"] or row["as_of_date"])
            spend = float(row["cost_usd"]) if not row["from_cache"] else 0.0
            if week is not None:
                bucket = weeks.setdefault(
                    week,
                    {"week": week, "scan_cost_usd": 0.0, "scan_runs": 0,
                     "research_cost_usd": 0.0, "research_calls": 0},
                )
                bucket["research_cost_usd"] += spend
                bucket["research_calls"] += 1
            model = f"{row['provider'] or '?'}/{row['model'] or '?'}"
            model_bucket = by_model.setdefault(
                model, {"key": model, "cost_usd": 0.0, "calls": 0, "cached_calls": 0}
            )
            model_bucket["cost_usd"] += spend
            model_bucket["calls"] += 1
            if row["from_cache"]:
                model_bucket["cached_calls"] += 1

    ordered_weeks = sorted(weeks.values(), key=lambda w: w["week"], reverse=True)
    for bucket in ordered_weeks:
        bucket["scan_cost_usd"] = round(bucket["scan_cost_usd"], 4)
        bucket["research_cost_usd"] = round(bucket["research_cost_usd"], 4)
        bucket["total_usd"] = round(bucket["scan_cost_usd"] + bucket["research_cost_usd"], 4)

    def _rounded(buckets: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        ordered = sorted(buckets.values(), key=lambda b: b["cost_usd"], reverse=True)
        for bucket in ordered:
            bucket["cost_usd"] = round(bucket["cost_usd"], 4)
        return ordered

    return {
        "weeks": ordered_weeks,
        "by_sector": _rounded(by_sector),
        "by_band": _rounded(by_band),
        "by_model": _rounded(by_model),
        "runs_with_cost": runs_with_cost,
        "runs_without_cost": runs_without_cost,
        "total_scan_usd": round(sum(w["scan_cost_usd"] for w in ordered_weeks), 4),
        "total_research_usd": round(sum(w["research_cost_usd"] for w in ordered_weeks), 4),
    }


# --- cron logs ---------------------------------------------------------------


def list_cron_logs() -> list[dict[str, Any]]:
    directory = cron_dir()
    if not directory.is_dir():
        return []
    files: list[dict[str, Any]] = []
    for path in directory.iterdir():
        if not path.is_file() or not _LOG_NAME_PATTERN.match(path.name):
            continue
        stat = path.stat()
        files.append(
            {
                "name": path.name,
                "size": stat.st_size,
                "mtime": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            }
        )
    files.sort(key=lambda f: f["mtime"], reverse=True)
    return files


def read_log_tail(name: str, *, lines: int = DEFAULT_TAIL_LINES) -> dict[str, Any]:
    """Tail of one cron log. Raises FileNotFoundError / ValueError on refusal.

    The name must be a plain validated filename and the resolved path must
    live directly in the cron directory — no separators, no traversal, no
    symlink escape.
    """
    if lines <= 0 or lines > MAX_TAIL_LINES:
        raise ValueError(f"lines must be in 1..{MAX_TAIL_LINES}")
    if not _LOG_NAME_PATTERN.match(name):
        raise ValueError(f"Invalid log name: {name!r}")
    directory = cron_dir().resolve()
    target = (directory / name).resolve()
    if target.parent != directory:
        raise ValueError(f"Invalid log name: {name!r}")
    if not target.is_file():
        raise FileNotFoundError(name)

    stat = target.stat()
    with target.open("rb") as handle:
        if stat.st_size > _TAIL_READ_BYTES:
            handle.seek(-_TAIL_READ_BYTES, 2)
        raw = handle.read()
    text = raw.decode("utf-8", errors="replace")
    all_lines = text.splitlines()
    tail = all_lines[-lines:]
    return {
        "name": name,
        "lines": tail,
        "truncated": stat.st_size > _TAIL_READ_BYTES or len(all_lines) > lines,
        "size": stat.st_size,
        "mtime": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }


# --- legacy consoles ---------------------------------------------------------


def consoles(conn: sqlite3.Connection) -> dict[str, Any]:
    """The legacy /ops tables as read-only lists (carried into the shell)."""
    deadletters = [
        {
            "id": int(row["id"]),
            "job_id": row["job_id"],
            "job_type": row["job_type"],
            "attempts": row["attempts"],
            "error_type": row["error_type"],
            "error_message": row["error_message"],
            "moved_at": row["moved_at"],
        }
        for row in conn.execute(
            """
            SELECT id, job_id, job_type, attempts, error_type, error_message, moved_at
            FROM dead_letter_jobs
            ORDER BY moved_at DESC, id DESC
            LIMIT 200
            """
        )
    ]
    backlog = int(
        conn.execute("SELECT COUNT(*) AS n FROM jobs WHERE status='pending'").fetchone()["n"]
    )

    research_gaps: list[dict[str, Any]] = []
    for row in conn.execute(
        """
        SELECT ticker, as_of_date, run_id, recency_days_min, item_count_30d,
               sentiment_flags_json, summary_json, created_at
        FROM research_signals
        ORDER BY
            CASE WHEN recency_days_min IS NULL THEN 1 ELSE 0 END DESC,
            recency_days_min DESC,
            item_count_30d ASC,
            created_at DESC
        LIMIT 200
        """
    ):
        try:
            flags = json.loads(row["sentiment_flags_json"] or "[]")
        except ValueError:
            flags = []
        try:
            summary = json.loads(row["summary_json"] or "{}")
        except ValueError:
            summary = {}
        research_gaps.append(
            {
                "ticker": row["ticker"],
                "as_of_date": row["as_of_date"],
                "run_id": row["run_id"],
                "recency_days_min": row["recency_days_min"],
                "item_count_30d": row["item_count_30d"],
                "risk_flags": [str(flag) for flag in flags] if isinstance(flags, list) else [],
                "freshness_bucket": str(summary.get("freshness_bucket", "UNKNOWN")),
                "flow_bucket": str(summary.get("flow_bucket", "UNKNOWN")),
                "created_at": row["created_at"],
            }
        )

    from app.ops.runs import list_runs

    legacy_runs = [
        {
            "run_id": str(entry.get("run_id") or ""),
            "as_of_date": entry.get("as_of_date"),
            "generated_at": entry.get("generated_at"),
            "status": entry.get("status"),
        }
        for entry in list_runs(limit=50)
        if isinstance(entry, dict)
    ]

    return {
        "backlog_size": backlog,
        "deadletters": deadletters,
        "research_gaps": research_gaps,
        "legacy_runs": legacy_runs,
    }
