"""Operational preflight guards.

Every heartbeat and cache-touching CLI entry point calls run_preflight()
before doing work. The checks are the platform's known silent-failure
modes, inverted into hard aborts:

- engine.db must exist at the configured absolute path with sane row
  counts — a wrong-cwd or misconfigured invocation otherwise creates an
  empty shadow DB that renders as "no names at target".
- data/cache and data/raw_filings must resolve — they are symlinks onto
  an external volume; unmounted, a census run would mass-downgrade the
  registrant universe and cache-first loaders would see nothing.

Row floors are read through a read-only URI connection so the preflight
itself can never create or mutate the file it is guarding.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config import AppConfig, check_data_volume, get_config
from app.logging import get_logger

logger = get_logger(__name__)

# Floors are deliberately far below live values (watchlist ~1.5k rows,
# companyfacts_facts ~3.2M) — they exist to catch empty/shadow DBs, not to
# police normal variance.
MIN_WATCHLIST_ROWS = 100
MIN_COMPANYFACTS_ROWS = 1_000_000


class PreflightError(RuntimeError):
    """Raised by assert_preflight when any check fails."""


@dataclass
class PreflightCheck:
    name: str
    ok: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail}


@dataclass
class PreflightResult:
    checks: list[PreflightCheck] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks)

    @property
    def failures(self) -> list[PreflightCheck]:
        return [check for check in self.checks if not check.ok]

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "checks": [check.to_dict() for check in self.checks]}


def _check(checks: list[PreflightCheck], name: str, ok: bool, detail: str) -> None:
    checks.append(PreflightCheck(name=name, ok=bool(ok), detail=detail))


def _dir_resolves(path: Path) -> tuple[bool, str]:
    """True when the path (following symlinks) is an existing directory.

    A symlink whose target volume is unmounted fails is_dir() while the
    symlink itself still lexists — exactly the condition to catch.
    """
    if path.is_dir():
        target = path.resolve()
        return True, f"resolves to {target}"
    if path.is_symlink():
        return False, f"symlink target unreachable (volume unmounted?): {path} -> {path.readlink()}"
    return False, f"missing: {path}"


def _row_count_readonly(db_path: Path, table: str) -> int | None:
    """Row count preferring a read-only URI connection; None when unreadable.

    Falls back to a plain connection when mode=ro cannot open (a WAL DB
    without a -shm file rejects read-only opens) — the caller only invokes
    this on an existing file, so the fallback cannot create one.
    """
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10.0)
    except sqlite3.Error:
        if db_path.is_file():
            try:
                conn = sqlite3.connect(str(db_path), timeout=10.0)
            except sqlite3.Error:
                return None
        else:
            return None
    try:
        row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        return int(row[0]) if row else None
    except sqlite3.Error:
        return None
    finally:
        conn.close()


def _row_count_connection(conn: sqlite3.Connection, table: str) -> int | None:
    """Read a row count from an already-pinned caller transaction."""

    try:
        row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        return int(row[0]) if row else None
    except sqlite3.Error:
        return None


def run_preflight(
    cfg: AppConfig | None = None,
    *,
    require_engine: bool = True,
    require_cache_dirs: bool = True,
    min_watchlist_rows: int = MIN_WATCHLIST_ROWS,
    min_companyfacts_rows: int = MIN_COMPANYFACTS_ROWS,
    db_conn: sqlite3.Connection | None = None,
) -> PreflightResult:
    cfg = cfg or get_config()
    checks: list[PreflightCheck] = []

    volume_ok, volume_detail = check_data_volume(cfg)
    _check(checks, "data_volume_mounted", volume_ok, volume_detail)

    db_path = Path(cfg.db_path)
    _check(
        checks,
        "db_path_absolute",
        db_path.is_absolute(),
        str(db_path),
    )

    if require_engine:
        exists = db_path.is_file()
        size = db_path.stat().st_size if exists else 0
        _check(
            checks,
            "engine_db_exists",
            exists and size > 0,
            f"{db_path} ({size} bytes)" if exists else f"missing: {db_path}",
        )
        if exists and size > 0:
            row_count = (
                (lambda table: _row_count_connection(db_conn, table))
                if db_conn is not None
                else (lambda table: _row_count_readonly(db_path, table))
            )
            wl = row_count("watchlist")
            cf = row_count("companyfacts_facts")
            _check(
                checks,
                "engine_db_row_floor_watchlist",
                wl is not None and wl >= min_watchlist_rows,
                f"watchlist={wl} (floor {min_watchlist_rows})",
            )
            _check(
                checks,
                "engine_db_row_floor_companyfacts",
                cf is not None and cf >= min_companyfacts_rows,
                f"companyfacts_facts={cf} (floor {min_companyfacts_rows})",
            )

    if require_cache_dirs:
        for name, path in (
            ("cache_dir_resolves", Path(cfg.cache_dir)),
            ("raw_filings_resolves", Path(cfg.raw_filings_dir)),
        ):
            ok, detail = _dir_resolves(path)
            _check(checks, name, ok, detail)

    result = PreflightResult(checks=checks)
    if not result.ok:
        for failure in result.failures:
            logger.error(
                "preflight_check_failed",
                extra={"stage_name": failure.name, "stage_detail": failure.detail},
            )
    return result


def assert_preflight(
    cfg: AppConfig | None = None,
    *,
    require_engine: bool = True,
    require_cache_dirs: bool = True,
) -> PreflightResult:
    """run_preflight, raising PreflightError with every failure listed."""
    result = run_preflight(
        cfg,
        require_engine=require_engine,
        require_cache_dirs=require_cache_dirs,
    )
    if not result.ok:
        summary = "; ".join(f"{c.name}: {c.detail}" for c in result.failures)
        raise PreflightError(f"preflight failed — {summary}")
    return result


__all__ = [
    "MIN_COMPANYFACTS_ROWS",
    "MIN_WATCHLIST_ROWS",
    "PreflightCheck",
    "PreflightError",
    "PreflightResult",
    "assert_preflight",
    "run_preflight",
]
