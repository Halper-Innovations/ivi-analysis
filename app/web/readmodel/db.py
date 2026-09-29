"""Read-only access to the books of record (``data/engine.db``).

The web UI never writes ``engine.db``. This module is the only place
the web layer opens it, and the connection is read-only at the SQLite level
(URI ``mode=ro`` plus ``PRAGMA query_only``) so even a bug cannot write the
books of record. UI-owned state lives in a separate database (see
``runs_index``).
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from app.config import AppConfig, get_config


class OfflineError(RuntimeError):
    """A named precondition for serving the UI is not met.

    ``precondition`` is a stable machine-readable code (mirroring the
    ``ivi ops preflight`` posture of naming the exact failed check) so the
    frontend can render a designed empty state instead of a stack trace.
    """

    def __init__(self, precondition: str, detail: str) -> None:
        super().__init__(f"{precondition}: {detail}")
        self.precondition = precondition
        self.detail = detail


def resolve_engine_db_path(
    db_path: str | Path | None = None, *, cfg: AppConfig | None = None
) -> Path:
    if db_path is not None:
        return Path(db_path)
    cfg = cfg or get_config()
    return Path(cfg.db_path)


def open_readonly(
    db_path: str | Path | None = None, *, cfg: AppConfig | None = None
) -> sqlite3.Connection:
    """Open ``engine.db`` strictly read-only.

    Raises :class:`OfflineError` (``engine_db_missing``) when the
    database file does not exist — e.g. the data volume is unmounted — so
    callers surface the precondition instead of implicitly creating an empty
    database, which is what a plain ``sqlite3.connect`` would do.
    """

    cfg = cfg or get_config()
    path = resolve_engine_db_path(db_path, cfg=cfg)
    if not path.exists():
        raise OfflineError("engine_db_missing", str(path))
    conn = sqlite3.connect(
        f"file:{path}?mode=ro",
        uri=True,
        timeout=max(1.0, float(cfg.sqlite_busy_timeout_ms) / 1000.0),
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout={int(cfg.sqlite_busy_timeout_ms)};")
    # Second lock on the same door: mode=ro already rejects writes; query_only
    # also rejects them at statement level if a future change loosens the URI.
    conn.execute("PRAGMA query_only=ON;")
    return conn


@contextmanager
def readonly_db(
    db_path: str | Path | None = None, *, cfg: AppConfig | None = None
) -> Iterator[sqlite3.Connection]:
    conn = open_readonly(db_path, cfg=cfg)
    try:
        yield conn
    finally:
        conn.close()


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    """Whether ``name`` exists in the connected database.

    ``init_db`` creates every table the read model queries, but a database
    created before that change (or a hand-copied one) may still lack the
    watchlist family. Readers treat an absent table as "not yet populated"
    and return their empty state; they never create it (this layer is
    read-only) and never let ``no such table`` surface as a 500.
    """

    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
        ).fetchone()
        is not None
    )
