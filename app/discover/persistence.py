"""SQLite persistence for discover sweep sessions.

Each sweep is a row in the sweeps table; Stage 2/3/4 per-ticker results
hang off the sweep via (sweep_id, ticker) composite keys. The schema
is defined inline so the module is self-contained and the session DB
can live alongside but independent of engine.db.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.autonomous.v1_financial_context import BoundV1FinancialScope


SESSION_TABLES = (
    "sweeps",
    "stage2_results",
    "stage3_results",
    "stage4_results",
    "sweep_universe",
    "discover_paid_attempts",
    "discover_publication_authorizations",
)


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sweeps (
    sweep_id          TEXT PRIMARY KEY,
    started_at        TEXT NOT NULL,
    finished_at       TEXT,
    universe_size     INTEGER NOT NULL,
    limit_applied     INTEGER,
    budget_usd        REAL,
    total_cost_usd    REAL DEFAULT 0.0,
    status            TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS stage2_results (
    sweep_id          TEXT NOT NULL,
    ticker            TEXT NOT NULL,
    decision          TEXT NOT NULL,
    confidence        TEXT NOT NULL,
    reason            TEXT,
    input_tokens      INTEGER NOT NULL,
    output_tokens     INTEGER NOT NULL,
    cost_usd          REAL NOT NULL,
    wall_ms           INTEGER NOT NULL,
    error             TEXT,
    financial_scope_fingerprint TEXT,
    financial_scope_publication_fingerprint TEXT,
    publication_evidence_json TEXT,
    paid_attempts_sha256 TEXT,
    publication_row_sha256 TEXT,
    PRIMARY KEY (sweep_id, ticker),
    FOREIGN KEY (sweep_id) REFERENCES sweeps(sweep_id)
);

CREATE TABLE IF NOT EXISTS stage3_results (
    sweep_id          TEXT NOT NULL,
    ticker            TEXT NOT NULL,
    verdict           TEXT NOT NULL,
    confidence        TEXT NOT NULL,
    thesis_summary    TEXT NOT NULL,
    key_numbers_json  TEXT NOT NULL,
    positives_json    TEXT NOT NULL,
    risks_json        TEXT NOT NULL,
    open_questions_json TEXT NOT NULL,
    reasoning_trace   TEXT NOT NULL,
    input_tokens      INTEGER NOT NULL,
    output_tokens     INTEGER NOT NULL,
    cost_usd          REAL NOT NULL,
    wall_ms           INTEGER NOT NULL,
    error             TEXT,
    financial_scope_fingerprint TEXT,
    financial_scope_publication_fingerprint TEXT,
    publication_evidence_json TEXT,
    paid_attempts_sha256 TEXT,
    publication_row_sha256 TEXT,
    PRIMARY KEY (sweep_id, ticker),
    FOREIGN KEY (sweep_id) REFERENCES sweeps(sweep_id)
);

CREATE TABLE IF NOT EXISTS stage4_results (
    sweep_id          TEXT NOT NULL,
    ticker            TEXT NOT NULL,
    verdict           TEXT NOT NULL,
    confidence        TEXT NOT NULL,
    thesis            TEXT NOT NULL,
    key_findings_json TEXT NOT NULL,
    open_questions_json TEXT NOT NULL,
    falsifiers_json   TEXT NOT NULL,
    reasoning_trace   TEXT NOT NULL,
    num_turns         INTEGER NOT NULL,
    tool_call_counts_json TEXT NOT NULL,
    termination_reason TEXT NOT NULL,
    input_tokens      INTEGER NOT NULL,
    output_tokens     INTEGER NOT NULL,
    cost_usd          REAL NOT NULL,
    wall_seconds      REAL NOT NULL,
    error             TEXT,
    financial_scope_fingerprint TEXT,
    financial_scope_publication_fingerprint TEXT,
    financial_scope_manifest_json TEXT,
    publication_evidence_json TEXT,
    paid_attempts_sha256 TEXT,
    publication_row_sha256 TEXT,
    PRIMARY KEY (sweep_id, ticker),
    FOREIGN KEY (sweep_id) REFERENCES sweeps(sweep_id)
);

CREATE TABLE IF NOT EXISTS sweep_universe (
    sweep_id      TEXT NOT NULL,
    ticker        TEXT NOT NULL,
    as_of_date    TEXT NOT NULL,
    PRIMARY KEY (sweep_id, ticker),
    FOREIGN KEY (sweep_id) REFERENCES sweeps(sweep_id)
);

CREATE TABLE IF NOT EXISTS discover_paid_attempts (
    stage                  INTEGER NOT NULL CHECK(stage IN (2, 3, 4)),
    sweep_id               TEXT NOT NULL,
    ticker                 TEXT NOT NULL,
    execution_id           TEXT NOT NULL,
    attempt_number         INTEGER NOT NULL,
    request_sha256         TEXT NOT NULL,
    provider               TEXT NOT NULL,
    model                  TEXT NOT NULL,
    status                 TEXT NOT NULL CHECK(status IN ('RESERVED', 'RETURNED', 'FAILED', 'PUBLISHED')),
    outcome                TEXT CHECK(outcome IN ('SUCCESS', 'ERROR')),
    estimated_cost_usd     REAL NOT NULL,
    input_tokens           INTEGER NOT NULL DEFAULT 0,
    output_tokens          INTEGER NOT NULL DEFAULT 0,
    accounted_cost_usd     REAL NOT NULL DEFAULT 0,
    cost_integrity_violation INTEGER NOT NULL DEFAULT 0
                              CHECK(cost_integrity_violation IN (0, 1)),
    error                  TEXT,
    reserved_at            TEXT NOT NULL,
    completed_at           TEXT,
    published_at           TEXT,
    publication_row_sha256 TEXT,
    PRIMARY KEY (stage, sweep_id, ticker, attempt_number),
    FOREIGN KEY (sweep_id) REFERENCES sweeps(sweep_id)
);

CREATE TRIGGER IF NOT EXISTS discover_paid_attempts_no_delete
BEFORE DELETE ON discover_paid_attempts
BEGIN
    SELECT RAISE(ABORT, 'discover paid attempts cannot be deleted');
END;

CREATE TRIGGER IF NOT EXISTS discover_paid_attempts_published_no_update
BEFORE UPDATE ON discover_paid_attempts
WHEN OLD.status = 'PUBLISHED'
BEGIN
    SELECT RAISE(ABORT, 'published discover paid attempts are immutable');
END;

CREATE TABLE IF NOT EXISTS discover_publication_authorizations (
    authorization_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    stage                  INTEGER NOT NULL CHECK(stage IN (2, 3, 4)),
    sweep_id               TEXT NOT NULL,
    ticker                 TEXT NOT NULL,
    publication_row_sha256 TEXT NOT NULL,
    financial_scope_fingerprint TEXT NOT NULL,
    publication_evidence_sha256 TEXT NOT NULL,
    authorized_at          TEXT NOT NULL,
    UNIQUE (
        stage, sweep_id, ticker, publication_row_sha256,
        financial_scope_fingerprint, publication_evidence_sha256
    ),
    FOREIGN KEY (sweep_id) REFERENCES sweeps(sweep_id)
);

CREATE TRIGGER IF NOT EXISTS discover_publication_authorizations_no_update
BEFORE UPDATE ON discover_publication_authorizations
BEGIN
    SELECT RAISE(ABORT, 'discover publication authorizations are append-only');
END;

CREATE TRIGGER IF NOT EXISTS discover_publication_authorizations_no_delete
BEFORE DELETE ON discover_publication_authorizations
BEGIN
    SELECT RAISE(ABORT, 'discover publication authorizations are append-only');
END;
"""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_schema(db_path: str | Path) -> None:
    """Create the discover session tables if they do not exist."""
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(_SCHEMA_SQL)
        for table_name in (
            "stage2_results",
            "stage3_results",
            "stage4_results",
        ):
            columns = {
                str(row[1]) for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
            }
            if "financial_scope_fingerprint" not in columns:
                conn.execute(
                    f"ALTER TABLE {table_name} ADD COLUMN financial_scope_fingerprint TEXT"
                )
            if "financial_scope_publication_fingerprint" not in columns:
                conn.execute(
                    f"ALTER TABLE {table_name} "
                    "ADD COLUMN financial_scope_publication_fingerprint TEXT"
                )
            if "publication_evidence_json" not in columns:
                conn.execute(f"ALTER TABLE {table_name} ADD COLUMN publication_evidence_json TEXT")
            if "publication_row_sha256" not in columns:
                conn.execute(f"ALTER TABLE {table_name} ADD COLUMN publication_row_sha256 TEXT")
            if "paid_attempts_sha256" not in columns:
                conn.execute(f"ALTER TABLE {table_name} ADD COLUMN paid_attempts_sha256 TEXT")
            if table_name == "stage4_results" and "financial_scope_manifest_json" not in columns:
                conn.execute(
                    "ALTER TABLE stage4_results ADD COLUMN financial_scope_manifest_json TEXT"
                )
        paid_attempt_columns = {
            str(row[1])
            for row in conn.execute("PRAGMA table_info(discover_paid_attempts)").fetchall()
        }
        if "cost_integrity_violation" not in paid_attempt_columns:
            conn.execute(
                "ALTER TABLE discover_paid_attempts "
                "ADD COLUMN cost_integrity_violation INTEGER NOT NULL DEFAULT 0 "
                "CHECK(cost_integrity_violation IN (0, 1))"
            )
        conn.commit()
    finally:
        conn.close()


def _connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PUBLICATION_EVIDENCE_SCHEMA = "discover_publication_evidence_v1"
_PAID_ATTEMPTS_SCHEMA = "discover_paid_attempts_v1"
_STAGE_TABLES = {
    2: "stage2_results",
    3: "stage3_results",
    4: "stage4_results",
}
_STAGE_JSON_FIELDS = {
    2: frozenset({"publication_evidence_json"}),
    3: frozenset(
        {
            "key_numbers_json",
            "positives_json",
            "risks_json",
            "open_questions_json",
            "publication_evidence_json",
        }
    ),
    4: frozenset(
        {
            "key_findings_json",
            "open_questions_json",
            "falsifiers_json",
            "tool_call_counts_json",
            "financial_scope_manifest_json",
            "publication_evidence_json",
        }
    ),
}
_STAGE_PUBLICATION_FIELDS = {
    2: (
        "sweep_id",
        "ticker",
        "decision",
        "confidence",
        "reason",
        "input_tokens",
        "output_tokens",
        "cost_usd",
        "wall_ms",
        "error",
        "financial_scope_fingerprint",
        "financial_scope_publication_fingerprint",
        "publication_evidence_json",
        "paid_attempts_sha256",
    ),
    3: (
        "sweep_id",
        "ticker",
        "verdict",
        "confidence",
        "thesis_summary",
        "key_numbers_json",
        "positives_json",
        "risks_json",
        "open_questions_json",
        "reasoning_trace",
        "input_tokens",
        "output_tokens",
        "cost_usd",
        "wall_ms",
        "error",
        "financial_scope_fingerprint",
        "financial_scope_publication_fingerprint",
        "publication_evidence_json",
        "paid_attempts_sha256",
    ),
    4: (
        "sweep_id",
        "ticker",
        "verdict",
        "confidence",
        "thesis",
        "key_findings_json",
        "open_questions_json",
        "falsifiers_json",
        "reasoning_trace",
        "num_turns",
        "tool_call_counts_json",
        "termination_reason",
        "input_tokens",
        "output_tokens",
        "cost_usd",
        "wall_seconds",
        "error",
        "financial_scope_fingerprint",
        "financial_scope_publication_fingerprint",
        "financial_scope_manifest_json",
        "publication_evidence_json",
        "paid_attempts_sha256",
    ),
}
_STAGE_PRIMARY_EVIDENCE_NAMES = {
    2: "stage2_provider_request",
    3: "stage3_provider_request",
    4: "stage4_provider_context",
}
_STAGE4_EVIDENCE_TOOL_NAMES = frozenset(
    {
        "fetch_current_price",
        "fetch_filing_section",
        "fetch_companyfacts",
        "fetch_historical_scorecards",
    }
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


class DiscoverPaidAttemptAmbiguousError(RuntimeError):
    """A prior physical provider attempt cannot be safely repeated."""


class DiscoverCostBudgetExceeded(RuntimeError):
    """A Discover provider call cannot be reserved inside its hard cost ceiling."""


class DiscoverCostIntegrityError(RuntimeError):
    """Provider-reported usage exceeded the conservative durable reservation."""


def _paid_attempt_payload(rows: list[Any]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Discover publication requires at least one paid attempt")
    first = rows[0]
    attempts: list[dict[str, Any]] = []
    for row in rows:
        if int(row["cost_integrity_violation"] or 0) != 0:
            raise DiscoverCostIntegrityError(
                "Discover paid-attempt history contains a cost-integrity overrun "
                "and cannot authorize publication"
            )
        if str(row["status"]) not in {"RETURNED", "FAILED", "PUBLISHED"}:
            raise DiscoverPaidAttemptAmbiguousError(
                "Discover paid attempt is still RESERVED; physical-call outcome is ambiguous"
            )
        outcome = str(row["outcome"] or "")
        if outcome not in {"SUCCESS", "ERROR"}:
            raise ValueError("Discover paid attempt is missing a terminal outcome")
        attempts.append(
            {
                "attempt_number": int(row["attempt_number"]),
                "request_sha256": str(row["request_sha256"]),
                "provider": str(row["provider"]),
                "model": str(row["model"]),
                "outcome": outcome,
                "estimated_cost_usd": float(row["estimated_cost_usd"]),
                "input_tokens": int(row["input_tokens"]),
                "output_tokens": int(row["output_tokens"]),
                "accounted_cost_usd": float(row["accounted_cost_usd"]),
                "error": str(row["error"]) if row["error"] is not None else None,
            }
        )
    return {
        "schema": _PAID_ATTEMPTS_SCHEMA,
        "stage": int(first["stage"]),
        "sweep_id": str(first["sweep_id"]),
        "ticker": str(first["ticker"]).strip().upper(),
        "attempts": attempts,
    }


def _paid_attempt_rows(
    conn: sqlite3.Connection,
    *,
    stage: int,
    sweep_id: str,
    ticker: str,
) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT *
        FROM discover_paid_attempts
        WHERE stage = ? AND sweep_id = ? AND ticker = ?
        ORDER BY attempt_number
        """,
        (int(stage), str(sweep_id), str(ticker).strip().upper()),
    ).fetchall()


def _paid_attempt_summary_from_rows(rows: list[Any]) -> dict[str, Any]:
    payload = _paid_attempt_payload(rows)
    attempts = payload["attempts"]
    return {
        "attempt_count": len(attempts),
        "input_tokens": sum(int(item["input_tokens"]) for item in attempts),
        "output_tokens": sum(int(item["output_tokens"]) for item in attempts),
        "cost_usd": sum(float(item["accounted_cost_usd"]) for item in attempts),
        "paid_attempts_sha256": _canonical_sha256(payload),
        "payload": payload,
    }


def require_no_prior_discover_paid_attempt(
    db_path: str | Path,
    *,
    stage: int,
    sweep_id: str,
    ticker: str,
) -> None:
    """Fail closed before resume can repeat any prior physical attempt."""

    conn = _connect(db_path)
    try:
        rows = _paid_attempt_rows(
            conn,
            stage=stage,
            sweep_id=sweep_id,
            ticker=ticker,
        )
    finally:
        conn.close()
    if rows:
        statuses = ",".join(str(row["status"]) for row in rows)
        raise DiscoverPaidAttemptAmbiguousError(
            f"Discover Stage {int(stage)} {sweep_id}/{str(ticker).strip().upper()} "
            f"has durable paid-attempt history ({statuses}) without an authorized result; "
            "resume is blocked to prevent duplicate spend"
        )


def _reserve_discover_paid_attempt(
    db_path: str | Path,
    *,
    stage: int,
    sweep_id: str,
    ticker: str,
    execution_id: str,
    request_payload: dict[str, Any],
    provider: str,
    model: str,
    estimated_cost_usd: float,
    hard_ticker_cost_limit_usd: float | None = None,
) -> int:
    normalized_ticker = str(ticker).strip().upper()
    reservation_cost_usd = max(0.0, float(estimated_cost_usd))
    conn = _connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        sweep = conn.execute(
            """
            SELECT budget_usd, COALESCE(total_cost_usd, 0.0) AS total_cost_usd
            FROM sweeps
            WHERE sweep_id = ?
            """,
            (str(sweep_id),),
        ).fetchone()
        if sweep is None:
            raise ValueError(f"Discover sweep does not exist: {sweep_id}")
        sweep_budget_usd = (
            max(0.0, float(sweep["budget_usd"])) if sweep["budget_usd"] is not None else None
        )
        sweep_total_cost_usd = max(0.0, float(sweep["total_cost_usd"]))
        if (
            sweep_budget_usd is not None
            and sweep_total_cost_usd + reservation_cost_usd > sweep_budget_usd + 1e-12
        ):
            raise DiscoverCostBudgetExceeded(
                f"Discover sweep {sweep_id} hard cost cap would be exceeded: "
                f"${sweep_total_cost_usd:.6f} accounted/reserved + "
                f"${reservation_cost_usd:.6f} requested > ${sweep_budget_usd:.6f}"
            )
        if hard_ticker_cost_limit_usd is not None:
            ticker_cost_row = conn.execute(
                """
                SELECT COALESCE(SUM(
                    CASE
                        WHEN status = 'RESERVED' THEN estimated_cost_usd
                        ELSE accounted_cost_usd
                    END
                ), 0.0) AS ticker_cost_usd
                FROM discover_paid_attempts
                WHERE stage = ? AND sweep_id = ? AND ticker = ?
                """,
                (int(stage), str(sweep_id), normalized_ticker),
            ).fetchone()
            ticker_cost_usd = max(0.0, float(ticker_cost_row["ticker_cost_usd"]))
            hard_limit_usd = max(0.0, float(hard_ticker_cost_limit_usd))
            if ticker_cost_usd + reservation_cost_usd > hard_limit_usd + 1e-12:
                raise DiscoverCostBudgetExceeded(
                    f"Discover Stage {int(stage)} {sweep_id}/{normalized_ticker} hard "
                    f"cost cap would be exceeded: ${ticker_cost_usd:.6f} "
                    f"accounted/reserved + ${reservation_cost_usd:.6f} requested > "
                    f"${hard_limit_usd:.6f}"
                )
        existing = _paid_attempt_rows(
            conn,
            stage=stage,
            sweep_id=sweep_id,
            ticker=normalized_ticker,
        )
        if existing:
            execution_ids = {str(row["execution_id"]) for row in existing}
            if execution_ids != {str(execution_id)}:
                raise DiscoverPaidAttemptAmbiguousError(
                    f"Discover Stage {int(stage)} {sweep_id}/{normalized_ticker} "
                    "already has a paid attempt from another execution"
                )
            if str(existing[-1]["status"]) != "RETURNED":
                raise DiscoverPaidAttemptAmbiguousError(
                    f"Discover Stage {int(stage)} {sweep_id}/{normalized_ticker} "
                    f"cannot start another call after {existing[-1]['status']}"
                )
        attempt_number = len(existing) + 1
        conn.execute(
            """
            INSERT INTO discover_paid_attempts (
                stage, sweep_id, ticker, execution_id, attempt_number,
                request_sha256, provider, model, status, estimated_cost_usd,
                reserved_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'RESERVED', ?, ?)
            """,
            (
                int(stage),
                str(sweep_id),
                normalized_ticker,
                str(execution_id),
                attempt_number,
                _canonical_sha256(request_payload),
                str(provider),
                str(model),
                reservation_cost_usd,
                _utc_now_iso(),
            ),
        )
        sweep_cursor = conn.execute(
            """
            UPDATE sweeps
            SET total_cost_usd = COALESCE(total_cost_usd, 0.0) + ?
            WHERE sweep_id = ?
            """,
            (
                reservation_cost_usd,
                str(sweep_id),
            ),
        )
        if sweep_cursor.rowcount != 1:
            raise ValueError(f"Discover sweep does not exist: {sweep_id}")
        conn.commit()
        return attempt_number
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def _complete_discover_paid_attempt(
    db_path: str | Path,
    *,
    stage: int,
    sweep_id: str,
    ticker: str,
    execution_id: str,
    attempt_number: int,
    outcome: str,
    input_tokens: int,
    output_tokens: int,
    accounted_cost_usd: float,
    error: str | None,
) -> None:
    normalized_outcome = str(outcome).upper()
    if normalized_outcome not in {"SUCCESS", "ERROR"}:
        raise ValueError(f"unsupported Discover paid-attempt outcome: {outcome}")
    conn = _connect(db_path)
    reservation_overrun: tuple[float, float] | None = None
    try:
        conn.execute("BEGIN IMMEDIATE")
        reservation = conn.execute(
            """
            SELECT estimated_cost_usd
            FROM discover_paid_attempts
            WHERE stage = ? AND sweep_id = ? AND ticker = ?
              AND execution_id = ? AND attempt_number = ? AND status = 'RESERVED'
            """,
            (
                int(stage),
                str(sweep_id),
                str(ticker).strip().upper(),
                str(execution_id),
                int(attempt_number),
            ),
        ).fetchone()
        if reservation is None:
            raise DiscoverPaidAttemptAmbiguousError(
                "Discover paid-attempt reservation changed before completion"
            )
        settled_cost_usd = max(0.0, float(accounted_cost_usd))
        reserved_cost_usd = max(0.0, float(reservation["estimated_cost_usd"]))
        if settled_cost_usd > reserved_cost_usd + 1e-12:
            reservation_overrun = (reserved_cost_usd, settled_cost_usd)
        cost_integrity_violation = reservation_overrun is not None
        completion_error = str(error)[:500] if error else None
        if cost_integrity_violation:
            completion_error = (
                "provider-reported cost exceeded conservative reservation: "
                f"${settled_cost_usd:.6f} actual > ${reserved_cost_usd:.6f} reserved"
            )
        cursor = conn.execute(
            """
            UPDATE discover_paid_attempts
            SET status = ?, outcome = ?, input_tokens = ?, output_tokens = ?,
                accounted_cost_usd = ?, cost_integrity_violation = ?,
                error = ?, completed_at = ?
            WHERE stage = ? AND sweep_id = ? AND ticker = ?
              AND execution_id = ? AND attempt_number = ? AND status = 'RESERVED'
            """,
            (
                (
                    "FAILED"
                    if cost_integrity_violation
                    else ("RETURNED" if normalized_outcome == "SUCCESS" else "FAILED")
                ),
                "ERROR" if cost_integrity_violation else normalized_outcome,
                max(0, int(input_tokens)),
                max(0, int(output_tokens)),
                settled_cost_usd,
                1 if cost_integrity_violation else 0,
                completion_error,
                _utc_now_iso(),
                int(stage),
                str(sweep_id),
                str(ticker).strip().upper(),
                str(execution_id),
                int(attempt_number),
            ),
        )
        if cursor.rowcount != 1:
            raise DiscoverPaidAttemptAmbiguousError(
                "Discover paid-attempt reservation changed before completion"
            )
        sweep_cursor = conn.execute(
            """
            UPDATE sweeps
            SET total_cost_usd = COALESCE(total_cost_usd, 0.0) + ?
            WHERE sweep_id = ?
            """,
            (
                settled_cost_usd - float(reservation["estimated_cost_usd"]),
                str(sweep_id),
            ),
        )
        if sweep_cursor.rowcount != 1:
            raise ValueError(f"Discover sweep does not exist: {sweep_id}")
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()
    if reservation_overrun is not None:
        reserved_cost_usd, settled_cost_usd = reservation_overrun
        raise DiscoverCostIntegrityError(
            "Discover provider-reported usage exceeded its conservative durable "
            f"reservation: ${settled_cost_usd:.6f} actual > "
            f"${reserved_cost_usd:.6f} reserved"
        )


def discover_paid_attempt_summary(
    db_path: str | Path,
    *,
    stage: int,
    sweep_id: str,
    ticker: str,
) -> dict[str, Any]:
    conn = _connect(db_path)
    try:
        rows = _paid_attempt_rows(
            conn,
            stage=stage,
            sweep_id=sweep_id,
            ticker=ticker,
        )
        return _paid_attempt_summary_from_rows(rows)
    finally:
        conn.close()


class _DurableDiscoverMessages:
    def __init__(self, owner: "DurableDiscoverClient", messages: Any) -> None:
        self._owner = owner
        self._messages = messages

    def __getattr__(self, name: str) -> Any:
        return getattr(self._messages, name)

    def create(self, **kwargs: Any) -> Any:
        return self._owner._create(self._messages.create, kwargs)


class DurableDiscoverClient:
    """Proxy that durably reserves and settles every physical Discover call."""

    def __init__(
        self,
        client: Any,
        *,
        db_path: str | Path,
        stage: int,
        sweep_id: str,
        ticker: str,
        input_usd_per_mtok: float,
        output_usd_per_mtok: float,
        hard_ticker_cost_limit_usd: float | None = None,
    ) -> None:
        declared_max_retries = getattr(client, "max_retries", None)
        if declared_max_retries is not None:
            try:
                normalized_max_retries = int(declared_max_retries)
            except (TypeError, ValueError) as exc:
                raise ValueError("Discover client max_retries must be explicitly zero") from exc
            if normalized_max_retries != 0:
                raise ValueError(
                    "Discover client max_retries must be zero so one durable "
                    "reservation represents exactly one physical provider attempt"
                )
        self._client = client
        self._db_path = db_path
        self._stage = int(stage)
        self._sweep_id = str(sweep_id)
        self._ticker = str(ticker).strip().upper()
        self._input_usd_per_mtok = max(0.0, float(input_usd_per_mtok))
        self._output_usd_per_mtok = max(0.0, float(output_usd_per_mtok))
        self._hard_ticker_cost_limit_usd = (
            max(0.0, float(hard_ticker_cost_limit_usd))
            if hard_ticker_cost_limit_usd is not None
            else None
        )
        self._execution_id = secrets.token_hex(16)
        self.messages = _DurableDiscoverMessages(self, client.messages)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    def _create(self, create: Any, kwargs: dict[str, Any]) -> Any:
        request_payload = json.loads(_canonical_json(kwargs))
        serialized = _canonical_json(request_payload)
        # UTF-8 bytes are a conservative token upper bound for byte-pair
        # tokenizers; the fixed margin covers provider envelope/special tokens.
        estimated_input_tokens = max(1, len(serialized.encode("utf-8")) + 1024)
        max_output_tokens = max(0, int(kwargs.get("max_tokens") or 0))
        estimated_cost_usd = (
            estimated_input_tokens / 1_000_000 * self._input_usd_per_mtok
            + max_output_tokens / 1_000_000 * self._output_usd_per_mtok
        )
        attempt_number = _reserve_discover_paid_attempt(
            self._db_path,
            stage=self._stage,
            sweep_id=self._sweep_id,
            ticker=self._ticker,
            execution_id=self._execution_id,
            request_payload=request_payload,
            provider="anthropic",
            model=str(kwargs.get("model") or ""),
            estimated_cost_usd=estimated_cost_usd,
            hard_ticker_cost_limit_usd=self._hard_ticker_cost_limit_usd,
        )
        try:
            response = create(**kwargs)
        except BaseException as exc:
            _complete_discover_paid_attempt(
                self._db_path,
                stage=self._stage,
                sweep_id=self._sweep_id,
                ticker=self._ticker,
                execution_id=self._execution_id,
                attempt_number=attempt_number,
                outcome="ERROR",
                input_tokens=0,
                output_tokens=0,
                accounted_cost_usd=estimated_cost_usd,
                error=str(exc),
            )
            raise
        usage = getattr(response, "usage", None)
        input_tokens = max(0, int(getattr(usage, "input_tokens", 0) or 0))
        output_tokens = max(0, int(getattr(usage, "output_tokens", 0) or 0))
        actual_cost_usd = (
            input_tokens / 1_000_000 * self._input_usd_per_mtok
            + output_tokens / 1_000_000 * self._output_usd_per_mtok
        )
        _complete_discover_paid_attempt(
            self._db_path,
            stage=self._stage,
            sweep_id=self._sweep_id,
            ticker=self._ticker,
            execution_id=self._execution_id,
            attempt_number=attempt_number,
            outcome="SUCCESS",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            accounted_cost_usd=actual_cost_usd,
            error=None,
        )
        return response


def _strict_stage4_tool_manifest(
    manifest: Any,
    *,
    ticker: str,
) -> list[dict[str, Any]] | None:
    """Return one canonical Stage 4 tool manifest or fail closed."""

    if not isinstance(manifest, list):
        return None
    normalized_ticker = str(ticker).strip().upper()
    canonical: list[dict[str, Any]] = []
    seen: set[tuple[int, str, str, str]] = set()
    previous_turn = 0
    for item in manifest:
        if not isinstance(item, dict) or set(item) != {
            "turn",
            "tool",
            "input",
            "output_sha256",
        }:
            return None
        turn = item["turn"]
        tool_name = item["tool"]
        tool_input = item["input"]
        output_sha256 = str(item["output_sha256"] or "").strip().lower()
        if (
            not isinstance(turn, int)
            or isinstance(turn, bool)
            or turn < 1
            or turn < previous_turn
            or tool_name not in _STAGE4_EVIDENCE_TOOL_NAMES
            or not isinstance(tool_input, dict)
            or _SHA256_RE.fullmatch(output_sha256) is None
        ):
            return None
        requested_ticker = str(tool_input.get("ticker") or "").strip().upper()
        if requested_ticker != normalized_ticker:
            return None
        try:
            canonical_input = json.loads(_canonical_json(tool_input))
        except (TypeError, ValueError):
            return None
        identity = (
            turn,
            str(tool_name),
            _canonical_json(canonical_input),
            output_sha256,
        )
        if identity in seen:
            return None
        seen.add(identity)
        previous_turn = turn
        canonical.append(
            {
                "turn": turn,
                "tool": str(tool_name),
                "input": canonical_input,
                "output_sha256": output_sha256,
            }
        )
    return canonical


def build_discover_publication_evidence(
    *,
    stage: int,
    ticker: str,
    scope_fingerprint: str,
    primary_evidence: Any,
    stage4_tool_manifest: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the canonical evidence identity persisted with a Discover result."""

    normalized_stage = int(stage)
    if normalized_stage not in _STAGE_TABLES:
        raise ValueError(f"unsupported discover stage: {stage}")
    normalized_ticker = str(ticker).strip().upper()
    normalized_scope = str(scope_fingerprint or "").strip().lower()
    if not normalized_ticker or _SHA256_RE.fullmatch(normalized_scope) is None:
        raise ValueError("publication evidence requires ticker and exact financial scope")
    entries = [
        {
            "sequence": 0,
            "kind": "provider_input",
            "name": _STAGE_PRIMARY_EVIDENCE_NAMES[normalized_stage],
            "sha256": _canonical_sha256(primary_evidence),
        }
    ]
    if normalized_stage == 4:
        canonical_manifest = _strict_stage4_tool_manifest(
            stage4_tool_manifest,
            ticker=normalized_ticker,
        )
        if canonical_manifest is None:
            raise ValueError("malformed Stage 4 financial evidence manifest")
        for sequence, item in enumerate(canonical_manifest, start=1):
            entries.append(
                {
                    "sequence": sequence,
                    "kind": "tool_evidence",
                    "name": f"stage4_tool:{item['turn']}:{item['tool']}",
                    "sha256": _canonical_sha256(item),
                }
            )
    elif stage4_tool_manifest is not None:
        raise ValueError("tool evidence is only valid for Discover Stage 4")
    return {
        "schema": _PUBLICATION_EVIDENCE_SCHEMA,
        "stage": normalized_stage,
        "ticker": normalized_ticker,
        "scope_fingerprint": normalized_scope,
        "entries": entries,
    }


def _strict_publication_evidence(
    value: Any,
    *,
    stage: int,
    ticker: str,
    scope_fingerprint: str,
) -> dict[str, Any] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, dict) or set(value) != {
        "schema",
        "stage",
        "ticker",
        "scope_fingerprint",
        "entries",
    }:
        return None
    normalized_ticker = str(ticker).strip().upper()
    normalized_scope = str(scope_fingerprint or "").strip().lower()
    if (
        value.get("schema") != _PUBLICATION_EVIDENCE_SCHEMA
        or value.get("stage") != int(stage)
        or value.get("ticker") != normalized_ticker
        or value.get("scope_fingerprint") != normalized_scope
    ):
        return None
    entries = value.get("entries")
    if not isinstance(entries, list) or not entries:
        return None
    canonical_entries: list[dict[str, Any]] = []
    for sequence, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) != {
            "sequence",
            "kind",
            "name",
            "sha256",
        }:
            return None
        expected_kind = "provider_input" if sequence == 0 else "tool_evidence"
        expected_name = _STAGE_PRIMARY_EVIDENCE_NAMES[int(stage)] if sequence == 0 else None
        if (
            entry.get("sequence") != sequence
            or entry.get("kind") != expected_kind
            or (expected_name is not None and entry.get("name") != expected_name)
            or not isinstance(entry.get("name"), str)
            or not str(entry.get("name")).strip()
            or _SHA256_RE.fullmatch(str(entry.get("sha256") or "").strip().lower()) is None
            or (int(stage) != 4 and sequence > 0)
        ):
            return None
        canonical_entries.append(
            {
                "sequence": sequence,
                "kind": expected_kind,
                "name": str(entry["name"]),
                "sha256": str(entry["sha256"]).strip().lower(),
            }
        )
    return {
        "schema": _PUBLICATION_EVIDENCE_SCHEMA,
        "stage": int(stage),
        "ticker": normalized_ticker,
        "scope_fingerprint": normalized_scope,
        "entries": canonical_entries,
    }


def _publication_payload(stage: int, row: Any) -> dict[str, Any] | None:
    payload: dict[str, Any] = {}
    for field in _STAGE_PUBLICATION_FIELDS[int(stage)]:
        try:
            value = row[field]
        except (KeyError, IndexError):
            return None
        if field in _STAGE_JSON_FIELDS[int(stage)]:
            if value is None:
                payload[field] = None
                continue
            try:
                value = json.loads(value) if isinstance(value, str) else value
                _canonical_json(value)
            except (json.JSONDecodeError, TypeError, ValueError):
                return None
        payload[field] = value
    return payload


def _publication_row_sha256(stage: int, row: Any) -> str | None:
    payload = _publication_payload(stage, row)
    return _canonical_sha256(payload) if payload is not None else None


def _row_publication_is_self_consistent(stage: int, row: Any) -> bool:
    paid = str(row["financial_scope_fingerprint"] or "").strip().lower()
    publication = str(row["financial_scope_publication_fingerprint"] or "").strip().lower()
    paid_attempts_sha = str(row["paid_attempts_sha256"] or "").strip().lower()
    stored_row_sha = str(row["publication_row_sha256"] or "").strip().lower()
    if (
        _SHA256_RE.fullmatch(paid) is None
        or publication != paid
        or _SHA256_RE.fullmatch(paid_attempts_sha) is None
        or _SHA256_RE.fullmatch(stored_row_sha) is None
    ):
        return False
    evidence = _strict_publication_evidence(
        row["publication_evidence_json"],
        stage=stage,
        ticker=row["ticker"],
        scope_fingerprint=paid,
    )
    if evidence is None:
        return False
    if int(stage) == 4:
        raw_manifest = row["financial_scope_manifest_json"]
        try:
            manifest = json.loads(raw_manifest) if isinstance(raw_manifest, str) else raw_manifest
        except json.JSONDecodeError:
            return False
        strict_manifest = _strict_stage4_tool_manifest(
            manifest,
            ticker=str(row["ticker"]),
        )
        if strict_manifest is None or len(evidence["entries"]) != len(strict_manifest) + 1:
            return False
        for entry, item in zip(evidence["entries"][1:], strict_manifest, strict=True):
            if entry["name"] != f"stage4_tool:{item['turn']}:{item['tool']}" or entry[
                "sha256"
            ] != _canonical_sha256(item):
                return False
    expected_row_sha = _publication_row_sha256(stage, row)
    return expected_row_sha is not None and expected_row_sha == stored_row_sha


def _row_publication_is_authorized(
    conn: sqlite3.Connection,
    stage: int,
    row: Any,
) -> bool:
    """Require one exact, separate authorization for this immutable row image."""

    if not _row_publication_is_self_consistent(stage, row):
        return False
    evidence = _strict_publication_evidence(
        row["publication_evidence_json"],
        stage=stage,
        ticker=row["ticker"],
        scope_fingerprint=row["financial_scope_fingerprint"],
    )
    if evidence is None:
        return False
    matching = conn.execute(
        """
        SELECT COUNT(*)
        FROM discover_publication_authorizations
        WHERE stage = ?
          AND sweep_id = ?
          AND ticker = ?
          AND publication_row_sha256 = ?
          AND financial_scope_fingerprint = ?
          AND publication_evidence_sha256 = ?
        """,
        (
            int(stage),
            str(row["sweep_id"]),
            str(row["ticker"]).strip().upper(),
            str(row["publication_row_sha256"]).strip().lower(),
            str(row["financial_scope_fingerprint"]).strip().lower(),
            _canonical_sha256(evidence),
        ),
    ).fetchone()[0]
    if matching != 1:
        return False
    attempt_rows = _paid_attempt_rows(
        conn,
        stage=stage,
        sweep_id=str(row["sweep_id"]),
        ticker=str(row["ticker"]),
    )
    if not attempt_rows or any(
        str(attempt["status"]) != "PUBLISHED"
        or str(attempt["publication_row_sha256"] or "").strip().lower()
        != str(row["publication_row_sha256"] or "").strip().lower()
        for attempt in attempt_rows
    ):
        return False
    try:
        attempt_summary = _paid_attempt_summary_from_rows(attempt_rows)
    except (DiscoverPaidAttemptAmbiguousError, TypeError, ValueError):
        return False
    return (
        attempt_summary["paid_attempts_sha256"]
        == str(row["paid_attempts_sha256"] or "").strip().lower()
    )


def _prepare_publication_row(
    stage: int,
    row: dict[str, Any],
    *,
    publication_evidence: dict[str, Any] | None,
) -> tuple[dict[str, Any], str | None]:
    """Canonicalize and hash one candidate row without authorizing it."""

    prepared = dict(row)
    paid = str(prepared.get("financial_scope_fingerprint") or "").strip().lower()
    publication = str(prepared.get("financial_scope_publication_fingerprint") or "").strip().lower()
    scope_claims_match = (
        _SHA256_RE.fullmatch(paid) is not None
        and _SHA256_RE.fullmatch(publication) is not None
        and paid == publication
    )
    if not scope_claims_match:
        prepared["publication_evidence_json"] = (
            _canonical_json(publication_evidence) if publication_evidence is not None else None
        )
        return prepared, None

    evidence = _strict_publication_evidence(
        publication_evidence,
        stage=stage,
        ticker=str(prepared["ticker"]),
        scope_fingerprint=paid,
    )
    if evidence is None:
        raise ValueError(
            f"Discover Stage {stage} publication candidate requires canonical evidence"
        )
    prepared["publication_evidence_json"] = _canonical_json(evidence)
    if int(stage) == 4:
        manifest = prepared.get("financial_scope_manifest_json")
        if (
            _strict_stage4_tool_manifest(
                json.loads(manifest) if isinstance(manifest, str) else manifest,
                ticker=str(prepared["ticker"]),
            )
            is None
        ):
            raise ValueError("Discover Stage 4 publication candidate has malformed tool evidence")
    row_sha = _publication_row_sha256(stage, prepared)
    if row_sha is None:
        raise ValueError(f"Discover Stage {stage} publication row is not canonical")
    return prepared, row_sha


def _authorize_discover_result_in_transaction(
    conn: sqlite3.Connection,
    *,
    stage: int,
    sweep_id: str,
    ticker: str,
    financial_scope: "BoundV1FinancialScope",
    publication_evidence: dict[str, Any],
    expected_publication_row_sha256: str,
    financial_scenarios: tuple[Any, ...] | None = None,
) -> None:
    """Authorize one Discover row inside its insertion transaction.

    This helper is intentionally private and never opens or commits a
    transaction. Production callers must use ``_publish_discover_result`` so a
    result row and its immutable authorization either commit together or do
    not exist. Generic insert helpers remain audit/test utilities and cannot
    authorize a durable row after the fact.
    """

    from app.autonomous.v1_financial_context import BoundV1FinancialScope

    try:
        normalized_stage = int(stage)
        table_name = _STAGE_TABLES[normalized_stage]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"unsupported discover stage: {stage}") from exc
    if not isinstance(financial_scope, BoundV1FinancialScope):
        raise TypeError("Discover publication authorization requires BoundV1FinancialScope")

    normalized_ticker = str(ticker).strip().upper()
    expected_row_sha256 = str(expected_publication_row_sha256 or "").strip().lower()
    if _SHA256_RE.fullmatch(expected_row_sha256) is None:
        raise ValueError("Discover publication authorization requires exact inserted row hash")
    packet_tickers = {
        str(packet.get("ticker") or "").strip().upper()
        for packet in financial_scope.packets
        if isinstance(packet, dict)
    }
    if normalized_ticker not in packet_tickers:
        raise ValueError("Discover publication scope does not contain the result ticker")

    gate_result = financial_scope.require(scenarios=financial_scenarios)
    scope_fingerprint = str(gate_result.scope_fingerprint or "").strip().lower()
    if (
        _SHA256_RE.fullmatch(scope_fingerprint) is None
        or scope_fingerprint != financial_scope.expected_scope_fingerprint
    ):
        raise ValueError("Discover publication scope did not revalidate exactly")

    if not conn.in_transaction:
        raise RuntimeError("Discover authorization requires the active insertion transaction")
    row = conn.execute(
        f"SELECT * FROM {table_name} WHERE sweep_id = ? AND ticker = ?",
        (sweep_id, normalized_ticker),
    ).fetchone()
    if row is None:
        raise ValueError(
            f"Discover Stage {normalized_stage} result does not exist for "
            f"{sweep_id}/{normalized_ticker}"
        )
    if not _row_publication_is_self_consistent(normalized_stage, row):
        raise ValueError(
            f"Discover Stage {normalized_stage} result is not a canonical publication row"
        )
    paid = str(row["financial_scope_fingerprint"] or "").strip().lower()
    publication = str(row["financial_scope_publication_fingerprint"] or "").strip().lower()
    if paid != scope_fingerprint or publication != scope_fingerprint:
        raise ValueError("Discover publication row does not match the revalidated bound scope")

    row_evidence = _strict_publication_evidence(
        row["publication_evidence_json"],
        stage=normalized_stage,
        ticker=normalized_ticker,
        scope_fingerprint=scope_fingerprint,
    )
    expected_evidence = _strict_publication_evidence(
        publication_evidence,
        stage=normalized_stage,
        ticker=normalized_ticker,
        scope_fingerprint=scope_fingerprint,
    )
    if (
        row_evidence is None
        or expected_evidence is None
        or _canonical_sha256(row_evidence) != _canonical_sha256(expected_evidence)
    ):
        raise ValueError("Discover publication evidence changed before authorization")

    row_sha256 = _publication_row_sha256(normalized_stage, row)
    if (
        row_sha256 != expected_row_sha256
        or row_sha256 != str(row["publication_row_sha256"] or "").strip().lower()
    ):
        raise ValueError("Discover publication row changed before authorization")
    conn.execute(
        """
        INSERT INTO discover_publication_authorizations (
            stage, sweep_id, ticker, publication_row_sha256,
            financial_scope_fingerprint, publication_evidence_sha256,
            authorized_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            normalized_stage,
            sweep_id,
            normalized_ticker,
            row_sha256,
            scope_fingerprint,
            _canonical_sha256(row_evidence),
            _utc_now_iso(),
        ),
    )


def _publish_discover_result(
    db_path: str | Path,
    *,
    stage: int,
    row: dict[str, Any],
    financial_scope: "BoundV1FinancialScope",
    publication_evidence: dict[str, Any],
    financial_scenarios: tuple[Any, ...] | None = None,
) -> str:
    """Atomically publish a result and bind its already-accounted attempt ledger."""

    try:
        normalized_stage = int(stage)
        table_name = _STAGE_TABLES[normalized_stage]
        publication_fields = _STAGE_PUBLICATION_FIELDS[normalized_stage]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"unsupported discover stage: {stage}") from exc
    conn = _connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        attempt_rows = _paid_attempt_rows(
            conn,
            stage=normalized_stage,
            sweep_id=str(row["sweep_id"]),
            ticker=str(row["ticker"]),
        )
        attempt_summary = _paid_attempt_summary_from_rows(attempt_rows)
        candidate_row = {
            **row,
            "input_tokens": attempt_summary["input_tokens"],
            "output_tokens": attempt_summary["output_tokens"],
            "cost_usd": attempt_summary["cost_usd"],
            "paid_attempts_sha256": attempt_summary["paid_attempts_sha256"],
        }
        prepared, publication_row_sha256 = _prepare_publication_row(
            normalized_stage,
            candidate_row,
            publication_evidence=publication_evidence,
        )
        if publication_row_sha256 is None:
            raise ValueError(
                f"Discover Stage {normalized_stage} production publication "
                "requires exact scope evidence"
            )
        for field in _STAGE_JSON_FIELDS[normalized_stage]:
            value = prepared.get(field)
            if value is None:
                continue
            parsed = json.loads(value) if isinstance(value, str) else value
            prepared[field] = _canonical_json(parsed)

        columns = (*publication_fields, "publication_row_sha256")
        placeholders = ", ".join("?" for _field in columns)
        conn.execute(
            f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})",
            (*(prepared[field] for field in publication_fields), publication_row_sha256),
        )
        _authorize_discover_result_in_transaction(
            conn,
            stage=normalized_stage,
            sweep_id=str(prepared["sweep_id"]),
            ticker=str(prepared["ticker"]),
            financial_scope=financial_scope,
            publication_evidence=publication_evidence,
            expected_publication_row_sha256=publication_row_sha256,
            financial_scenarios=financial_scenarios,
        )
        published_at = _utc_now_iso()
        attempt_cursor = conn.execute(
            """
            UPDATE discover_paid_attempts
            SET status = 'PUBLISHED', published_at = ?, publication_row_sha256 = ?
            WHERE stage = ? AND sweep_id = ? AND ticker = ?
              AND status IN ('RETURNED', 'FAILED')
            """,
            (
                published_at,
                publication_row_sha256,
                normalized_stage,
                str(prepared["sweep_id"]),
                str(prepared["ticker"]).strip().upper(),
            ),
        )
        if attempt_cursor.rowcount != int(attempt_summary["attempt_count"]):
            raise DiscoverPaidAttemptAmbiguousError(
                "Discover paid-attempt ledger changed during publication"
            )
        sweep_cost = conn.execute(
            "SELECT total_cost_usd FROM sweeps WHERE sweep_id = ?",
            (str(prepared["sweep_id"]),),
        ).fetchone()
        ledger_cost = conn.execute(
            """
            SELECT COALESCE(SUM(
                CASE
                    WHEN status = 'RESERVED' THEN estimated_cost_usd
                    ELSE accounted_cost_usd
                END
            ), 0.0) AS cost_usd
            FROM discover_paid_attempts
            WHERE sweep_id = ?
            """,
            (str(prepared["sweep_id"]),),
        ).fetchone()
        if sweep_cost is None or float(sweep_cost["total_cost_usd"] or 0.0) + 1e-12 < float(
            ledger_cost["cost_usd"] or 0.0
        ):
            raise DiscoverPaidAttemptAmbiguousError(
                "Discover sweep cost understates its durable paid-attempt ledger"
            )
        published_row = conn.execute(
            f"SELECT * FROM {table_name} WHERE sweep_id = ? AND ticker = ?",
            (
                str(prepared["sweep_id"]),
                str(prepared["ticker"]).strip().upper(),
            ),
        ).fetchone()
        if published_row is None or not _row_publication_is_authorized(
            conn,
            normalized_stage,
            published_row,
        ):
            raise ValueError("Discover paid result publication did not authorize atomically")
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()
    return publication_row_sha256


def _filter_authorized_rows(
    conn: sqlite3.Connection,
    stage: int,
    rows: list[sqlite3.Row],
    *,
    require_financial_scope: bool,
) -> list[dict[str, Any]]:
    if not require_financial_scope:
        return [dict(row) for row in rows]
    return [dict(row) for row in rows if _row_publication_is_authorized(conn, stage, row)]


def create_sweep(
    db_path: str | Path,
    sweep_id: str,
    universe_size: int,
    limit_applied: int | None,
    budget_usd: float | None,
) -> str:
    """Insert a new sweep row with status='running'. Returns the sweep_id."""
    conn = _connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO sweeps (sweep_id, started_at, universe_size, limit_applied, budget_usd, status)
            VALUES (?, ?, ?, ?, ?, 'running')
            """,
            (sweep_id, _utc_now_iso(), universe_size, limit_applied, budget_usd),
        )
        conn.commit()
        return sweep_id
    finally:
        conn.close()


def get_sweep(db_path: str | Path, sweep_id: str) -> dict[str, Any] | None:
    conn = _connect(db_path)
    try:
        row = conn.execute("SELECT * FROM sweeps WHERE sweep_id = ?", (sweep_id,)).fetchone()
        if row is None:
            return None
        attempt_costs = conn.execute(
            """
            SELECT
                COUNT(*) AS paid_attempt_count,
                COALESCE(SUM(
                    CASE WHEN status = 'RESERVED' THEN estimated_cost_usd ELSE 0.0 END
                ), 0.0) AS reserved_attempt_cost_usd,
                COALESCE(SUM(
                    CASE
                        WHEN status = 'PUBLISHED' THEN 0.0
                        WHEN status = 'RESERVED' THEN estimated_cost_usd
                        ELSE accounted_cost_usd
                    END
                ), 0.0) AS unpublished_attempt_cost_usd
            FROM discover_paid_attempts
            WHERE sweep_id = ?
            """,
            (sweep_id,),
        ).fetchone()
        payload = dict(row)
        payload.update(
            {
                "paid_attempt_count": int(attempt_costs["paid_attempt_count"] or 0),
                "reserved_attempt_cost_usd": float(
                    attempt_costs["reserved_attempt_cost_usd"] or 0.0
                ),
                "unpublished_attempt_cost_usd": float(
                    attempt_costs["unpublished_attempt_cost_usd"] or 0.0
                ),
            }
        )
        return payload
    finally:
        conn.close()


def insert_sweep_universe(
    db_path: str | Path,
    sweep_id: str,
    universe: list[tuple[str, str]],
) -> None:
    """Bulk-insert (ticker, as_of_date) pairs for a sweep's universe snapshot."""
    conn = _connect(db_path)
    try:
        conn.executemany(
            "INSERT INTO sweep_universe (sweep_id, ticker, as_of_date) VALUES (?, ?, ?)",
            [(sweep_id, ticker, as_of_date) for ticker, as_of_date in universe],
        )
        conn.commit()
    finally:
        conn.close()


def get_sweep_universe(db_path: str | Path, sweep_id: str) -> list[tuple[str, str]]:
    """Return (ticker, as_of_date) pairs for a sweep's universe snapshot.

    Returns an empty list if no snapshot exists (pre-snapshot sweep).
    """
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT ticker, as_of_date FROM sweep_universe WHERE sweep_id = ? ORDER BY ticker",
            (sweep_id,),
        ).fetchall()
        return [(r["ticker"], r["as_of_date"]) for r in rows]
    finally:
        conn.close()


def add_sweep_cost(db_path: str | Path, sweep_id: str, delta_usd: float) -> None:
    conn = _connect(db_path)
    try:
        conn.execute(
            "UPDATE sweeps SET total_cost_usd = total_cost_usd + ? WHERE sweep_id = ?",
            (delta_usd, sweep_id),
        )
        conn.commit()
    finally:
        conn.close()


def finalize_sweep(db_path: str | Path, sweep_id: str, status: str) -> None:
    conn = _connect(db_path)
    try:
        conn.execute(
            "UPDATE sweeps SET status = ?, finished_at = ? WHERE sweep_id = ?",
            (status, _utc_now_iso(), sweep_id),
        )
        conn.commit()
    finally:
        conn.close()


def update_sweep_status(db_path: str | Path, sweep_id: str, status: str) -> None:
    """Set sweeps.status (e.g. back to 'running' on resume)."""
    conn = _connect(db_path)
    try:
        conn.execute(
            "UPDATE sweeps SET status = ? WHERE sweep_id = ?",
            (status, sweep_id),
        )
        conn.commit()
    finally:
        conn.close()


def update_sweep_budget(db_path: str | Path, sweep_id: str, new_budget_usd: float) -> None:
    """Set sweeps.budget_usd to a new value."""
    conn = _connect(db_path)
    try:
        conn.execute(
            "UPDATE sweeps SET budget_usd = ? WHERE sweep_id = ?",
            (new_budget_usd, sweep_id),
        )
        conn.commit()
    finally:
        conn.close()


def list_stage2_tickers(
    db_path: str | Path,
    sweep_id: str,
    *,
    require_financial_scope: bool = False,
) -> set[str]:
    """Return set of tickers that have a stage2_results row."""
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM stage2_results WHERE sweep_id = ?",
            (sweep_id,),
        ).fetchall()
        if require_financial_scope:
            rows = [row for row in rows if _row_publication_is_authorized(conn, 2, row)]
        return {r["ticker"] for r in rows}
    finally:
        conn.close()


def list_stage3_tickers(
    db_path: str | Path,
    sweep_id: str,
    *,
    require_financial_scope: bool = False,
) -> set[str]:
    """Return set of tickers that have a stage3_results row."""
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM stage3_results WHERE sweep_id = ?",
            (sweep_id,),
        ).fetchall()
        if require_financial_scope:
            rows = [row for row in rows if _row_publication_is_authorized(conn, 3, row)]
        return {r["ticker"] for r in rows}
    finally:
        conn.close()


def list_stage4_tickers(
    db_path: str | Path,
    sweep_id: str,
    *,
    require_financial_scope: bool = False,
) -> set[str]:
    """Return set of tickers that have a stage4_results row."""
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM stage4_results WHERE sweep_id = ?",
            (sweep_id,),
        ).fetchall()
        if require_financial_scope:
            rows = [row for row in rows if _row_publication_is_authorized(conn, 4, row)]
        return {r["ticker"] for r in rows}
    finally:
        conn.close()


def stage_result_financial_fingerprints(
    db_path: str | Path,
    *,
    stage: int,
    sweep_id: str,
) -> dict[str, str | None]:
    """Return only cache bindings that passed the publication-time comparison."""

    try:
        normalized_stage = int(stage)
        table_name = _STAGE_TABLES[normalized_stage]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"unsupported discover stage: {stage}") from exc
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            f"""
            SELECT *
            FROM {table_name}
            WHERE sweep_id = ?
            """,
            (sweep_id,),
        ).fetchall()
        bindings: dict[str, str | None] = {}
        for row in rows:
            paid = str(row["financial_scope_fingerprint"] or "").strip().lower()
            publication = str(row["financial_scope_publication_fingerprint"] or "").strip().lower()
            publication_valid = (
                re.fullmatch(r"[0-9a-f]{64}", paid) is not None
                and re.fullmatch(r"[0-9a-f]{64}", publication) is not None
                and paid == publication
                and _row_publication_is_authorized(conn, normalized_stage, row)
            )
            bindings[str(row["ticker"]).strip().upper()] = paid if publication_valid else None
        return bindings
    finally:
        conn.close()


def stage4_result_financial_manifests(
    db_path: str | Path,
    *,
    sweep_id: str,
) -> dict[str, str | None]:
    """Return persisted Stage 4 tool-evidence manifests by ticker."""

    conn = _connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT ticker, financial_scope_manifest_json
            FROM stage4_results
            WHERE sweep_id = ?
            """,
            (sweep_id,),
        ).fetchall()
        return {
            str(row["ticker"]).strip().upper(): (
                str(row["financial_scope_manifest_json"])
                if row["financial_scope_manifest_json"] is not None
                else None
            )
            for row in rows
        }
    finally:
        conn.close()


def require_discover_result_financial_scope(
    *,
    stage: int,
    sweep_id: str,
    ticker: str,
    run_as_of_date: str,
    stored_fingerprint: str | None,
    expected_fingerprint: str,
) -> None:
    """Reject legacy or stale Discover results before skip/reuse/publication."""

    from app.autonomous.financial_integrity import (
        INVALID_FINANCIAL_INPUT,
        NEEDS_DATA,
        FinancialIntegrityGateResult,
        FinancialIntegrityViolation,
        InvalidFinancialInputError,
    )

    stored = str(stored_fingerprint or "").strip().lower()
    expected = str(expected_fingerprint or "").strip().lower()
    stored_is_hash = re.fullmatch(r"[0-9a-f]{64}", stored) is not None
    expected_is_hash = re.fullmatch(r"[0-9a-f]{64}", expected) is not None
    if stored_is_hash and expected_is_hash and stored == expected:
        return
    missing = not stored_is_hash
    code = (
        "DISCOVER_RESULT_FINANCIAL_SCOPE_MISSING"
        if missing
        else "DISCOVER_RESULT_FINANCIAL_SCOPE_MISMATCH"
    )
    status = NEEDS_DATA if missing else INVALID_FINANCIAL_INPUT
    violation = FinancialIntegrityViolation(
        code=code,
        ticker=str(ticker).strip().upper(),
        field=f"stage{int(stage)}_results.financial_scope_fingerprint",
        source_values={
            "stage": int(stage),
            "sweep_id": str(sweep_id),
            "stored_fingerprint": stored or None,
            "expected_fingerprint": expected or None,
        },
        expected_relationship=(
            "persisted Discover result is bound to the exact current "
            "canonical financial input scope"
        ),
        observed_relationship=(
            "missing or malformed" if missing else f"{stored} != {expected or 'MISSING'}"
        ),
        reason=(
            "A cached Discover decision cannot be reused because its exact "
            "financial input scope is missing or changed."
        ),
        terminal_status=status,
    )
    raise InvalidFinancialInputError(
        FinancialIntegrityGateResult(
            context=(f"discover_stage{int(stage)}_cache:{sweep_id}:{str(ticker).strip().upper()}"),
            run_as_of_date=str(run_as_of_date or "").strip()[:10],
            status=status,
            violations=(violation,),
            scope_fingerprint=expected,
        )
    )


def require_discover_financial_scope_unchanged(
    *,
    stage: int,
    ticker: str,
    run_as_of_date: str,
    authorized_fingerprint: str | None,
    current_fingerprint: str | None,
    phase: str,
    sweep_id: str | None = None,
) -> None:
    """Require the current Discover input scope to match its paid authorization."""

    from app.autonomous.financial_integrity import (
        INVALID_FINANCIAL_INPUT,
        NEEDS_DATA,
        FinancialIntegrityGateResult,
        FinancialIntegrityViolation,
        InvalidFinancialInputError,
    )

    authorized = str(authorized_fingerprint or "").strip().lower()
    current = str(current_fingerprint or "").strip().lower()
    authorized_is_hash = re.fullmatch(r"[0-9a-f]{64}", authorized) is not None
    current_is_hash = re.fullmatch(r"[0-9a-f]{64}", current) is not None
    if authorized_is_hash and current_is_hash and authorized == current:
        return

    missing = not authorized_is_hash or not current_is_hash
    code = (
        "DISCOVER_FINANCIAL_SCOPE_AUTHORIZATION_MISSING"
        if missing
        else "DISCOVER_FINANCIAL_SCOPE_MUTATED"
    )
    status = NEEDS_DATA if missing else INVALID_FINANCIAL_INPUT
    normalized_phase = str(phase or "scope_check").strip().lower()
    violation = FinancialIntegrityViolation(
        code=code,
        ticker=str(ticker).strip().upper(),
        field=f"stage{int(stage)}.{normalized_phase}_financial_scope_fingerprint",
        source_values={
            "stage": int(stage),
            "phase": normalized_phase,
            "sweep_id": str(sweep_id) if sweep_id is not None else None,
            "authorized_fingerprint": authorized or None,
            "current_fingerprint": current or None,
        },
        expected_relationship=(
            "the current canonical financial input scope exactly matches the "
            "scope authorized at the paid boundary"
        ),
        observed_relationship=("missing or malformed" if missing else f"{current} != {authorized}"),
        reason=(
            "Discover financial inputs changed after paid-boundary authorization; "
            "the resulting decision cannot be persisted or published."
        ),
        terminal_status=status,
    )
    raise InvalidFinancialInputError(
        FinancialIntegrityGateResult(
            context=(
                f"discover_stage{int(stage)}_{normalized_phase}:"
                f"{str(sweep_id or 'direct')}:{str(ticker).strip().upper()}"
            ),
            run_as_of_date=str(run_as_of_date or "").strip()[:10],
            status=status,
            violations=(violation,),
            scope_fingerprint=current,
        )
    )


def _scope_bound_clause(*, require_financial_scope: bool) -> str:
    if not require_financial_scope:
        return ""
    return (
        " AND financial_scope_fingerprint IS NOT NULL"
        " AND LENGTH(financial_scope_fingerprint) = 64"
        " AND financial_scope_fingerprint"
        " NOT GLOB '*[^0-9a-fA-F]*'"
        " AND financial_scope_publication_fingerprint IS NOT NULL"
        " AND LENGTH(financial_scope_publication_fingerprint) = 64"
        " AND financial_scope_publication_fingerprint"
        " NOT GLOB '*[^0-9a-fA-F]*'"
        " AND LOWER(financial_scope_publication_fingerprint)"
        " = LOWER(financial_scope_fingerprint)"
    )


def insert_stage2_result(
    db_path: str | Path,
    *,
    sweep_id: str,
    ticker: str,
    decision: str,
    confidence: str,
    reason: str | None,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    wall_ms: int,
    error: str | None = None,
    financial_scope_fingerprint: str | None = None,
    financial_scope_publication_fingerprint: str | None = None,
    publication_evidence: dict[str, Any] | None = None,
) -> str | None:
    row, publication_row_sha256 = _prepare_publication_row(
        2,
        {
            "sweep_id": sweep_id,
            "ticker": ticker,
            "decision": decision,
            "confidence": confidence,
            "reason": reason,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost_usd,
            "wall_ms": wall_ms,
            "error": error,
            "financial_scope_fingerprint": financial_scope_fingerprint,
            "financial_scope_publication_fingerprint": (financial_scope_publication_fingerprint),
            "publication_evidence_json": None,
            "paid_attempts_sha256": None,
        },
        publication_evidence=publication_evidence,
    )
    conn = _connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO stage2_results (
                sweep_id, ticker, decision, confidence, reason,
                input_tokens, output_tokens, cost_usd, wall_ms, error,
                financial_scope_fingerprint,
                financial_scope_publication_fingerprint,
                publication_evidence_json, paid_attempts_sha256,
                publication_row_sha256
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                *(row[field] for field in _STAGE_PUBLICATION_FIELDS[2]),
                publication_row_sha256,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return publication_row_sha256


def list_stage2_keeps(
    db_path: str | Path,
    sweep_id: str,
    *,
    require_financial_scope: bool = False,
) -> list[dict[str, Any]]:
    """Return all rows from stage2_results where decision = 'KEEP'."""
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM stage2_results "
            "WHERE sweep_id = ? AND decision = 'KEEP'"
            + _scope_bound_clause(require_financial_scope=require_financial_scope)
            + " ORDER BY ticker",
            (sweep_id,),
        ).fetchall()
        return _filter_authorized_rows(
            conn,
            2,
            rows,
            require_financial_scope=require_financial_scope,
        )
    finally:
        conn.close()


def list_stage2_results(
    db_path: str | Path,
    sweep_id: str,
    *,
    require_financial_scope: bool = False,
) -> list[dict[str, Any]]:
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM stage2_results WHERE sweep_id = ?"
            + _scope_bound_clause(require_financial_scope=require_financial_scope)
            + " ORDER BY ticker",
            (sweep_id,),
        ).fetchall()
        return _filter_authorized_rows(
            conn,
            2,
            rows,
            require_financial_scope=require_financial_scope,
        )
    finally:
        conn.close()


def insert_stage3_result(
    db_path: str | Path,
    *,
    sweep_id: str,
    ticker: str,
    verdict: str,
    confidence: str,
    thesis_summary: str,
    key_numbers: list[str],
    positives: list[str],
    risks: list[str],
    open_questions: list[str],
    reasoning_trace: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    wall_ms: int,
    error: str | None = None,
    financial_scope_fingerprint: str | None = None,
    financial_scope_publication_fingerprint: str | None = None,
    publication_evidence: dict[str, Any] | None = None,
) -> str | None:
    row, publication_row_sha256 = _prepare_publication_row(
        3,
        {
            "sweep_id": sweep_id,
            "ticker": ticker,
            "verdict": verdict,
            "confidence": confidence,
            "thesis_summary": thesis_summary,
            "key_numbers_json": _canonical_json(key_numbers),
            "positives_json": _canonical_json(positives),
            "risks_json": _canonical_json(risks),
            "open_questions_json": _canonical_json(open_questions),
            "reasoning_trace": reasoning_trace,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost_usd,
            "wall_ms": wall_ms,
            "error": error,
            "financial_scope_fingerprint": financial_scope_fingerprint,
            "financial_scope_publication_fingerprint": (financial_scope_publication_fingerprint),
            "publication_evidence_json": None,
            "paid_attempts_sha256": None,
        },
        publication_evidence=publication_evidence,
    )
    conn = _connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO stage3_results (
                sweep_id, ticker, verdict, confidence, thesis_summary,
                key_numbers_json, positives_json, risks_json, open_questions_json,
                reasoning_trace, input_tokens, output_tokens, cost_usd, wall_ms,
                error, financial_scope_fingerprint,
                financial_scope_publication_fingerprint,
                publication_evidence_json, paid_attempts_sha256,
                publication_row_sha256
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                *(row[field] for field in _STAGE_PUBLICATION_FIELDS[3]),
                publication_row_sha256,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return publication_row_sha256


def list_stage3_survivors(
    db_path: str | Path,
    sweep_id: str,
    *,
    require_financial_scope: bool = False,
) -> list[dict[str, Any]]:
    """Return rows from stage3_results with verdict in ('BUY_CANDIDATE', 'WATCH')."""
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT * FROM stage3_results
            WHERE sweep_id = ? AND verdict IN ('BUY_CANDIDATE', 'WATCH')
            {scope_clause}
            ORDER BY ticker
            """.format(
                scope_clause=_scope_bound_clause(require_financial_scope=require_financial_scope)
            ),
            (sweep_id,),
        ).fetchall()
        return _filter_authorized_rows(
            conn,
            3,
            rows,
            require_financial_scope=require_financial_scope,
        )
    finally:
        conn.close()


def list_stage3_results(
    db_path: str | Path,
    sweep_id: str,
    *,
    require_financial_scope: bool = False,
) -> list[dict[str, Any]]:
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM stage3_results WHERE sweep_id = ?"
            + _scope_bound_clause(require_financial_scope=require_financial_scope)
            + " ORDER BY ticker",
            (sweep_id,),
        ).fetchall()
        return _filter_authorized_rows(
            conn,
            3,
            rows,
            require_financial_scope=require_financial_scope,
        )
    finally:
        conn.close()


def insert_stage4_result(
    db_path: str | Path,
    *,
    sweep_id: str,
    ticker: str,
    verdict: str,
    confidence: str,
    thesis: str,
    key_findings: list[str],
    open_questions: list[str],
    falsifiers: list[str],
    reasoning_trace: str,
    num_turns: int,
    tool_call_counts: dict[str, int],
    termination_reason: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    wall_seconds: float,
    error: str | None = None,
    financial_scope_fingerprint: str | None = None,
    financial_scope_publication_fingerprint: str | None = None,
    financial_scope_manifest: list[dict[str, Any]] | None = None,
    publication_evidence: dict[str, Any] | None = None,
) -> str | None:
    row, publication_row_sha256 = _prepare_publication_row(
        4,
        {
            "sweep_id": sweep_id,
            "ticker": ticker,
            "verdict": verdict,
            "confidence": confidence,
            "thesis": thesis,
            "key_findings_json": _canonical_json(key_findings),
            "open_questions_json": _canonical_json(open_questions),
            "falsifiers_json": _canonical_json(falsifiers),
            "reasoning_trace": reasoning_trace,
            "num_turns": num_turns,
            "tool_call_counts_json": _canonical_json(tool_call_counts),
            "termination_reason": termination_reason,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost_usd,
            "wall_seconds": wall_seconds,
            "error": error,
            "financial_scope_fingerprint": financial_scope_fingerprint,
            "financial_scope_publication_fingerprint": (financial_scope_publication_fingerprint),
            "financial_scope_manifest_json": (
                _canonical_json(financial_scope_manifest)
                if financial_scope_manifest is not None
                else None
            ),
            "publication_evidence_json": None,
            "paid_attempts_sha256": None,
        },
        publication_evidence=publication_evidence,
    )
    conn = _connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO stage4_results (
                sweep_id, ticker, verdict, confidence, thesis,
                key_findings_json, open_questions_json, falsifiers_json,
                reasoning_trace, num_turns, tool_call_counts_json, termination_reason,
                input_tokens, output_tokens, cost_usd, wall_seconds, error,
                financial_scope_fingerprint,
                financial_scope_publication_fingerprint,
                financial_scope_manifest_json,
                publication_evidence_json, paid_attempts_sha256,
                publication_row_sha256
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                *(row[field] for field in _STAGE_PUBLICATION_FIELDS[4]),
                publication_row_sha256,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return publication_row_sha256


def list_stage4_results(
    db_path: str | Path,
    sweep_id: str,
    *,
    require_financial_scope: bool = False,
) -> list[dict[str, Any]]:
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM stage4_results WHERE sweep_id = ?"
            + _scope_bound_clause(require_financial_scope=require_financial_scope)
            + (
                " AND financial_scope_manifest_json IS NOT NULL"
                " AND json_valid(financial_scope_manifest_json)"
                " AND json_type(financial_scope_manifest_json) = 'array'"
                if require_financial_scope
                else ""
            )
            + " ORDER BY ticker",
            (sweep_id,),
        ).fetchall()
        return _filter_authorized_rows(
            conn,
            4,
            rows,
            require_financial_scope=require_financial_scope,
        )
    finally:
        conn.close()
