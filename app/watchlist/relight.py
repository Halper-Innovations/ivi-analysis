"""Scoped, budget-gated re-review runner for provenance-blocked watchlist rows.

A watchlist row is decision-eligible only when the canonical financial-integrity
manifest positively authorizes its exact source run (see
``app.watchlist.lineage``).  Rows whose source run is audited-INVALID, or whose
source run lives outside the audited artifact roots entirely, are dark: no price
trigger, no digest line, no disposition.  The only repair for those rows is a
fresh analyst pass that publishes a new ``autonomous_sector`` run through the
self-authorizing publish path.

This module plans and executes exactly that repair over an *explicit* ticker
list.  Its contract is fail-closed at every boundary:

- **Planning is free.**  ``plan_relight`` performs no LLM call.  It freezes the
  ticker target, the sector/band cells, the provider/model binding, and a cost
  estimate derived from recorded historical usage.
- **Absent budget authorizes nothing.**  ``execute_relight`` requires a finite,
  positive ``budget_usd``; there is no default and no implicit ceiling.
- **An unusable manifest refuses to spend.**  If the audit manifest cannot
  authorize anything, a paid re-review could not relight a single row, so the
  runner refuses before the first call rather than burning money on artifacts
  that would stay dark.
- **Execution is resumable.**  Cell outcomes and cumulative spend persist to a
  checkpoint after every cell; a re-invocation skips completed cells and
  re-validates the ceiling against spend already recorded.
- **The watchlist is never mutated here.**  Rows change only through the
  ordinary authorized publish path inside ``autonomous-sector-run``.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import re
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from statistics import median
from typing import Any, Callable, Iterator, Sequence

from app.autonomous.artifact_financial_audit import (
    active_financial_integrity_manifest_path,
    financial_integrity_manifest_is_usable,
    run_id_is_decision_eligible,
)
from app.autonomous.candidate_review import provider_usage_attestation
from app.autonomous.cap_resolver import band_for_market_cap
from app.autonomous.sweep_delta import CANONICAL_SWEEP_SECTORS, V1_ATOMIC_BANDS
from app.config import get_config
from app.db import connect
from app.market.split_evidence import prepare_relight_split_lineage_evidence

RELIGHT_SCHEMA_VERSION = "watchlist_relight_campaign_v1"

_RELIGHT_ID_RE = re.compile(r"^relight_[0-9]{8}T[0-9]{6}Z_[a-f0-9]{8}$")
_DEFAULT_CELL_TIMEOUT_SECONDS = 7_200
_DEFAULT_CELL_RESERVATION_FLOOR_USD = 0.50
_COST_SAFETY_MULTIPLIER = 1.50
_FALLBACK_COST_PER_REVIEW_USD = 0.05
_FALLBACK_BASE_COST_PER_CELL_USD = 0.05
_COST_SAMPLE_LIMIT = 150

# ``band_for_market_cap`` emits the storage token; the sector runner takes the
# ``*_cap`` focus label.  One mapping, so the two never drift apart silently.
_BAND_TO_FOCUS = {
    "micro": "micro_cap",
    "small": "small_cap",
    "mid": "mid_cap",
    "large_cap": "large_cap",
    "mega_cap": "mega_cap",
}

# Statuses that make a row part of the decision-relevant head: an at-target
# surfacing the operator is expected to act on, or a name already carrying an open
# disposition awaiting triage.
AT_TARGET_STATUSES = ("DEPLOY_READY", "BUY_CONFIRMED")


class RelightError(RuntimeError):
    """Base fail-closed relight error."""


class RelightLocked(RelightError):
    """Another paid relight campaign is already running."""


class RelightAuthorizationError(RelightError):
    """Paid execution was not authorized against the exact plan contract."""


class RelightPlanError(RelightError):
    """The requested relight cannot be planned safely."""


class RelightPreflightError(RelightError):
    """The environment cannot make a fresh run decision-eligible."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(payload: Any) -> str:
    return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _positive_budget(value: Any) -> float:
    """Coerce an authorized ceiling, refusing anything not finite and positive."""

    if value is None:
        raise RelightAuthorizationError(
            "a relight budget is required; an absent --budget-usd authorizes nothing"
        )
    try:
        ceiling = float(value)
    except (TypeError, ValueError) as exc:
        raise RelightAuthorizationError(
            "authorized budget must be a finite positive number"
        ) from exc
    if not math.isfinite(ceiling) or ceiling <= 0:
        raise RelightAuthorizationError("authorized budget must be a finite positive number")
    return ceiling


def _positive_cell_reservation_floor(value: Any) -> float:
    """Coerce the execution-time minimum hard cap for one paid child."""

    try:
        floor = float(value)
    except (TypeError, ValueError) as exc:
        raise RelightAuthorizationError(
            "cell reservation floor must be a finite positive number"
        ) from exc
    if not math.isfinite(floor) or floor <= 0:
        raise RelightAuthorizationError("cell reservation floor must be a finite positive number")
    return floor


def _normalized_tickers(values: Sequence[Any]) -> list[str]:
    seen: dict[str, None] = {}
    for value in values:
        token = str(value or "").strip().upper()
        if token:
            seen.setdefault(token, None)
    return sorted(seen)


def parse_ticker_list(raw: str | None) -> list[str]:
    """Parse a comma/whitespace-separated explicit ticker list."""

    if raw is None:
        return []
    return _normalized_tickers(re.split(r"[,\s]+", str(raw)))


def parse_cell_order(raw: str | None) -> list[str] | None:
    """Parse an optional comma-separated execution-order permutation."""

    if raw is None:
        return None
    return [token.strip() for token in str(raw).split(",") if token.strip()]


def _apply_cell_order(
    cells: Sequence[dict[str, Any]],
    cell_order: Sequence[str] | None,
) -> list[dict[str, Any]]:
    """Return cells in an explicitly authorized exact-permutation order."""

    if cell_order is None:
        return list(cells)

    requested = [str(cell_id).strip() for cell_id in cell_order]
    duplicates = sorted(
        cell_id for cell_id in set(requested) if requested.count(cell_id) > 1
    )
    if duplicates:
        raise RelightAuthorizationError(
            "cell order contains duplicate cell_id(s): " + ", ".join(duplicates)
        )

    by_id = {str(cell["cell_id"]): cell for cell in cells}
    unknown = sorted(set(requested) - set(by_id))
    if unknown:
        raise RelightAuthorizationError(
            "cell order contains unknown cell_id(s): " + ", ".join(unknown)
        )

    missing = sorted(set(by_id) - set(requested))
    if missing:
        raise RelightAuthorizationError(
            "cell order omits planned cell_id(s): " + ", ".join(missing)
        )
    return [by_id[cell_id] for cell_id in requested]


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RelightPlanError(f"expected a JSON object at {path}")
    return payload


SUPPORTED_RELIGHT_PROVIDERS = ("anthropic", "openai", "deepseek")


def _configured_provider_binding() -> dict[str, Any]:
    cfg = get_config()
    provider = str(cfg.llm_provider or "disabled").strip().lower()
    if provider == "anthropic":
        model = str(cfg.anthropic_model or "").strip()
        credential_present = bool(cfg.anthropic_api_key)
    elif provider == "openai":
        model = str(cfg.openai_model or "").strip()
        credential_present = bool(cfg.openai_api_key)
    elif provider == "deepseek":
        # Read through getattr: the DeepSeek config surface lands with its
        # provider module, and a relight planned before that must report an
        # unready binding rather than raise on a missing attribute.
        model = str(getattr(cfg, "deepseek_model", "") or "").strip()
        credential_present = bool(getattr(cfg, "deepseek_api_key", None))
    else:
        model = ""
        credential_present = False
    return {
        "provider": provider,
        "model": model or "disabled",
        "credential_present": credential_present,
        "ready": provider in SUPPORTED_RELIGHT_PROVIDERS and bool(model) and credential_present,
        "relight_provider_policy": "exact_provider_model_strict_no_fallback",
    }


def _validate_relight_id(relight_id: str) -> str:
    token = str(relight_id or "").strip()
    if not _RELIGHT_ID_RE.match(token):
        raise RelightPlanError(f"invalid relight id: {relight_id!r}")
    return token


def _relight_root(output_root: str | Path | None = None) -> Path:
    return Path(output_root) if output_root is not None else get_config().runs_dir / "relight"


def _campaign_dir(relight_id: str, output_root: str | Path | None = None) -> Path:
    return _relight_root(output_root) / _validate_relight_id(relight_id)


def state_path(relight_id: str, output_root: str | Path | None = None) -> Path:
    """Checkpoint location for one relight campaign."""

    return _campaign_dir(relight_id, output_root) / "relight_state.json"


# --------------------------------------------------------------------------
# Blocked-subset discovery
# --------------------------------------------------------------------------


def _db_connection(db_path: str | Path | None = None):
    from app.watchlist.schema import resolve_db_path

    conn = connect(resolve_db_path(db_path))
    conn.row_factory = __import__("sqlite3").Row
    return conn


def blocked_decision_relevant_rows(
    *,
    db_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Current at-target / open-disposition rows that the provenance gate blocks.

    Membership is read from the database, never from a cached report: a row
    qualifies when it is the current row for its ticker, is not REMOVED, sits in
    an at-target status or carries an open AT_TARGET disposition, and its source
    run is *not* decision-eligible under the active manifest.
    """

    from app.watchlist.lineage import watchlist_row_is_decision_eligible
    from app.watchlist.store import current_watchlist_cte

    conn = _db_connection(db_path)
    try:
        rows = conn.execute(
            f"""
            WITH latest_watchlist AS ({current_watchlist_cte()})
            SELECT w.*
            FROM watchlist w
            JOIN latest_watchlist ON latest_watchlist.latest_id = w.id
            WHERE w.status != 'REMOVED'
            ORDER BY w.ticker
            """
        ).fetchall()
        try:
            open_disposition_tickers = {
                str(row["ticker"]).strip().upper()
                for row in conn.execute(
                    "SELECT DISTINCT ticker FROM dispositions "
                    "WHERE status = 'OPEN' AND kind = 'AT_TARGET'"
                ).fetchall()
            }
        except Exception:
            # A database without the disposition ledger still has an at-target
            # head; absence of the table must not silently widen the subset.
            open_disposition_tickers = set()
    finally:
        conn.close()

    blocked: list[dict[str, Any]] = []
    for row in rows:
        ticker = str(row["ticker"]).strip().upper()
        status = str(row["status"] or "").strip().upper()
        decision_relevant = status in AT_TARGET_STATUSES or ticker in open_disposition_tickers
        if not decision_relevant:
            continue
        if watchlist_row_is_decision_eligible(row, manifest_path, ticker=ticker):
            continue
        blocked.append(
            {
                "ticker": ticker,
                "status": status,
                "conviction_grade": row["conviction_grade"],
                "source_run_id": row["source_run_id"],
                "source_sector": row["source_sector"],
                "cap_band": row["cap_band"],
                "market_cap_mm": row["market_cap_mm"],
                "has_open_disposition": ticker in open_disposition_tickers,
            }
        )
    return blocked


def _census_identity(conn, ticker: str) -> tuple[str | None, float | None]:
    """Canonical sector and market cap (in millions) from the registrant census.

    The census is the authoritative universe record: it carries a mapped
    ``canonical_sector`` and a resolved cap with a stated as-of date.  Its own
    ``cap_band`` column is a coarser census band (``large_and_mega``), so the
    atomic sweep band is always re-derived from the cap itself rather than
    read across from a differently-grained vocabulary.
    """

    try:
        row = conn.execute(
            "SELECT canonical_sector, market_cap_usd FROM us_equity_census_issuers "
            "WHERE primary_ticker = ? AND market_cap_status = 'RESOLVED' "
            "ORDER BY market_cap_as_of_date DESC, id DESC LIMIT 1",
            (ticker,),
        ).fetchone()
    except Exception:
        return None, None
    if row is None:
        return None, None
    sector = str(row["canonical_sector"] or "").strip().lower() or None
    try:
        cap_usd = float(row["market_cap_usd"])
    except (TypeError, ValueError):
        return sector, None
    return sector, (cap_usd / 1_000_000.0 if cap_usd > 0 else None)


def _inferred_sector(conn, ticker: str) -> str | None:
    try:
        row = conn.execute(
            "SELECT inferred_sector FROM sector_inference WHERE ticker = ? "
            "ORDER BY as_of_date DESC, id DESC LIMIT 1",
            (ticker,),
        ).fetchone()
    except Exception:
        return None
    if row is None:
        return None
    sector = str(row["inferred_sector"] or "").strip().lower()
    return sector or None


def _latest_market_cap_mm(conn, ticker: str) -> float | None:
    try:
        row = conn.execute(
            "SELECT market_cap FROM market_caps WHERE ticker = ? "
            "AND market_cap IS NOT NULL AND market_cap > 0 "
            "ORDER BY effective_as_of_date DESC, id DESC LIMIT 1",
            (ticker,),
        ).fetchone()
    except Exception:
        return None
    if row is None:
        return None
    try:
        return float(row["market_cap"])
    except (TypeError, ValueError):
        return None


def _resolve_identity(conn, ticker: str) -> tuple[str | None, float | None]:
    """Sector and cap for routing: census first, then the inference/cap tables."""

    sector, cap_mm = _census_identity(conn, ticker)
    if sector is None:
        sector = _inferred_sector(conn, ticker)
    if cap_mm is None:
        cap_mm = _latest_market_cap_mm(conn, ticker)
    return sector, cap_mm


def resolve_cells(
    tickers: Sequence[str],
    *,
    db_path: str | Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Group tickers into canonical ``sector`` x ``band`` re-review cells.

    A watchlist row's stored ``source_sector``/``cap_band`` cannot be trusted for
    routing: rows sourced from a filing-watch pass carry a scan-family label such as
    ``filing_watch`` and often no band at all.  Routing therefore resolves
    against the canonical sector inference and market-cap tables, and any ticker
    that cannot be placed in a real cell is returned as unroutable rather than
    guessed into one.
    """

    conn = _db_connection(db_path)
    try:
        cells: dict[tuple[str, str], list[str]] = {}
        unroutable: list[dict[str, Any]] = []
        for ticker in _normalized_tickers(tickers):
            sector, cap_mm = _resolve_identity(conn, ticker)
            band_token = band_for_market_cap(cap_mm)
            focus = _BAND_TO_FOCUS.get(band_token or "")
            reasons = []
            if sector is None:
                reasons.append("NO_INFERRED_SECTOR")
            elif sector not in CANONICAL_SWEEP_SECTORS:
                reasons.append(f"SECTOR_NOT_CANONICAL:{sector}")
            if cap_mm is None:
                reasons.append("NO_MARKET_CAP")
            elif focus is None or focus not in V1_ATOMIC_BANDS:
                reasons.append(f"BAND_NOT_CANONICAL:{band_token}")
            if reasons:
                unroutable.append({"ticker": ticker, "reasons": reasons})
                continue
            assert sector is not None and focus is not None
            cells.setdefault((sector, focus), []).append(ticker)
        planned = [
            {
                "cell_id": f"{sector}:{band}",
                "sector": sector,
                "band": band,
                "tickers": sorted(members),
                "status": "PENDING",
            }
            for (sector, band), members in sorted(cells.items())
        ]
        return planned, sorted(unroutable, key=lambda item: item["ticker"])
    finally:
        conn.close()


# --------------------------------------------------------------------------
# Cost estimation (no LLM call)
# --------------------------------------------------------------------------


def _artifact_cost_samples(
    *, provider: str, model: str, limit: int = _COST_SAMPLE_LIMIT
) -> list[tuple[float, int]]:
    """Recorded (cost, reviews) pairs for completed v1 runs on this binding."""

    try:
        artifacts = sorted(
            get_config().runs_dir.glob("autonomous_sector/*/autonomous_sector_run.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return []
    samples: list[tuple[float, int]] = []
    for path in artifacts:
        if len(samples) >= limit:
            break
        try:
            artifact = _read_json_object(path)
        except Exception:
            continue
        if str(artifact.get("pipeline_version") or "v1").lower() != "v1":
            continue
        if str(artifact.get("status") or "").upper() != "COMPLETED":
            continue
        packets = artifact.get("company_packets") or []
        if not packets:
            continue
        attestation = provider_usage_attestation(artifact)
        bindings = {
            (str(row["provider"]), str(row["model"])) for row in attestation["provider_models"]
        }
        cost = float(attestation["cost_estimate_usd"])
        if attestation["valid"] and bindings == {(provider, model)} and cost > 0:
            samples.append((cost, len(packets)))
    return samples


def estimate_relight_cost(
    *,
    provider: str,
    model: str,
    cells: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Estimate the relight cost from recorded usage on this exact binding.

    A scoped relight runs few tickers across many cells, so the per-cell fixed
    overhead dominates and must be modelled separately from the per-review
    slope; a pooled dollars-per-ticker average understates a many-cell plan
    badly.
    """

    samples = _artifact_cost_samples(provider=provider, model=model)
    review_units = sum(len(cell["tickers"]) for cell in cells)
    if len(samples) >= 5:
        xs = [float(count) for _cost, count in samples]
        ys = [float(cost) for cost, _count in samples]
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        denominator = sum((value - mean_x) ** 2 for value in xs)
        slope = (
            sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / denominator
            if denominator > 0
            else mean_y / max(1.0, mean_x)
        )
        per_review = max(0.005, slope)
        per_cell = max(
            0.005,
            median(max(0.0, y - per_review * x) for x, y in zip(xs, ys, strict=True)),
        )
        method = "recorded_provider_usage_history_with_50pct_reserve"
    else:
        per_review = _FALLBACK_COST_PER_REVIEW_USD
        per_cell = _FALLBACK_BASE_COST_PER_CELL_USD
        method = "no_matching_history_conservative_fallback_with_50pct_reserve"
    raw = per_review * review_units + per_cell * len(cells)
    per_cell_estimates = {
        str(cell["cell_id"]): round(
            (per_review * len(cell["tickers"]) + per_cell) * _COST_SAFETY_MULTIPLIER, 4
        )
        for cell in cells
    }
    return {
        "estimated_cost_usd": round(raw * _COST_SAFETY_MULTIPLIER, 2),
        "method": method,
        "matching_historical_runs": len(samples),
        "estimated_cost_per_review_usd": round(per_review, 6),
        "estimated_base_cost_per_cell_usd": round(per_cell, 6),
        "safety_multiplier": _COST_SAFETY_MULTIPLIER,
        "review_units": review_units,
        "cell_count": len(cells),
        "per_cell_estimated_cost_usd": per_cell_estimates,
    }


# --------------------------------------------------------------------------
# Preflight
# --------------------------------------------------------------------------


def relight_preflight(manifest_path: str | Path | None = None) -> dict[str, Any]:
    """Whether a freshly published run could become decision-eligible at all.

    The publish path self-authorizes a new run through a post-write
    authorization sidecar, but that sidecar is only consulted when the canonical
    manifest itself loads and authorizes.  With an unusable manifest every fresh
    artifact would stay dark, so paid execution must refuse rather than spend.
    """

    selected = active_financial_integrity_manifest_path(manifest_path)
    usable = financial_integrity_manifest_is_usable(manifest_path)
    blockers: list[str] = []
    if selected is None:
        blockers.append("NO_ACTIVE_FINANCIAL_INTEGRITY_MANIFEST")
    elif not Path(selected).is_file():
        blockers.append(f"MANIFEST_FILE_MISSING:{selected}")
    if not usable:
        blockers.append("MANIFEST_NOT_USABLE_FRESH_RUNS_WOULD_STAY_DARK")
    return {
        "manifest_path": str(selected) if selected is not None else None,
        "manifest_usable": bool(usable),
        "ready": not blockers,
        "blockers": blockers,
    }


# --------------------------------------------------------------------------
# Planning ($0)
# --------------------------------------------------------------------------


def plan_relight(
    *,
    tickers: Sequence[str] | None = None,
    from_blocked: bool = False,
    as_of: str | None = None,
    relight_id: str | None = None,
    output_root: str | Path | None = None,
    db_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    populate_watchlist: bool = True,
    cell_order: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Freeze an explicit relight target and cost estimate.  Performs no LLM call."""

    explicit = _normalized_tickers(tickers or ())
    if from_blocked:
        discovered = [
            row["ticker"]
            for row in blocked_decision_relevant_rows(db_path=db_path, manifest_path=manifest_path)
        ]
        target = _normalized_tickers([*explicit, *discovered])
    else:
        target = explicit
    if not target:
        raise RelightPlanError(
            "a relight requires an explicit ticker list (or --from-blocked with a "
            "non-empty blocked subset)"
        )

    cells, unroutable = resolve_cells(target, db_path=db_path)
    if not cells:
        raise RelightPlanError(
            "no requested ticker could be routed to a canonical sector/band cell; "
            f"unroutable={[item['ticker'] for item in unroutable]}"
        )
    cells = _apply_cell_order(cells, cell_order)

    binding = _configured_provider_binding()
    estimate = estimate_relight_cost(
        provider=binding["provider"], model=binding["model"], cells=cells
    )
    now = _utc_now()
    stamp = now.replace("-", "").replace(":", "")
    resolved_id = _validate_relight_id(
        relight_id or f"relight_{stamp}_{_sha256([now, target])[:8]}"
    )
    effective_as_of = str(as_of).strip() if as_of else datetime.now(timezone.utc).date().isoformat()

    plan = {
        "schema_version": RELIGHT_SCHEMA_VERSION,
        "relight_id": resolved_id,
        "created_at": now,
        "effective_as_of": effective_as_of,
        "requested_tickers": target,
        "routed_tickers": sorted({t for cell in cells for t in cell["tickers"]}),
        "unroutable": unroutable,
        "cells": cells,
        "provider_binding": binding,
        "cost_estimate": estimate,
        "populate_watchlist": bool(populate_watchlist),
        "preflight": relight_preflight(manifest_path),
        "cumulative_cost_usd": 0.0,
        "status": "PLANNED",
    }
    plan["plan_sha256"] = _sha256(
        {
            "relight_id": resolved_id,
            "requested_tickers": target,
            "cells": [{"cell_id": c["cell_id"], "tickers": c["tickers"]} for c in plan["cells"]],
            "effective_as_of": effective_as_of,
            "provider_binding": binding,
        }
    )
    _atomic_write_json(state_path(resolved_id, output_root), plan)
    return plan


# --------------------------------------------------------------------------
# Execution (paid)
# --------------------------------------------------------------------------


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RelightLocked(f"relight lock is already held: {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _child_argv(state: dict[str, Any], cell: dict[str, Any], reservation: float) -> list[str]:
    sector = str(cell.get("sector") or "")
    band = str(cell.get("band") or "")
    if sector not in CANONICAL_SWEEP_SECTORS or band not in V1_ATOMIC_BANDS:
        raise RelightPlanError(f"invalid planned cell: {sector}:{band}")
    argv = [
        sys.executable,
        "-m",
        "app.cli",
        "autonomous-sector-run",
        "--sector",
        sector,
        "--market-cap-focus",
        band,
        "--pipeline-version",
        "v1",
        "--tickers",
        ",".join(cell["tickers"]),
        "--as-of",
        state["effective_as_of"],
        "--max-cost-usd",
        f"{reservation:.6f}",
        "--strict-cost-cap",
    ]
    if not state.get("populate_watchlist", True):
        argv.append("--no-watchlist")
    return argv


def _default_command_runner(
    argv: Sequence[str], *, cwd: Path, timeout_seconds: int
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout_seconds,
        shell=False,
    )


def _last_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for index in range(len(text) - 1, -1, -1):
        if text[index] != "{":
            continue
        try:
            value, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and not text[index + end :].strip():
            return value
    return None


def _child_cost_usd(payload: dict[str, Any] | None) -> float:
    """Recorded spend for one child run; unreadable usage counts as the reservation."""

    if not isinstance(payload, dict):
        return float("nan")
    attestation = payload.get("provider_usage_attestation")
    if isinstance(attestation, dict):
        try:
            cost = float(attestation.get("cost_estimate_usd"))
        except (TypeError, ValueError):
            return float("nan")
        if attestation.get("valid") is not True or not math.isfinite(cost) or cost < 0:
            return float("nan")
        return cost
    return float("nan")


def _validate_execution_authorization(
    state: dict[str, Any],
    *,
    budget_usd: Any,
    accept_estimate_shortfall: bool,
    manifest_path: str | Path | None,
) -> float:
    ceiling = _positive_budget(budget_usd)

    preflight = relight_preflight(manifest_path)
    if not preflight["ready"]:
        raise RelightPreflightError(
            "refusing to spend: a fresh run could not become decision-eligible "
            f"({', '.join(preflight['blockers'])})"
        )

    binding = state["provider_binding"]
    current = _configured_provider_binding()
    if not current["ready"]:
        raise RelightAuthorizationError("configured LLM provider is not ready")
    if current != binding:
        raise RelightAuthorizationError(
            "configured provider/model changed after planning; re-plan before spending"
        )

    estimate = float((state.get("cost_estimate") or {}).get("estimated_cost_usd") or 0.0)
    if ceiling < estimate and not accept_estimate_shortfall:
        raise RelightAuthorizationError(
            f"authorized ceiling ${ceiling:.2f} is below the plan estimate ${estimate:.2f}; "
            "refusing to silently run a partial relight without accept_estimate_shortfall"
        )
    spent = float(state.get("cumulative_cost_usd") or 0.0)
    if spent > ceiling:
        raise RelightAuthorizationError(
            f"recorded relight spend ${spent:.2f} already exceeds this authorization ${ceiling:.2f}"
        )
    return ceiling


def execute_relight(
    *,
    relight_id: str,
    budget_usd: Any,
    output_root: str | Path | None = None,
    manifest_path: str | Path | None = None,
    accept_estimate_shortfall: bool = False,
    cell_reservation_floor_usd: Any = _DEFAULT_CELL_RESERVATION_FLOOR_USD,
    cell_timeout_seconds: int = _DEFAULT_CELL_TIMEOUT_SECONDS,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run the planned relight under a hard ceiling, checkpointing every cell.

    Resumable: cells already recorded COMPLETED are skipped, and the ceiling is
    re-validated against spend already on the checkpoint so a resumed run can
    never exceed the authorization in aggregate.
    """

    path = state_path(relight_id, output_root)
    if not path.is_file():
        raise RelightPlanError(f"no relight plan at {path}; plan before executing")
    runner = command_runner or _default_command_runner
    emit = progress or (lambda _message: None)
    reservation_floor = _positive_cell_reservation_floor(cell_reservation_floor_usd)

    with _exclusive_lock(path.with_name("relight.lock")):
        state = _read_json_object(path)
        if state.get("schema_version") != RELIGHT_SCHEMA_VERSION:
            raise RelightPlanError("relight checkpoint schema is not recognized")
        ceiling = _validate_execution_authorization(
            state,
            budget_usd=budget_usd,
            accept_estimate_shortfall=accept_estimate_shortfall,
            manifest_path=manifest_path,
        )
        state["authorized_max_cost_usd"] = ceiling
        state["cell_reservation_floor_usd"] = round(reservation_floor, 6)
        state["status"] = "RUNNING"
        state["started_at"] = _utc_now()
        _atomic_write_json(path, state)

        runnable_tickers = sorted(
            {
                str(ticker).strip().upper()
                for cell in state["cells"]
                if str(cell.get("status")) != "COMPLETED"
                for ticker in cell.get("tickers") or []
                if str(ticker).strip()
            }
        )
        if runnable_tickers:
            emit(f"prepare split lineage ({len(runnable_tickers)} tickers)")
            try:
                split_evidence = prepare_relight_split_lineage_evidence(
                    tickers=runnable_tickers,
                    as_of_date=str(state["effective_as_of"]),
                )
            except Exception as exc:  # noqa: BLE001 - cells remain fail-closed
                split_evidence = {
                    "as_of_date": str(state["effective_as_of"]),
                    "requested": len(runnable_tickers),
                    "ready": 0,
                    "unknown": len(runnable_tickers),
                    "results": [],
                    "error": type(exc).__name__,
                }
            state["split_lineage_evidence"] = split_evidence
            _atomic_write_json(path, state)
            emit(
                "split lineage prepared "
                f"ready={split_evidence['ready']} unknown={split_evidence['unknown']}"
            )

        per_cell_estimate = (state.get("cost_estimate") or {}).get(
            "per_cell_estimated_cost_usd"
        ) or {}
        repo_root = Path(__file__).resolve().parents[2]

        for cell in state["cells"]:
            if str(cell.get("status")) == "COMPLETED":
                emit(f"skip {cell['cell_id']} (already completed)")
                continue
            spent = float(state.get("cumulative_cost_usd") or 0.0)
            remaining = ceiling - spent
            if remaining < reservation_floor:
                cell["status"] = "SKIPPED_BUDGET_EXHAUSTED"
                state["status"] = "BUDGET_EXHAUSTED"
                _atomic_write_json(path, state)
                emit(
                    f"budget exhausted before {cell['cell_id']} "
                    f"(${remaining:.4f} remaining < ${reservation_floor:.4f} floor)"
                )
                break
            estimated = float(per_cell_estimate.get(cell["cell_id"], remaining))
            reservation = min(remaining, max(estimated, reservation_floor))
            if reservation <= 0:
                cell["status"] = "SKIPPED_BUDGET_EXHAUSTED"
                _atomic_write_json(path, state)
                continue

            argv = _child_argv(state, cell, reservation)
            cell["status"] = "RUNNING"
            cell["reservation_usd"] = round(reservation, 6)
            cell["started_at"] = _utc_now()
            _atomic_write_json(path, state)
            emit(f"run {cell['cell_id']} ({len(cell['tickers'])} tickers, ${reservation:.4f})")

            try:
                completed = runner(argv, cwd=repo_root, timeout_seconds=cell_timeout_seconds)
                returncode = int(completed.returncode)
                stdout = str(completed.stdout or "")
                stderr = str(completed.stderr or "")
            except subprocess.TimeoutExpired:
                returncode = -1
                stdout = ""
                stderr = "cell timed out"

            payload = _last_json_object(stdout)
            observed = _child_cost_usd(payload)
            # An unreadable attestation must never bank as free: charge the full
            # reservation so a malformed child cannot loop the budget forever.
            charged = reservation if math.isnan(observed) else observed
            cell["cost_usd"] = round(charged, 6)
            cell["cost_attested"] = not math.isnan(observed)
            cell["returncode"] = returncode
            cell["run_id"] = (payload or {}).get("run_id")
            cell["completed_at"] = _utc_now()
            cell["status"] = "COMPLETED" if returncode == 0 else "FAILED"
            if returncode != 0:
                stream_tails = []
                if stdout:
                    stream_tails.append(f"stdout tail:\n{stdout[-2000:]}")
                if stderr:
                    stream_tails.append(f"stderr tail:\n{stderr[-2000:]}")
                cell["error"] = "\n".join(stream_tails)
            state["cumulative_cost_usd"] = round(
                float(state.get("cumulative_cost_usd") or 0.0) + charged, 6
            )
            _atomic_write_json(path, state)
            emit(
                f"{cell['status']} {cell['cell_id']} "
                f"spent ${charged:.4f} (total ${state['cumulative_cost_usd']:.4f})"
            )

        state["eligibility"] = verify_relight_eligibility(state, manifest_path=manifest_path)
        pending = [c for c in state["cells"] if str(c.get("status")) == "PENDING"]
        failed = [c for c in state["cells"] if str(c.get("status")) == "FAILED"]
        if state.get("status") != "BUDGET_EXHAUSTED":
            state["status"] = "COMPLETED" if not pending and not failed else "INCOMPLETE"
        state["finished_at"] = _utc_now()
        _atomic_write_json(path, state)
        return state


def verify_relight_eligibility(
    state: dict[str, Any],
    *,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Report which produced runs are actually decision-eligible now."""

    eligible_runs: list[str] = []
    ineligible_runs: list[str] = []
    for cell in state.get("cells") or []:
        run_id = cell.get("run_id")
        if not run_id:
            continue
        if run_id_is_decision_eligible(str(run_id), manifest_path):
            eligible_runs.append(str(run_id))
        else:
            ineligible_runs.append(str(run_id))
    return {
        "checked_at": _utc_now(),
        "decision_eligible_run_ids": sorted(eligible_runs),
        "ineligible_run_ids": sorted(ineligible_runs),
    }


def relight_status(
    *,
    relight_id: str,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """Read one relight checkpoint without mutating it."""

    path = state_path(relight_id, output_root)
    if not path.is_file():
        raise RelightPlanError(f"no relight plan at {path}")
    state = _read_json_object(path)
    cells = state.get("cells") or []
    counts: dict[str, int] = {}
    for cell in cells:
        counts[str(cell.get("status"))] = counts.get(str(cell.get("status")), 0) + 1
    return {
        "relight_id": state.get("relight_id"),
        "status": state.get("status"),
        "cell_status_counts": counts,
        "cumulative_cost_usd": state.get("cumulative_cost_usd"),
        "authorized_max_cost_usd": state.get("authorized_max_cost_usd"),
        "cell_reservation_floor_usd": state.get("cell_reservation_floor_usd"),
        "cost_estimate": state.get("cost_estimate"),
        "eligibility": state.get("eligibility"),
        "state_path": str(path),
    }


__all__ = [
    "RELIGHT_SCHEMA_VERSION",
    "RelightAuthorizationError",
    "RelightError",
    "RelightLocked",
    "RelightPlanError",
    "RelightPreflightError",
    "blocked_decision_relevant_rows",
    "estimate_relight_cost",
    "execute_relight",
    "parse_cell_order",
    "parse_ticker_list",
    "plan_relight",
    "relight_preflight",
    "relight_status",
    "resolve_cells",
    "state_path",
    "verify_relight_eligibility",
]
