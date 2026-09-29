"""Zero-spend bootstrap for the first decision-eligible valuation rows.

The bootstrap is deliberately a producer, not an eligibility exception.  It
reuses cached relight split evidence, runs the deterministic valuation writer,
publishes an ordinary autonomous-sector product artifact, and lets the
existing post-write binder attach the exact artifact bytes to the new rows.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from secrets import token_hex
from typing import Any, Callable

from app.autonomous.financial_integrity import (
    FinancialIntegrityScope,
    InvalidFinancialInputError,
    require_financial_integrity_scope,
    stable_quote_hash,
)
from app.autonomous.output_store import persist_autonomous_sector_run
from app.autonomous.run_contract import ToolCallRecord
from app.autonomous.sector_contract import AutonomousSectorFinancialRunArtifact
from app.autonomous.sector_financial_packets import (
    build_sector_company_financial_packets_from_signal_packets,
)
from app.autonomous.sector_runtime import (
    _financial_integrity_result_payload,
    _financial_integrity_run_binding,
    _require_applied_financial_integrity_scope,
)
from app.autonomous.sector_scenarios import build_expected_return_scenarios_for_packets
from app.autonomous.v1_financial_context import build_canonical_v1_financial_context
from app.config import AppConfig, get_config
from app.db import connect
from app.market.split_evidence import (
    load_persisted_split_lineage_quote,
    prepare_relight_split_lineage_evidence,
)
from app.valuation.lineage import (
    latest_decision_eligible_valuation_row,
    valuation_row_is_decision_eligible,
)
from app.valuation.valuation_writer import ensure_valuation
from app.watchlist.relight import RELIGHT_SCHEMA_VERSION, RelightPlanError, state_path

VALUATION_BOOTSTRAP_SCHEMA_VERSION = "relight_valuation_bootstrap_summary_v1"
VALUATION_BOOTSTRAP_SCAN_FAMILY = "valuation_bootstrap"
VALUATION_BOOTSTRAP_RUN_PREFIX = "autonomous_sector_valuation_bootstrap"
VALUATION_BOOTSTRAP_OBJECTIVE = (
    "Publish deterministic valuation rows from cached point-in-time facts and "
    "proof-carrying quotes without invoking an LLM."
)

_REQUIRED_VALUATION_COLUMNS = {
    "ticker",
    "as_of_date",
    "method",
    "source_run_id",
    "source_artifact_path",
    "source_artifact_sha256",
    "financial_integrity_fingerprint",
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_iso(value: datetime | None = None) -> str:
    return (value or _utc_now()).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _slug(value: object) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    return token or "unknown"


def _read_relight_plan(
    relight_id: str,
    *,
    output_root: str | Path | None,
) -> tuple[dict[str, Any], Path]:
    path = state_path(relight_id, output_root)
    if not path.is_file():
        raise RelightPlanError(f"no relight plan at {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != RELIGHT_SCHEMA_VERSION:
        raise RelightPlanError("relight checkpoint schema is not recognized")
    if str(payload.get("relight_id") or "") != str(relight_id):
        raise RelightPlanError("relight checkpoint id does not match its directory")
    return payload, path


def _assert_valuation_schema(cfg: AppConfig) -> None:
    """Inspect the existing table contract before any bootstrap selection."""

    conn = connect(cfg.db_path, cfg=cfg)
    try:
        columns = {
            str(row[1]) for row in conn.execute("PRAGMA table_info(valuations)").fetchall()
        }
    finally:
        conn.close()
    missing = sorted(_REQUIRED_VALUATION_COLUMNS - columns)
    if missing:
        raise RuntimeError(
            "valuation bootstrap requires existing valuations columns: " + ", ".join(missing)
        )


def _offline_fetch_refusal(url: str, params: dict[str, object]) -> bytes:
    del params
    raise RuntimeError(f"valuation bootstrap refuses network fetch: {url}")


def _usable_anchor(valuation: Any) -> bool:
    if not isinstance(valuation, dict):
        return False
    method = str(valuation.get("anchor_method") or "").strip()
    value = valuation.get("valuation_anchor")
    methods = valuation.get("available_methods")
    return bool(
        method
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
        and isinstance(methods, list)
        and method in {str(item) for item in methods}
    )


def _existing_authorized_scorecard(
    *,
    ticker: str,
    as_of_date: str,
    cfg: AppConfig,
) -> Any | None:
    conn = connect(cfg.db_path, cfg=cfg)
    conn.row_factory = __import__("sqlite3").Row
    try:
        return latest_decision_eligible_valuation_row(
            conn,
            ticker=ticker,
            method="scorecard",
            as_of_date=as_of_date,
            exact_as_of_date=True,
        )
    finally:
        conn.close()


def _assert_published_valuation_bindings(
    *,
    records: list[dict[str, Any]],
    run_id: str,
    artifact_path: Path,
    cfg: AppConfig,
) -> list[str]:
    """Verify every mass-written method carries the published run/path/SHA."""

    resolved_path = artifact_path.resolve(strict=True)
    expected_sha256 = hashlib.sha256(resolved_path.read_bytes()).hexdigest()
    conn = connect(cfg.db_path, cfg=cfg)
    conn.row_factory = sqlite3.Row
    bound_methods: list[str] = []
    try:
        columns = {
            str(row[1]) for row in conn.execute("PRAGMA table_info(valuations)").fetchall()
        }
        missing = sorted(_REQUIRED_VALUATION_COLUMNS - columns)
        if missing:
            raise RuntimeError(
                "valuation bootstrap requires existing valuations columns: " + ", ".join(missing)
            )
        for record in records:
            source_row = record.get("row") if isinstance(record, dict) else None
            if not isinstance(source_row, dict):
                raise RuntimeError("valuation bootstrap received a malformed writer record")
            row = conn.execute(
                """
                SELECT *
                FROM valuations
                WHERE ticker = ? AND as_of_date = ? AND method = ?
                """,
                (
                    source_row.get("ticker"),
                    source_row.get("as_of_date"),
                    source_row.get("method"),
                ),
            ).fetchone()
            if (
                row is None
                or row["source_run_id"] != run_id
                or row["source_artifact_path"] != str(resolved_path)
                or row["source_artifact_sha256"] != expected_sha256
                or not valuation_row_is_decision_eligible(row)
            ):
                raise RuntimeError(
                    "valuation bootstrap failed to bind published method "
                    f"{source_row.get('method')} for {source_row.get('ticker')}"
                )
            bound_methods.append(str(row["method"]))
    finally:
        conn.close()
    return sorted(bound_methods)


def _integrity_scope(
    *,
    as_of_date: str,
    packets: list[Any],
    scenarios: list[Any],
) -> FinancialIntegrityScope:
    return FinancialIntegrityScope(
        context="relight_valuation_bootstrap",
        run_as_of_date=as_of_date,
        packets=tuple(packets),
        scenarios=tuple(scenarios),
    )


def _run_id(*, sector: str, ticker: str, now: datetime) -> str:
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    return (
        f"{VALUATION_BOOTSTRAP_RUN_PREFIX}_{_slug(sector)}_"
        f"{_slug(ticker)}_{stamp}_{token_hex(4)}"
    )


def _bootstrap_one(
    *,
    relight_id: str,
    sector: str,
    market_cap_focus: str,
    ticker: str,
    as_of_date: str,
    cfg: AppConfig,
) -> dict[str, Any]:
    context = build_canonical_v1_financial_context(
        tickers=[ticker],
        as_of_date=as_of_date,
        db_path=cfg.db_path,
        cfg=cfg,
    )
    packets = build_sector_company_financial_packets_from_signal_packets(
        context.packets,
        sector=sector,
        as_of_date=as_of_date,
        cap_classifications=context.issuer_contexts,
        pipeline_version="v1",
    )
    scenarios_by_ticker = build_expected_return_scenarios_for_packets(packets)
    scenarios = [
        scenario
        for scenario_ticker in sorted(scenarios_by_ticker)
        for scenario in scenarios_by_ticker[scenario_ticker]
    ]
    if len(packets) != 1 or packets[0].ticker != ticker:
        return {
            "ticker": ticker,
            "status": "NEEDS_DATA",
            "reason": "FINANCIAL_PACKET_NOT_ASSEMBLED",
        }

    # Refuse before writing valuation rows if the exact cached inputs do not
    # pass the ordinary pre-provider financial-integrity contract.
    preliminary_scope = _integrity_scope(
        as_of_date=as_of_date,
        packets=packets,
        scenarios=scenarios,
    )
    try:
        require_financial_integrity_scope(preliminary_scope)
    except InvalidFinancialInputError as exc:
        return {
            "ticker": ticker,
            "status": "NEEDS_DATA",
            "reason": exc.status,
            "violations": [row.to_dict() for row in exc.result.violations],
        }

    if _usable_anchor(packets[0].valuation) and _existing_authorized_scorecard(
        ticker=ticker,
        as_of_date=as_of_date,
        cfg=cfg,
    ):
        return {
            "ticker": ticker,
            "status": "ALREADY_AUTHORIZED",
            "anchor_method": packets[0].valuation["anchor_method"],
            "valuation_anchor": packets[0].valuation["valuation_anchor"],
        }

    quote = load_persisted_split_lineage_quote(
        ticker,
        as_of_date,
        db_path=cfg.db_path,
        cfg=cfg,
    )
    if quote is None:
        return {
            "ticker": ticker,
            "status": "NEEDS_DATA",
            "reason": "PERSISTED_SPLIT_LINEAGE_QUOTE_UNAVAILABLE",
        }
    price = quote.get("price")
    if (
        not isinstance(price, (int, float))
        or isinstance(price, bool)
        or not math.isfinite(float(price))
        or float(price) <= 0
    ):
        return {
            "ticker": ticker,
            "status": "NEEDS_DATA",
            "reason": "PERSISTED_SPLIT_LINEAGE_QUOTE_INVALID",
        }

    now = _utc_now()
    run_id = _run_id(sector=sector, ticker=ticker, now=now)
    issuer_aliases = tuple(
        dict.fromkeys(
            str(item).strip().upper()
            for item in (
                ticker,
                quote.get("issuer_primary_ticker"),
                *(quote.get("issuer_aliases") or ()),
            )
            if str(item or "").strip()
        )
    )
    quote_provenance = {
        **quote,
        "source": str(quote.get("source") or "eodhd_split_lineage"),
        "quote_snapshot_id": stable_quote_hash(quote),
    }
    records = ensure_valuation(
        ticker,
        as_of_date,
        provider=None,
        run_id=run_id,
        price_override=float(price),
        force_refresh=True,
        cfg=cfg,
        db_path=cfg.db_path,
        raise_on_error=True,
        issuer_cik=str(quote.get("issuer_cik") or ""),
        issuer_aliases=issuer_aliases,
        require_filed_asof=True,
        price_provenance=quote_provenance,
    )
    if not records:
        raise RuntimeError(f"valuation bootstrap produced no controlled rows for {ticker}")

    final_scope = _integrity_scope(
        as_of_date=as_of_date,
        packets=packets,
        scenarios=scenarios,
    )
    integrity_result = _require_applied_financial_integrity_scope(final_scope)
    selection = {
        "source": "relight_valuation_bootstrap",
        "source_relight_id": relight_id,
        "requested_tickers": [ticker],
        "loaded_tickers": [ticker],
        "selected_tickers": [ticker],
        "cap_classifications": context.issuer_contexts,
        "data_gap_repair": {
            "status": "VALUATION_BOOTSTRAP_WRITTEN",
            "candidate_states": [
                {
                    "ticker": ticker,
                    "valuation_source_records": records,
                }
            ],
        },
        "financial_integrity": _financial_integrity_result_payload(integrity_result),
        "financial_integrity_binding": _financial_integrity_run_binding(
            final_scope,
            scope_fingerprint=integrity_result.scope_fingerprint,
        ),
    }
    completed_at = _utc_iso()
    artifact = AutonomousSectorFinancialRunArtifact(
        run_id=run_id,
        sector=sector,
        market_cap_focus=market_cap_focus,
        objective=VALUATION_BOOTSTRAP_OBJECTIVE,
        as_of_date=as_of_date,
        created_at=_utc_iso(now),
        completed_at=completed_at,
        status="COMPLETED",
        final_verdict="NO_SELECTION",
        selected_ticker=None,
        confidence="LOW",
        scan_family=VALUATION_BOOTSTRAP_SCAN_FAMILY,
        candidate_selection=selection,
        company_packets=packets,
        expected_return_scenarios=scenarios,
        tool_calls=[
            ToolCallRecord(
                call_id="TC1",
                tool_name="compute_deterministic_valuation",
                tool_input={"ticker": ticker, "as_of_date": as_of_date},
                rationale=(
                    "Compute the first point-in-time valuation rows from cached "
                    "facts and an authoritative proof-carrying quote."
                ),
                status="OK",
                started_at=_utc_iso(now),
                completed_at=completed_at,
                output_preview=(
                    f"Published {len(records)} controlled deterministic valuation rows."
                ),
            )
        ],
        provider_usage=[],
        no_selection_reason=(
            "Bootstrap artifacts publish valuation evidence only; paid reasoning "
            "must consume the authorized rows in a later run."
        ),
        audit_notes=[
            "No LLM or live provider was invoked.",
            "The existing autonomous-sector publication and completion binder "
            "authorized the exact deterministic writer rows.",
        ],
    )
    paths = persist_autonomous_sector_run(
        artifact,
        valuation_source_records=records,
    )
    bound_methods = _assert_published_valuation_bindings(
        records=records,
        run_id=run_id,
        artifact_path=paths.artifact_json,
        cfg=cfg,
    )
    authorized = _existing_authorized_scorecard(
        ticker=ticker,
        as_of_date=as_of_date,
        cfg=cfg,
    )
    if authorized is None:
        raise RuntimeError(f"valuation bootstrap failed to bind an authorized scorecard for {ticker}")
    return {
        "ticker": ticker,
        "status": "PUBLISHED",
        "run_id": run_id,
        "artifact_path": str(paths.artifact_json.resolve()),
        "bound_methods": bound_methods,
    }


def bootstrap_relight_valuations(
    *,
    relight_id: str,
    output_root: str | Path | None = None,
    db_path: str | Path | None = None,
    cfg: AppConfig | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Publish first-run valuation rows for every split-evidence READY name."""

    base_cfg = cfg or get_config()
    resolved_cfg = (
        base_cfg.model_copy(update={"db_path": Path(db_path)})
        if db_path is not None and Path(base_cfg.db_path) != Path(db_path)
        else base_cfg
    )
    plan, checkpoint_path = _read_relight_plan(relight_id, output_root=output_root)
    as_of_date = str(plan.get("effective_as_of") or "").strip()[:10]
    if not as_of_date:
        raise RelightPlanError("relight checkpoint has no effective as-of date")
    cells = [
        dict(cell)
        for cell in plan.get("cells") or []
        if isinstance(cell, dict)
    ]
    tickers = sorted(
        {
            str(ticker).strip().upper()
            for cell in cells
            for ticker in cell.get("tickers") or []
            if str(ticker).strip()
        }
    )
    if not tickers:
        raise RelightPlanError("relight checkpoint has no routed tickers")
    _assert_valuation_schema(resolved_cfg)
    emit = progress or (lambda _message: None)
    split_summary = prepare_relight_split_lineage_evidence(
        tickers=tickers,
        as_of_date=as_of_date,
        db_path=resolved_cfg.db_path,
        cfg=resolved_cfg,
        fetch_bytes=_offline_fetch_refusal,
    )
    ready = {
        str(row.get("ticker") or "").strip().upper()
        for row in split_summary.get("results") or []
        if isinstance(row, dict) and row.get("status") == "READY"
    }
    cell_by_ticker = {
        str(ticker).strip().upper(): cell
        for cell in cells
        for ticker in cell.get("tickers") or []
        if str(ticker).strip()
    }
    results: list[dict[str, Any]] = []
    for index, ticker in enumerate(tickers, start=1):
        if ticker not in ready:
            split_row = next(
                (
                    row
                    for row in split_summary.get("results") or []
                    if isinstance(row, dict)
                    and str(row.get("ticker") or "").strip().upper() == ticker
                ),
                {},
            )
            results.append(
                {
                    "ticker": ticker,
                    "status": "NEEDS_DATA",
                    "reason": str(split_row.get("reason") or "SPLIT_LINEAGE_UNKNOWN"),
                }
            )
            continue
        cell = cell_by_ticker[ticker]
        emit(f"valuation bootstrap {index}/{len(tickers)} {ticker}")
        try:
            results.append(
                _bootstrap_one(
                    relight_id=relight_id,
                    sector=str(cell.get("sector") or ""),
                    market_cap_focus=str(cell.get("band") or ""),
                    ticker=ticker,
                    as_of_date=as_of_date,
                    cfg=resolved_cfg,
                )
            )
        except Exception as exc:  # noqa: BLE001 - isolate failures by ticker
            results.append(
                {
                    "ticker": ticker,
                    "status": "FAILED",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )

    counts = {
        status: sum(row["status"] == status for row in results)
        for status in ("PUBLISHED", "ALREADY_AUTHORIZED", "NEEDS_DATA", "FAILED")
    }
    return {
        "schema_version": VALUATION_BOOTSTRAP_SCHEMA_VERSION,
        "relight_id": relight_id,
        "source_checkpoint_path": str(checkpoint_path.resolve()),
        "as_of_date": as_of_date,
        "requested": len(tickers),
        "split_ready": int(split_summary.get("ready") or 0),
        "split_unknown": int(split_summary.get("unknown") or 0),
        **counts,
        "provider_calls": 0,
        "llm_cost_usd": 0.0,
        "results": results,
    }


__all__ = [
    "VALUATION_BOOTSTRAP_RUN_PREFIX",
    "VALUATION_BOOTSTRAP_SCAN_FAMILY",
    "VALUATION_BOOTSTRAP_SCHEMA_VERSION",
    "bootstrap_relight_valuations",
]
