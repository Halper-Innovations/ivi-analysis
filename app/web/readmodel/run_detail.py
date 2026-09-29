"""One sector run parsed into the funnel, gate X-ray, cost panel, and links.

The artifact is the book of record here — ``run_index`` (ui.db) only resolves
``run_id`` to a path. Two pipeline generations coexist on disk:

* **v2** runs carry ``candidate_dispositions[]`` (terminal-state funnel) with
  ``screen_result.gate_evaluations[]`` per candidate — the gate X-ray.
* **v1** runs (the ~277-run majority) predate dispositions: their narrowing is
  ``candidate_selection`` → ``company_packets`` → ``relative_ranking`` →
  verdict, so the funnel is built from those stages instead.

Cost telemetry (``lane_budget`` / ``lane_usage`` / ``provider_usage``) is
empty ``{}`` / ``[]`` on every v2 artifact currently on disk — the reader
must treat "no telemetry" as a first-class state, not an error.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from app.autonomous.artifact_financial_audit import (
    PASS as FINANCIAL_INTEGRITY_PASS,
    STALE_AUDIT as FINANCIAL_INTEGRITY_STALE_AUDIT,
    authorized_artifact_bytes,
)
from app.web.readmodel.md import render_markdown
from app.web.readmodel.runs_index import _extract_summary
from app.watchlist.lineage import watchlist_row_is_decision_eligible

# v2 terminal states in pipeline order — the narrowing waterfall.
V2_FUNNEL_STAGES = (
    ("OUT_OF_SCOPE", "out of scope"),
    ("SCREENED_OUT", "screened out"),
    ("NEEDS_DATA", "needs data"),
    ("READY_FOR_UNDERWRITING", "ready for underwriting"),
    ("UNDERWRITTEN", "underwritten"),
)


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _stringify(value: Any) -> str | None:
    """Gate observed/threshold values may be strings, numbers, or bools."""
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if _str_or_none(item)]


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    return float(value) if isinstance(value, (int, float)) else None


def _bool_or_none(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _suppress_summary_decision_fields(summary: dict[str, Any]) -> None:
    for field in (
        "decision_status",
        "final_verdict",
        "selected_ticker",
        "no_selection_reason",
        "disposition_counts_json",
    ):
        summary[field] = None


def resolve_indexed_run(ui_conn: sqlite3.Connection, ref: str) -> dict[str, Any] | None:
    """Resolve a run reference — the unique path slug first, run_id second.

    Embedded run_ids repeat across v2 smoke/replay trees; when a bare run_id
    matches several artifacts the newest created_at wins only when every
    matching candidate is readable and authorized. One unreadable or
    unauthorized same-ID candidate makes the bare-ID lookup fail closed onto
    that quarantined path. Slug links remain exact and path-specific.
    """
    from app.web.readmodel.runs_index import artifact_slug

    candidates: list[tuple[float, str, dict[str, Any], bool]] = []
    exact_slug: dict[str, Any] | None = None
    for raw_row in ui_conn.execute("SELECT * FROM run_index").fetchall():
        row = dict(raw_row)
        path = Path(str(row.get("path") or ""))
        stat = None
        try:
            stat = path.stat()
        except OSError:
            pass
        current_slug = artifact_slug(path)
        current_run_id: str | None = None
        current_created_at: str | None = None
        current_sha256: str | None = None
        current_bytes: bytes | None = None
        current_payload: dict[str, Any] | None = None
        try:
            current_bytes = path.read_bytes()
            current_sha256 = hashlib.sha256(current_bytes).hexdigest()
            raw_payload = json.loads(current_bytes.decode("utf-8"))
            if isinstance(raw_payload, dict):
                current_payload = raw_payload
                current_run_id = _str_or_none(raw_payload.get("run_id"))
                current_created_at = _str_or_none(raw_payload.get("created_at"))
        except OSError:
            pass
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass
        if current_created_at is not None:
            try:
                rank = datetime.fromisoformat(current_created_at.replace("Z", "+00:00")).timestamp()
            except ValueError:
                rank = stat.st_mtime_ns / 1_000_000_000 if stat is not None else float("-inf")
        else:
            cached_created_at = _str_or_none(row.get("created_at"))
            try:
                rank = (
                    datetime.fromisoformat(cached_created_at.replace("Z", "+00:00")).timestamp()
                    if cached_created_at is not None
                    else (stat.st_mtime_ns / 1_000_000_000 if stat is not None else float("-inf"))
                )
            except ValueError:
                rank = stat.st_mtime_ns / 1_000_000_000 if stat is not None else float("-inf")

        row["slug"] = current_slug
        row["run_id"] = current_run_id
        row["created_at"] = current_created_at
        row["_resolved_current_sha256"] = current_sha256
        if current_slug == ref:
            row["_resolved_ref"] = ref
            row["_resolved_ref_kind"] = "slug"
            exact_slug = row
            continue

        cached_run_id = _str_or_none(raw_row["run_id"])
        if current_run_id == ref or (current_run_id is None and cached_run_id == ref):
            # Cached run_id is used only to keep an unreadable newest candidate
            # in the suppression set. It never supplies returned decision data.
            integrity_status, authorized_bytes = authorized_artifact_bytes(path)
            safe = (
                current_payload is not None
                and current_run_id == ref
                and current_bytes is not None
                and integrity_status == FINANCIAL_INTEGRITY_PASS
                and authorized_bytes is not None
                and hmac.compare_digest(
                    hashlib.sha256(current_bytes).hexdigest(),
                    hashlib.sha256(authorized_bytes).hexdigest(),
                )
            )
            row["_resolved_ref"] = ref
            row["_resolved_ref_kind"] = "run_id"
            candidates.append((rank, str(path), row, safe))

    if exact_slug is not None:
        return exact_slug
    if not candidates:
        return None
    unsafe = [candidate for candidate in candidates if not candidate[3]]
    _, _, selected, _ = max(unsafe or candidates, key=lambda item: (item[0], item[1]))
    return selected


def _authorized_payload_matches_resolution(
    index_row: dict[str, Any],
    payload: dict[str, Any],
    authorized_bytes: bytes,
) -> bool:
    """Bind the selected current-byte candidate to the exact authorized bytes."""

    resolved_sha256 = index_row.get("_resolved_current_sha256")
    if not isinstance(resolved_sha256, str) or not hmac.compare_digest(
        resolved_sha256,
        hashlib.sha256(authorized_bytes).hexdigest(),
    ):
        return False
    resolved_ref = _str_or_none(index_row.get("_resolved_ref"))
    resolved_kind = index_row.get("_resolved_ref_kind")
    if resolved_ref is None:
        return False
    if resolved_kind == "run_id":
        return _str_or_none(payload.get("run_id")) == resolved_ref
    if resolved_kind == "slug":
        from app.web.readmodel.runs_index import artifact_slug

        return artifact_slug(Path(str(index_row.get("path") or ""))) == resolved_ref
    return False


# ---------------------------------------------------------------- funnel


def _funnel_v2(dispositions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_state: dict[str, list[str]] = {}
    for item in dispositions:
        if not isinstance(item, dict):
            continue
        state = str(item.get("terminal_state") or "UNKNOWN")
        ticker = _str_or_none(item.get("ticker")) or _str_or_none(item.get("primary_ticker"))
        by_state.setdefault(state, []).append(ticker or "?")
    stages = []
    known = {key for key, _ in V2_FUNNEL_STAGES}
    for key, label in V2_FUNNEL_STAGES:
        tickers = sorted(by_state.get(key, []))
        if tickers:
            stages.append({"key": key, "label": label, "count": len(tickers), "tickers": tickers})
    for key in sorted(set(by_state) - known):
        tickers = sorted(by_state[key])
        stages.append(
            {
                "key": key,
                "label": key.replace("_", " ").lower(),
                "count": len(tickers),
                "tickers": tickers,
            }
        )
    return stages


def _funnel_v1(payload: dict[str, Any]) -> list[dict[str, Any]]:
    selection = payload.get("candidate_selection") or {}
    loaded = _str_list(selection.get("loaded_tickers") if isinstance(selection, dict) else None)
    examined = sorted(
        {
            str(packet.get("ticker"))
            for packet in payload.get("company_packets") or []
            if isinstance(packet, dict) and _str_or_none(packet.get("ticker"))
        }
    )
    ranked = sorted(
        {
            str(row.get("ticker"))
            for row in payload.get("relative_ranking") or []
            if isinstance(row, dict) and _str_or_none(row.get("ticker"))
        }
    )
    selected = _str_or_none(payload.get("selected_ticker"))
    stages = []
    for key, label, tickers in (
        ("LOADED", "loaded", sorted(loaded)),
        ("EXAMINED", "examined", examined),
        ("RANKED", "ranked", ranked),
        ("SELECTED", "selected", [selected] if selected else []),
    ):
        if tickers or key == "SELECTED":
            stages.append({"key": key, "label": label, "count": len(tickers), "tickers": tickers})
    return stages


# ------------------------------------------------------------ candidates


def _gate(evaluation: dict[str, Any]) -> dict[str, Any]:
    return {
        "rule_id": _str_or_none(evaluation.get("rule_id")) or "UNKNOWN_RULE",
        "status": _str_or_none(evaluation.get("status")) or "UNKNOWN",
        "applicable": bool(evaluation.get("applicable", True)),
        "observed_value": _stringify(evaluation.get("observed_value")),
        "threshold": _stringify(evaluation.get("threshold")),
        "reason_code": _str_or_none(evaluation.get("reason_code")),
        "evidence_url": _str_or_none(evaluation.get("evidence_url")),
        "notes": _str_list(evaluation.get("notes")),
    }


def _candidates_v2(dispositions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = []
    for item in dispositions:
        if not isinstance(item, dict):
            continue
        screen_result = item.get("screen_result")
        gates = []
        screen_reason_codes: list[str] = []
        if isinstance(screen_result, dict):
            gates = [
                _gate(evaluation)
                for evaluation in screen_result.get("gate_evaluations") or []
                if isinstance(evaluation, dict)
            ]
            screen_reason_codes = _str_list(screen_result.get("reason_codes"))
        ticker = _str_or_none(item.get("ticker")) or _str_or_none(item.get("primary_ticker"))
        candidates.append(
            {
                "ticker": ticker or "?",
                "primary_ticker": _str_or_none(item.get("primary_ticker")),
                "terminal_state": _str_or_none(item.get("terminal_state")) or "UNKNOWN",
                "last_completed_stage": _str_or_none(item.get("last_completed_stage")),
                "scope_status": _str_or_none(item.get("scope_status")),
                "screen_status": _str_or_none(item.get("screen_status")),
                "review_status": _str_or_none(item.get("review_status")),
                "reason_codes": sorted(
                    set(_str_list(item.get("reason_codes")) + screen_reason_codes)
                ),
                "security_type": _str_or_none(item.get("security_type")),
                "is_adr": _bool_or_none(item.get("is_adr")),
                "is_secondary_class": _bool_or_none(item.get("is_secondary_class")),
                "frontier_status": _str_or_none(item.get("frontier_status")),
                "frontier_dominated_by": _str_or_none(item.get("frontier_dominated_by")),
                "underwriting_verdict": _str_or_none(item.get("underwriting_verdict")),
                "underwriting_confidence": _str_or_none(item.get("underwriting_confidence")),
                "watchlist_eligible": _bool_or_none(item.get("watchlist_eligible")),
                "gates": gates,
                "failed_gates": sum(
                    1 for gate in gates if gate["status"] == "FAIL" and gate["applicable"]
                ),
            }
        )
    order = {key: index for index, (key, _) in enumerate(V2_FUNNEL_STAGES)}
    candidates.sort(key=lambda c: (order.get(c["terminal_state"], 99), c["ticker"]))
    return candidates


# ------------------------------------------------------- v1 ranking/packets


def _ranking_v1(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in payload.get("relative_ranking") or []:
        if not isinstance(row, dict) or not _str_or_none(row.get("ticker")):
            continue
        rows.append(
            {
                "rank": _int_or_none(row.get("rank")),
                "ticker": str(row.get("ticker")),
                "audit_status": _str_or_none(row.get("audit_status")),
                "actionable": _bool_or_none(row.get("actionable")),
                "buy_candidate": _bool_or_none(row.get("buy_candidate")),
                "buy_candidate_reason": _str_or_none(row.get("buy_candidate_reason")),
                "hard_blockers": _str_list(row.get("hard_blockers")),
                "cross_sectional_rank": _int_or_none(row.get("cross_sectional_rank")),
                "cross_sectional_percentile": _float_or_none(row.get("cross_sectional_percentile")),
                "best_base_annualized_return": _float_or_none(
                    row.get("best_base_annualized_return")
                ),
                "positioning_summary": _str_or_none(row.get("positioning_summary")),
            }
        )
    rows.sort(key=lambda r: (r["rank"] is None, r["rank"] if r["rank"] is not None else 0))
    return rows


def _packets_v1(payload: dict[str, Any]) -> list[dict[str, Any]]:
    packets = []
    for packet in payload.get("company_packets") or []:
        if not isinstance(packet, dict) or not _str_or_none(packet.get("ticker")):
            continue
        valuation = packet.get("valuation") if isinstance(packet.get("valuation"), dict) else {}
        packets.append(
            {
                "ticker": str(packet.get("ticker")),
                "blockers": _str_list(packet.get("blockers")),
                "data_quality_status": _str_or_none(packet.get("data_quality_status")),
                "financial_status": _str_or_none(packet.get("financial_status")),
                "model_fit_status": _str_or_none(packet.get("model_fit_status")),
                "market_cap_mm": _float_or_none(packet.get("market_cap_mm")),
                "market_cap_category": _str_or_none(packet.get("market_cap_category")),
                "anchor_method": _str_or_none(valuation.get("anchor_method")),
                "current_price": _float_or_none(valuation.get("current_price")),
                "discount_to_anchor": _float_or_none(valuation.get("discount_to_anchor")),
            }
        )
    packets.sort(key=lambda p: p["ticker"])
    return packets


# ----------------------------------------------------------------- costs


def _microdollars_to_usd(value: int | None) -> float | None:
    return round(value / 1_000_000, 6) if isinstance(value, int) else None


def _lane_rows(lane_usage: Any, lane_budget: Any) -> list[dict[str, Any]]:
    usage_lanes = lane_usage.get("lanes") if isinstance(lane_usage, dict) else None
    budget_lanes = lane_budget.get("lanes") if isinstance(lane_budget, dict) else None
    usage_lanes = usage_lanes if isinstance(usage_lanes, dict) else {}
    budget_lanes = budget_lanes if isinstance(budget_lanes, dict) else {}
    rows = []
    for lane in list(usage_lanes) + [lane for lane in budget_lanes if lane not in usage_lanes]:
        usage = usage_lanes.get(lane) if isinstance(usage_lanes.get(lane), dict) else {}
        budget = budget_lanes.get(lane) if isinstance(budget_lanes.get(lane), dict) else {}
        cost = _int_or_none(usage.get("cost_microdollars"))
        cap = _int_or_none(budget.get("max_cost_microdollars"))
        rows.append(
            {
                "lane": str(lane),
                "cost_microdollars": cost,
                "cost_usd": _microdollars_to_usd(cost),
                "max_cost_microdollars": cap,
                "max_cost_usd": _microdollars_to_usd(cap),
                "tool_call_attempts": _int_or_none(usage.get("tool_call_attempts")),
                "provider_call_attempts": _int_or_none(usage.get("provider_call_attempts")),
                "input_tokens": _int_or_none(usage.get("input_tokens")),
                "cached_input_tokens": _int_or_none(usage.get("cached_input_tokens")),
                "output_tokens": _int_or_none(usage.get("output_tokens")),
            }
        )
    return rows


def _provider_rows(provider_usage: Any) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for record in provider_usage if isinstance(provider_usage, list) else []:
        if not isinstance(record, dict):
            continue
        key = (str(record.get("provider") or "unknown"), str(record.get("model") or "unknown"))
        row = grouped.setdefault(
            key,
            {
                "provider": key[0],
                "model": key[1],
                "calls": 0,
                "ok_calls": 0,
                "failed_calls": 0,
                "input_tokens": 0,
                "cached_input_tokens": 0,
                "output_tokens": 0,
                "cost_estimate_usd": 0.0,
            },
        )
        row["calls"] += 1
        if str(record.get("status") or "").upper() == "OK":
            row["ok_calls"] += 1
        else:
            row["failed_calls"] += 1
        for field in ("input_tokens", "cached_input_tokens", "output_tokens"):
            value = _int_or_none(record.get(field))
            if value is not None:
                row[field] += value
        cost = record.get("cost_estimate_usd")
        if isinstance(cost, (int, float)) and not isinstance(cost, bool):
            row["cost_estimate_usd"] += float(cost)
    rows = sorted(grouped.values(), key=lambda r: -r["cost_estimate_usd"])
    for row in rows:
        row["cost_estimate_usd"] = round(row["cost_estimate_usd"], 6)
    return rows


def _costs(payload: dict[str, Any]) -> dict[str, Any]:
    lane_usage = payload.get("lane_usage")
    lanes = _lane_rows(lane_usage, payload.get("lane_budget"))
    providers = _provider_rows(payload.get("provider_usage"))
    aggregate = lane_usage.get("aggregate") if isinstance(lane_usage, dict) else None
    aggregate_cost = (
        _int_or_none(aggregate.get("cost_microdollars")) if isinstance(aggregate, dict) else None
    )
    if aggregate_cost is None:
        lane_costs = [
            row["cost_microdollars"] for row in lanes if row["cost_microdollars"] is not None
        ]
        aggregate_cost = sum(lane_costs) if lane_costs else None
    return {
        "available": bool(lanes or providers),
        "aggregate_cost_microdollars": aggregate_cost,
        "aggregate_cost_usd": _microdollars_to_usd(aggregate_cost),
        "lanes": lanes,
        "providers": providers,
    }


# ---------------------------------------------------------------- detail


def _selection(payload: dict[str, Any]) -> dict[str, Any]:
    selection = payload.get("candidate_selection")
    selection = selection if isinstance(selection, dict) else {}
    return {
        "source": _str_or_none(selection.get("source")),
        "requested_tickers": _str_list(selection.get("requested_tickers")),
        "excluded_tickers": _str_list(selection.get("excluded_tickers")),
        "loaded_tickers": _str_list(selection.get("loaded_tickers")),
        "selected_tickers": _str_list(selection.get("selected_tickers")),
        "admitted_tickers": _str_list(payload.get("admitted_tickers")),
        "warnings": _str_list(selection.get("warnings")),
    }


def _decision(payload: dict[str, Any]) -> dict[str, Any]:
    final_decision = payload.get("final_decision")
    final_decision = final_decision if isinstance(final_decision, dict) else {}
    return {
        "status": _str_or_none(payload.get("status")),
        "execution_status": _str_or_none(payload.get("execution_status")),
        "decision_status": _str_or_none(payload.get("decision_status")),
        "final_verdict": _str_or_none(payload.get("final_verdict")),
        "selected_ticker": _str_or_none(payload.get("selected_ticker")),
        "no_selection_reason": _str_or_none(payload.get("no_selection_reason")),
        "confidence": _str_or_none(payload.get("confidence")),
        "confidence_cap_reasons": _str_list(final_decision.get("confidence_cap_reasons")),
        "memo_present": bool(_str_or_none(payload.get("memo_body"))),
    }


def _watchlist_links(engine_conn: sqlite3.Connection | None, run_id: str) -> list[dict[str, Any]]:
    if engine_conn is None:
        return []
    try:
        rows = engine_conn.execute(
            """
            WITH ranked AS (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY ticker
                           ORDER BY added_at DESC, id DESC
                       ) AS source_rank
                FROM watchlist
                WHERE source_run_id = ?
            )
            SELECT *
            FROM ranked
            WHERE source_rank = 1
            ORDER BY ticker
            """,
            (run_id,),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [
        {
            "watchlist_id": int(row["id"]),
            "ticker": str(row["ticker"]),
            "status": _str_or_none(row["status"]),
            "conviction_grade": _str_or_none(row["conviction_grade"]),
        }
        for row in rows
        if watchlist_row_is_decision_eligible(row)
    ]


def load_run_detail(
    ui_conn: sqlite3.Connection,
    ref: str,
    *,
    engine_conn: sqlite3.Connection | None = None,
) -> dict[str, Any] | None:
    """Full detail for one indexed run; None when the reference is unknown."""
    index_row = resolve_indexed_run(ui_conn, ref)
    if index_row is None:
        return None
    integrity_status, run_bytes = authorized_artifact_bytes(Path(str(index_row["path"])))
    payload: dict[str, Any] | None = None
    parse_error: str | None = None
    if integrity_status == FINANCIAL_INTEGRITY_PASS and run_bytes is not None:
        try:
            parsed = json.loads(run_bytes.decode("utf-8"))
            if not isinstance(parsed, dict):
                raise ValueError(f"artifact root is {type(parsed).__name__}, expected object")
            if not _authorized_payload_matches_resolution(index_row, parsed, run_bytes):
                integrity_status = FINANCIAL_INTEGRITY_STALE_AUDIT
                parse_error = (
                    "ResolutionError: selected path bytes changed or no longer match "
                    "the requested run reference"
                )
                index_row["parse_error"] = parse_error
            else:
                payload = parsed
                index_row.update(
                    _extract_summary(
                        payload,
                        Path(str(index_row["path"])),
                        integrity_status=integrity_status,
                    )
                )
                index_row["artifact_sha256"] = hashlib.sha256(run_bytes).hexdigest()
                index_row["parse_error"] = None
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
            parse_error = f"{type(exc).__name__}: {exc}"
            index_row["parse_error"] = parse_error
    decision_eligible = payload is not None
    index_row["integrity_status"] = integrity_status
    index_row["decision_eligible"] = int(decision_eligible)
    if not decision_eligible:
        _suppress_summary_decision_fields(index_row)
    for internal_field in (
        "_resolved_ref",
        "_resolved_ref_kind",
        "_resolved_current_sha256",
    ):
        index_row.pop(internal_field, None)
    base = {
        "summary": index_row,
        "quarantined": parse_error is not None,
        "parse_error": parse_error,
        "funnel": [],
        "candidates": [],
        "ranking": [],
        "packets": [],
        "selection": _selection({}),
        "decision": _decision({}),
        "costs": _costs({}),
        "watchlist_rows": (
            _watchlist_links(engine_conn, str((payload or {}).get("run_id") or ""))
            if decision_eligible
            else []
        ),
        "report_available": bool(index_row.get("report_path")),
    }
    if base["quarantined"]:
        return base
    if not decision_eligible:
        return base
    assert payload is not None
    dispositions = payload.get("candidate_dispositions")
    if isinstance(dispositions, list) and dispositions:
        base["funnel"] = _funnel_v2(dispositions)
        base["candidates"] = _candidates_v2(dispositions)
    else:
        base["funnel"] = _funnel_v1(payload)
    base["ranking"] = _ranking_v1(payload)
    base["packets"] = _packets_v1(payload)
    base["selection"] = _selection(payload)
    base["decision"] = _decision(payload)
    base["costs"] = _costs(payload)
    return base


def load_run_report(ui_conn: sqlite3.Connection, ref: str) -> dict[str, Any] | None:
    """The run's markdown report rendered for the Reader; None when absent."""
    index_row = resolve_indexed_run(ui_conn, ref)
    if index_row is None:
        return None
    artifact_path = index_row.get("path")
    if not artifact_path:
        return None
    artifact_path = Path(str(artifact_path))
    path = artifact_path.parent / "autonomous_sector_report.md"
    if not path.is_file():
        return None
    run_integrity_status, run_bytes = authorized_artifact_bytes(artifact_path)
    report_integrity_status, report_bytes = authorized_artifact_bytes(path)
    current_payload: dict[str, Any] | None = None
    if run_integrity_status == FINANCIAL_INTEGRITY_PASS and run_bytes is not None:
        try:
            parsed = json.loads(run_bytes.decode("utf-8"))
            if isinstance(parsed, dict) and _authorized_payload_matches_resolution(
                index_row,
                parsed,
                run_bytes,
            ):
                current_payload = parsed
            elif isinstance(parsed, dict):
                run_integrity_status = FINANCIAL_INTEGRITY_STALE_AUDIT
        except (UnicodeDecodeError, json.JSONDecodeError):
            current_payload = None
    integrity_status = (
        run_integrity_status
        if run_integrity_status != FINANCIAL_INTEGRITY_PASS
        else report_integrity_status
    )
    decision_eligible = (
        run_integrity_status == FINANCIAL_INTEGRITY_PASS
        and current_payload is not None
        and report_integrity_status == FINANCIAL_INTEGRITY_PASS
        and report_bytes is not None
    )
    if not decision_eligible:
        text = (
            "# Historical artifact — excluded from current decisions\n\n"
            f"Financial-integrity status: **{integrity_status}**. The original report "
            "body is suppressed because this historical run is not decision-eligible."
        )
    else:
        try:
            assert report_bytes is not None
            text = report_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return None
    return {
        "run_id": str((current_payload or {}).get("run_id") or index_row.get("run_id") or ref),
        "slug": str(index_row.get("slug") or ref),
        "path": str(path),
        "bytes": len(text.encode("utf-8")),
        "html": render_markdown(text),
        "integrity_status": integrity_status,
        "decision_eligible": decision_eligible,
    }
