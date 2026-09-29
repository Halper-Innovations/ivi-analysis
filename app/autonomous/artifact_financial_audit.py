"""Read-only financial-integrity audit and product eligibility overlay.

Historical artifacts are evidence and are never rewritten by this module.  A
tree audit records the SHA-256 of every inspected artifact in a timestamped
manifest.  Current-product readers can then reject an artifact whose audited
record is invalid or whose bytes have changed since the audit.

The payload path is intentionally legacy-compatible.  It understands the v1
``company_packets`` representation as well as the stricter packet contract
used by newer runs, and it does not require deserialising either contract.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4


AUDIT_SCHEMA_VERSION = "financial_integrity_audit_v1"
AUDIT_FILENAME_PREFIX = "financial_integrity_audit_"
AUDIT_SCOPE_ID = "ivi_current_decision_artifacts_v1"
RUN_AUTHORIZATION_SCHEMA_VERSION = "financial_integrity_run_authorization_v1"
RUN_AUTHORIZATION_FILENAME = "financial_integrity_authorization.json"
TYPED_AUTHORIZATION_SCHEMA_VERSION = "financial_integrity_typed_authorization_v1"
TYPED_AUTHORIZATION_SUFFIX = ".financial_integrity_authorization.json"
DIGEST_LINEAGE_SCHEMA_VERSION = "financial_integrity_digest_lineage_v4"
DIGEST_RENDERED_STATE_SCHEMA_VERSION = "financial_integrity_digest_rendered_state_v2"
_DIGEST_RENDERED_STATE_MARKER = "IVI_DIGEST_RENDERED_STATE_V2:"
_FINANCIAL_INTEGRITY_BLOCKED_DIGEST_MESSAGE = (
    "**BLOCKED: the canonical financial-integrity audit manifest is missing, "
    "malformed, or unreadable. Current decision rows are suppressed.**"
)
DIGEST_DECISION_STATE_FIELDS = (
    "id",
    "ticker",
    "status",
    "conviction_grade",
    "confidence",
    "conviction_source",
    "scan_family",
    "valuation_anchor_method",
    "valuation_anchor_value",
    "buy_price_target",
    "current_price_at_addition",
    "thesis_text",
    "key_risks_json",
    "falsifiers_json",
    "open_questions_json",
    "source_run_id",
    "source_sector",
    "added_at",
    "last_evaluated_at",
    "status_reason",
    "market_cap_mm",
    "cap_source",
    "cap_band",
    "cap_asof",
    "event_pending",
    "adv_dollar_20d",
    "adv_dollar_60d",
    "adv_asof",
    "capacity_class",
    "pipeline_version",
    "candidate_disposition",
    "decision_basis",
    "selection_validation_status",
)

CANONICAL_AUDIT_ROOT_IDS = {
    "autonomous_sector": "autonomous_sector_runs",
    "analyst_output": "analyst_outputs",
    "scan": "scan_outputs",
    "research_output": "research_outputs",
    "watchlist_report": "watchlist_reports",
}

PASS = "PASS"
INVALID = "INVALID"
UNAUDITED = "UNAUDITED"
STALE_AUDIT = "STALE_AUDIT"
ELIGIBILITY_STATES = (PASS, INVALID, UNAUDITED, STALE_AUDIT)

SAFE_DETERMINISTIC_RERENDER = "SAFE_DETERMINISTIC_RERENDER"
REQUIRES_PACKET_REBUILD = "REQUIRES_PACKET_REBUILD"
REQUIRES_LLM_REREVIEW = "REQUIRES_LLM_REREVIEW"
MISSING_PROVENANCE_UNREPAIRABLE = "MISSING_PROVENANCE_UNREPAIRABLE"
REPAIR_CLASSIFICATIONS = (
    SAFE_DETERMINISTIC_RERENDER,
    REQUIRES_PACKET_REBUILD,
    REQUIRES_LLM_REREVIEW,
    MISSING_PROVENANCE_UNREPAIRABLE,
)

_NUMERIC_REL_TOL = 1e-6
_QUOTE_REL_TOL = 1e-3
_DATE_RE = re.compile(r"(?P<date>20\d{2}-\d{2}-\d{2})")
_SCAN_FINANCIAL_TEXT_RE = re.compile(
    r"(?im)\b(?:current_price|market_cap(?:_mm)?|fcf_yield|dcf_base|"
    r"epv_adjusted|ev/ebitda|p/e|price/book)\s*:"
)
_SCAN_FINANCIAL_KEYS = {
    "adjusted_dcf",
    "adjusted_epv",
    "adjusted_intrinsic_mid",
    "adjusted_margin_of_safety",
    "base_case_value",
    "bear_case_value",
    "bull_case_value",
    "current_price",
    "price",
    "market_cap",
    "market_cap_mm",
    "fcf_yield",
    "dcf_base",
    "epv_adjusted",
    "ev_ebitda",
    "pe_ratio",
    "price_to_book",
    "price_to_p15",
    "p15_per_share",
    "original_dcf",
    "original_epv",
    "original_graham",
    "scorecard_dcf",
    "scorecard_epv",
    "scorecard_graham",
    "scorecard_price",
}


def _canonical_audit_roots() -> dict[str, Path]:
    """Return the only artifact roots authorized for the active IVI runtime."""

    from app.config import get_config

    cfg = get_config()
    return {
        "autonomous_sector": (Path(cfg.runs_dir) / "autonomous_sector").resolve(),
        "analyst_output": Path(cfg.analyst_outputs_dir).resolve(),
        "scan": (Path(cfg.outputs_dir) / "scans").resolve(),
        "research_output": Path(cfg.research_dir).resolve(),
        "watchlist_report": (Path(cfg.outputs_dir) / "digests").resolve(),
    }


def _canonical_audit_root_cache_key() -> tuple[tuple[str, str], ...]:
    """Bind manifest-derived caches to the active canonical runtime roots."""

    return tuple(sorted((family, str(path)) for family, path in _canonical_audit_roots().items()))


@dataclass(frozen=True)
class AuditReportPaths:
    manifest_json: Path
    report_markdown: Path


@dataclass(frozen=True)
class _ManifestSnapshot:
    path: Path
    sha256: str
    canonical_root_key: tuple[tuple[str, str], ...]
    payload: Mapping[str, Any]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _current_sha256(path: Path) -> str:
    """Hash the current bytes on every authorization check.

    Path/mtime/size caches are not an authorization boundary: a same-size
    replacement can restore its prior mtime.  Product eligibility therefore
    pays the bounded cost of hashing the exact bytes it authorizes.
    """

    return _sha256_file(path.resolve())


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _close(left: float, right: float, *, rel_tol: float = _NUMERIC_REL_TOL) -> bool:
    return math.isclose(left, right, rel_tol=rel_tol, abs_tol=1e-12)


def _material_quote_difference(left: float, right: float) -> bool:
    return not math.isclose(left, right, rel_tol=_QUOTE_REL_TOL, abs_tol=0.01)


def _normalized_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _llm_consumed(payload: Mapping[str, Any]) -> bool:
    """Return whether any recorded provider call consumed the artifact context."""

    stack: list[Any] = [payload]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            usage = current.get("provider_usage")
            if isinstance(usage, list) and any(
                isinstance(item, dict)
                and str(item.get("status") or "OK").upper() not in {"FAILED", "ERROR"}
                for item in usage
            ):
                return True
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)
    return False


def _repair_action(classification: str) -> str:
    if classification == SAFE_DETERMINISTIC_RERENDER:
        return (
            "Preserve the source artifact and publish a newly generated deterministic "
            "rendering that explicitly supersedes this hash."
        )
    if classification == REQUIRES_PACKET_REBUILD:
        return (
            "Exclude this hash from current decision surfaces and rebuild the packet from "
            "the original as-of quote, shares, split basis, and unit-labelled inputs."
        )
    if classification == REQUIRES_LLM_REREVIEW:
        return (
            "Exclude this hash and all dependent prose from current decision surfaces; "
            "after a deterministic packet rebuild, create a new owner-authorized LLM review."
        )
    return (
        "Preserve this artifact as visibly invalid audit history and exclude it from current "
        "decision surfaces because the original quote or split basis cannot be proven."
    )


def _violation(
    *,
    invariant: str,
    ticker: str | None,
    field: str,
    source_values: Mapping[str, Any],
    expected_relationship: str,
    observed_relationship: str,
    reason: str,
    llm_consumed: bool,
    repair_classification: str,
    run_id: str | None = None,
    as_of_date: str | None = None,
) -> dict[str, Any]:
    if repair_classification not in REPAIR_CLASSIFICATIONS:
        raise ValueError(f"unsupported repair classification: {repair_classification}")
    return {
        "invariant": invariant,
        "run_id": run_id,
        "ticker": ticker,
        "as_of_date": as_of_date,
        "field": field,
        "source_values": dict(source_values),
        "expected_relationship": expected_relationship,
        "observed_relationship": observed_relationship,
        "reason": reason,
        "llm_consumed": bool(llm_consumed),
        "repair_classification": repair_classification,
        "invalidation_action": _repair_action(repair_classification),
    }


def _cap_classifications(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    selection = payload.get("candidate_selection")
    if not isinstance(selection, dict):
        return {}
    raw = selection.get("cap_classifications")
    if not isinstance(raw, dict):
        return {}
    return {
        str(ticker).strip().upper(): item
        for ticker, item in raw.items()
        if str(ticker).strip() and isinstance(item, dict)
    }


def _nested_analyst_quotes(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Extract analyst-context prices embedded in company-autonomy evidence."""

    found: list[dict[str, Any]] = []
    seen: set[tuple[str, float, str | None]] = set()
    runs = payload.get("company_autonomy_runs")
    if not isinstance(runs, list):
        return found
    for run in runs:
        if not isinstance(run, dict):
            continue
        ticker = str(run.get("ticker") or "").strip().upper()
        nested = run.get("artifact")
        evidence = nested.get("evidence") if isinstance(nested, dict) else None
        if not ticker or not isinstance(evidence, list):
            continue
        for item in evidence:
            if not isinstance(item, dict):
                continue
            source_type = str(item.get("source_type") or "").lower()
            source_label = str(item.get("source_label") or "").lower()
            if source_type != "analysis_report" and source_label != "analyst_context":
                continue
            excerpt = item.get("excerpt")
            try:
                context = json.loads(excerpt) if isinstance(excerpt, str) else excerpt
            except json.JSONDecodeError:
                continue
            if not isinstance(context, dict):
                continue
            valuation = context.get("valuation")
            if not isinstance(valuation, dict):
                continue
            price = _as_float(valuation.get("price"))
            if price is None:
                price = _as_float(valuation.get("current_price"))
            if price is None:
                continue
            paths = context.get("paths")
            report_path = (
                _normalized_text(paths.get("analysis_report_json"))
                if isinstance(paths, dict)
                else None
            )
            key = (ticker, price, report_path)
            if key in seen:
                continue
            seen.add(key)
            found.append(
                {
                    "ticker": ticker,
                    "price": price,
                    "analysis_report_path": report_path,
                    "quote_source": _normalized_text(valuation.get("price_source")),
                    "quote_as_of_date": _normalized_text(valuation.get("price_as_of_date")),
                    "quote_currency": _normalized_text(valuation.get("price_currency")),
                }
            )
    return found


def _ratio_metrics_present(valuation: Mapping[str, Any]) -> bool:
    ratio_tokens = (
        "yield",
        "ratio",
        "margin",
        "multiple",
        "percentile",
        "discount",
        "return",
        "growth_dependency",
        "ev_to_",
        "price_to_",
    )
    stack: list[tuple[str, Any]] = [("valuation", valuation)]
    while stack:
        prefix, current = stack.pop()
        if not isinstance(current, dict):
            continue
        for key, value in current.items():
            field = f"{prefix}.{key}"
            if isinstance(value, dict):
                stack.append((field, value))
            elif _as_float(value) is not None and any(
                token in str(key).lower() for token in ratio_tokens
            ):
                return True
    return False


def _nonfinite_financial_fields(value: Any, *, prefix: str) -> list[str]:
    bad: list[str] = []
    stack: list[tuple[str, Any]] = [(prefix, value)]
    while stack:
        field, current = stack.pop()
        if isinstance(current, dict):
            stack.extend((f"{field}.{key}", item) for key, item in current.items())
        elif isinstance(current, list):
            stack.extend((f"{field}[{index}]", item) for index, item in enumerate(current))
        elif isinstance(current, float) and not math.isfinite(current):
            bad.append(field)
    return sorted(bad)


def _future_dated_fields(
    packet: Mapping[str, Any],
    cap: Mapping[str, Any],
    *,
    run_as_of_date: str | None,
) -> dict[str, str]:
    if not run_as_of_date:
        return {}
    try:
        cutoff = date.fromisoformat(run_as_of_date[:10])
    except ValueError:
        return {}
    candidates = {
        "current_price_as_of_date": packet.get("current_price_as_of_date"),
        "cap_stage_price_as_of_date": packet.get("cap_stage_price_as_of_date"),
        "market_cap_effective_as_of_date": packet.get("market_cap_effective_as_of_date"),
        "identity_as_of_date": packet.get("identity_as_of_date"),
        "shares_period_end": cap.get("shares_period_end"),
        "cap_price_as_of_date": cap.get("price_as_of_date"),
        "cap_effective_as_of_date": cap.get("cap_effective_as_of_date"),
    }
    future: dict[str, str] = {}
    for field, raw in candidates.items():
        text = str(raw or "").strip()[:10]
        if not text:
            continue
        try:
            observed = date.fromisoformat(text)
        except ValueError:
            continue
        if observed > cutoff:
            future[field] = text
    return future


def _missing_contract_fields(
    packet: Mapping[str, Any],
    valuation: Mapping[str, Any],
) -> list[str]:
    required = (
        "market_cap_unit",
        "quote_snapshot_id",
        "current_price_unit",
        "price_basis",
        "shares_unit",
        "shares_basis",
        "split_adjustment_factor",
    )
    missing = [
        field
        for field in required
        if packet.get(field) is None
        or (isinstance(packet.get(field), str) and not str(packet.get(field)).strip())
    ]
    traces = packet.get("metric_traces")
    if _ratio_metrics_present(valuation) and (not isinstance(traces, dict) or not traces):
        missing.append("metric_traces")
    return missing


def _invalid_contract_values(packet: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "market_cap_unit": {"USD_millions"},
        "current_price_unit": {"USD_per_share"},
        "price_basis": {"UNADJUSTED", "SPLIT_ADJUSTED"},
        "shares_unit": {"shares_millions"},
        # Historical packets used ISSUER_REPORTED as a basis token even
        # though it is provenance. Retain it for historical classification;
        # current producers emit the literal UNADJUSTED/SPLIT_ADJUSTED basis.
        "shares_basis": {"ISSUER_REPORTED", "UNADJUSTED", "SPLIT_ADJUSTED"},
    }
    invalid: dict[str, Any] = {}
    for field, allowed in expected.items():
        value = packet.get(field)
        if value is not None and str(value) not in allowed:
            invalid[field] = {"observed": value, "expected": sorted(allowed)}
    split_factor = packet.get("split_adjustment_factor")
    if split_factor is not None and (_as_float(split_factor) is None or float(split_factor) <= 0):
        invalid["split_adjustment_factor"] = {
            "observed": split_factor,
            "expected": "finite positive factor",
        }
    snapshot_id = _normalized_text(packet.get("quote_snapshot_id"))
    if snapshot_id:
        from app.autonomous.financial_integrity import stable_quote_snapshot_id

        expected_snapshot_id = stable_quote_snapshot_id(packet)
        if snapshot_id.lower() != expected_snapshot_id:
            invalid["quote_snapshot_id"] = {
                "observed": snapshot_id,
                "expected": expected_snapshot_id,
            }
    return invalid


def _invalid_metric_traces(packet: Mapping[str, Any]) -> list[str]:
    traces = packet.get("metric_traces")
    if not isinstance(traces, dict):
        return []
    invalid: list[str] = []
    for metric, trace in traces.items():
        if not isinstance(trace, dict):
            invalid.append(str(metric))
            continue
        output = _as_float(trace.get("output"))
        recomputed = _as_float(trace.get("recomputed_output"))
        if (
            trace.get("reconciles") is not True
            or output is None
            or recomputed is None
            or not _close(output, recomputed, rel_tol=1e-9)
        ):
            invalid.append(str(metric))
    return sorted(invalid)


def _sector_payload_violations(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    packets = payload.get("company_packets")
    if not isinstance(packets, list):
        return []
    run_id = _normalized_text(payload.get("run_id"))
    as_of_date = _normalized_text(payload.get("as_of_date"))
    consumed = _llm_consumed(payload)
    caps = _cap_classifications(payload)
    packet_by_ticker: dict[str, Mapping[str, Any]] = {}
    violations: list[dict[str, Any]] = []

    for packet in packets:
        if not isinstance(packet, dict):
            continue
        ticker = str(packet.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        packet_by_ticker[ticker] = packet
        valuation = packet.get("valuation")
        valuation = valuation if isinstance(valuation, dict) else {}
        cap = caps.get(ticker, {})

        missing_contract = _missing_contract_fields(packet, valuation)
        if missing_contract:
            violations.append(
                _violation(
                    invariant="FINANCIAL_CONTRACT_PROVENANCE_MISSING",
                    run_id=run_id,
                    ticker=ticker,
                    as_of_date=as_of_date,
                    field=f"company_packets[{ticker}].financial_contract",
                    source_values={"missing_fields": missing_contract},
                    expected_relationship=(
                        "packet explicitly identifies units, quote snapshot, price/share split bases, "
                        "split adjustment, and formula traces for ratio outputs"
                    ),
                    observed_relationship=f"missing_fields={missing_contract!r}",
                    reason="The packet lacks the explicit lineage required to prove its financial quantities.",
                    llm_consumed=consumed,
                    repair_classification=MISSING_PROVENANCE_UNREPAIRABLE,
                )
            )

        invalid_contract = _invalid_contract_values(packet)
        if invalid_contract:
            classification = REQUIRES_LLM_REREVIEW if consumed else REQUIRES_PACKET_REBUILD
            violations.append(
                _violation(
                    invariant="FINANCIAL_CONTRACT_VALUE_INVALID",
                    run_id=run_id,
                    ticker=ticker,
                    as_of_date=as_of_date,
                    field=f"company_packets[{ticker}].financial_contract",
                    source_values=invalid_contract,
                    expected_relationship="contract units, bases, split factor, and quote identity use canonical values",
                    observed_relationship="one or more explicit contract values are invalid",
                    reason="The packet's explicit financial lineage is internally invalid.",
                    llm_consumed=consumed,
                    repair_classification=classification,
                )
            )

        nonfinite_fields = _nonfinite_financial_fields(packet, prefix=f"company_packets[{ticker}]")
        if nonfinite_fields:
            classification = REQUIRES_LLM_REREVIEW if consumed else REQUIRES_PACKET_REBUILD
            violations.append(
                _violation(
                    invariant="NONFINITE_FINANCIAL_VALUE",
                    run_id=run_id,
                    ticker=ticker,
                    as_of_date=as_of_date,
                    field=f"company_packets[{ticker}]",
                    source_values={"nonfinite_fields": nonfinite_fields},
                    expected_relationship="all financial quantities are finite real numbers",
                    observed_relationship=f"nonfinite_fields={nonfinite_fields!r}",
                    reason="A NaN or infinite financial value cannot enter a product or reasoning boundary.",
                    llm_consumed=consumed,
                    repair_classification=classification,
                )
            )

        future_fields = _future_dated_fields(packet, cap, run_as_of_date=as_of_date)
        if future_fields:
            classification = REQUIRES_LLM_REREVIEW if consumed else REQUIRES_PACKET_REBUILD
            violations.append(
                _violation(
                    invariant="FUTURE_DATED_FINANCIAL_INPUT",
                    run_id=run_id,
                    ticker=ticker,
                    as_of_date=as_of_date,
                    field=f"company_packets[{ticker}].as_of_lineage",
                    source_values={"run_as_of_date": as_of_date, "future_fields": future_fields},
                    expected_relationship="every financial input date is at or before the run as-of date",
                    observed_relationship=f"future_fields={future_fields!r}",
                    reason="The packet contains look-ahead financial evidence.",
                    llm_consumed=consumed,
                    repair_classification=classification,
                )
            )

        bad_traces = _invalid_metric_traces(packet)
        if bad_traces:
            classification = REQUIRES_LLM_REREVIEW if consumed else REQUIRES_PACKET_REBUILD
            violations.append(
                _violation(
                    invariant="METRIC_TRACE_RECONCILIATION_MISMATCH",
                    run_id=run_id,
                    ticker=ticker,
                    as_of_date=as_of_date,
                    field=f"company_packets[{ticker}].metric_traces",
                    source_values={"invalid_metric_traces": bad_traces},
                    expected_relationship="every metric trace output equals its deterministic recomputation",
                    observed_relationship=f"invalid_metric_traces={bad_traces!r}",
                    reason="One or more formula lineage records do not reconcile.",
                    llm_consumed=consumed,
                    repair_classification=classification,
                )
            )

        observed_yield = _as_float(valuation.get("fcf_yield"))
        ttm_fcf_mm = _as_float(valuation.get("ttm_fcf"))
        current_price = _as_float(packet.get("current_price"))
        shares_mm = _as_float(cap.get("shares_mm"))
        market_cap_mm = _as_float(packet.get("market_cap_mm"))

        shares_for_cap = _as_float(packet.get("shares_outstanding_mm"))
        if shares_for_cap is None:
            shares_for_cap = _as_float(cap.get("shares_mm"))
        if (
            market_cap_mm is not None
            and current_price is not None
            and shares_for_cap is not None
            and current_price > 0
            and shares_for_cap > 0
        ):
            expected_market_cap_mm = current_price * shares_for_cap
            if not _close(market_cap_mm, expected_market_cap_mm, rel_tol=1e-6):
                classification = REQUIRES_LLM_REREVIEW if consumed else REQUIRES_PACKET_REBUILD
                violations.append(
                    _violation(
                        invariant="MARKET_CAP_RECONCILIATION_MISMATCH",
                        run_id=run_id,
                        ticker=ticker,
                        as_of_date=as_of_date,
                        field=f"company_packets[{ticker}].market_cap_mm",
                        source_values={
                            "market_cap_mm": market_cap_mm,
                            "current_price_usd_per_share": current_price,
                            "shares_outstanding_mm": shares_for_cap,
                            "expected_market_cap_mm": expected_market_cap_mm,
                        },
                        expected_relationship=(
                            "market_cap_mm = current_price_usd_per_share * shares_outstanding_mm"
                        ),
                        observed_relationship=(
                            f"market_cap_mm={market_cap_mm!r}; recomputed={expected_market_cap_mm!r}"
                        ),
                        reason="Market cap does not reconcile to the packet's quote and share basis.",
                        llm_consumed=consumed,
                        repair_classification=classification,
                    )
                )

        expected_candidates: list[tuple[str, float]] = []
        if (
            ttm_fcf_mm is not None
            and current_price is not None
            and shares_mm is not None
            and current_price > 0
            and shares_mm > 0
        ):
            expected_candidates.append(
                (
                    "ttm_fcf_mm / (current_price_usd * shares_mm)",
                    ttm_fcf_mm / (current_price * shares_mm),
                )
            )
        if ttm_fcf_mm is not None and market_cap_mm is not None and market_cap_mm > 0:
            expected_candidates.append(("ttm_fcf_mm / market_cap_mm", ttm_fcf_mm / market_cap_mm))

        if observed_yield is not None and expected_candidates:
            million_basis = next(
                (
                    (relationship, expected)
                    for relationship, expected in expected_candidates
                    if _close(observed_yield, expected * 1_000_000.0)
                ),
                None,
            )
            if million_basis is not None:
                relationship, expected = million_basis
                classification = REQUIRES_LLM_REREVIEW if consumed else REQUIRES_PACKET_REBUILD
                violations.append(
                    _violation(
                        invariant="FCF_YIELD_USD_VS_USD_MILLIONS",
                        run_id=run_id,
                        ticker=ticker,
                        as_of_date=as_of_date,
                        field=f"company_packets[{ticker}].valuation.fcf_yield",
                        source_values={
                            "ttm_fcf_mm": ttm_fcf_mm,
                            "current_price_usd": current_price,
                            "shares_mm": shares_mm,
                            "market_cap_mm": market_cap_mm,
                            "observed_fcf_yield": observed_yield,
                            "expected_fcf_yield": expected,
                            "observed_to_expected": observed_yield / expected if expected else None,
                        },
                        expected_relationship=f"fcf_yield = {relationship}",
                        observed_relationship="stored fcf_yield = expected fcf_yield * 1,000,000",
                        reason="A USD-versus-USD-millions scale factor was applied to a dimensionless yield.",
                        llm_consumed=consumed,
                        repair_classification=classification,
                    )
                )
            elif not missing_contract and not any(
                _close(observed_yield, expected, rel_tol=1e-4)
                for _, expected in expected_candidates
            ):
                relationship, expected = expected_candidates[0]
                classification = REQUIRES_LLM_REREVIEW if consumed else REQUIRES_PACKET_REBUILD
                violations.append(
                    _violation(
                        invariant="FCF_YIELD_RECONCILIATION_MISMATCH",
                        run_id=run_id,
                        ticker=ticker,
                        as_of_date=as_of_date,
                        field=f"company_packets[{ticker}].valuation.fcf_yield",
                        source_values={
                            "ttm_fcf_mm": ttm_fcf_mm,
                            "current_price_usd": current_price,
                            "shares_mm": shares_mm,
                            "market_cap_mm": market_cap_mm,
                            "observed_fcf_yield": observed_yield,
                            "expected_fcf_yield": expected,
                        },
                        expected_relationship=f"fcf_yield = {relationship}",
                        observed_relationship=f"stored fcf_yield={observed_yield!r}",
                        reason="The stored yield does not reconcile with any explicit packet denominator.",
                        llm_consumed=consumed,
                        repair_classification=classification,
                    )
                )

        valuation_price = _as_float(valuation.get("current_price"))
        if (
            current_price is not None
            and valuation_price is not None
            and _material_quote_difference(current_price, valuation_price)
        ):
            classification = REQUIRES_LLM_REREVIEW if consumed else REQUIRES_PACKET_REBUILD
            violations.append(
                _violation(
                    invariant="PACKET_VALUATION_QUOTE_MISMATCH",
                    run_id=run_id,
                    ticker=ticker,
                    as_of_date=as_of_date,
                    field=f"company_packets[{ticker}].valuation.current_price",
                    source_values={
                        "packet_current_price": current_price,
                        "valuation_current_price": valuation_price,
                    },
                    expected_relationship="packet current price and valuation current price share one quote snapshot",
                    observed_relationship=f"{current_price!r} != {valuation_price!r}",
                    reason="The packet carries two materially different current prices.",
                    llm_consumed=consumed,
                    repair_classification=classification,
                )
            )

        cap_price = _as_float(packet.get("cap_stage_price"))
        current_date = _normalized_text(packet.get("current_price_as_of_date"))
        cap_date = _normalized_text(packet.get("cap_stage_price_as_of_date"))
        current_source = _normalized_text(packet.get("current_price_source"))
        cap_source = _normalized_text(packet.get("cap_stage_price_source"))
        current_currency = _normalized_text(packet.get("current_price_currency"))
        cap_currency = _normalized_text(packet.get("cap_stage_price_currency"))
        same_claimed_snapshot = (
            current_date is not None
            and current_date == cap_date
            and current_source is not None
            and current_source == cap_source
            and current_currency is not None
            and current_currency == cap_currency
        )
        if (
            same_claimed_snapshot
            and current_price is not None
            and cap_price is not None
            and _material_quote_difference(current_price, cap_price)
        ):
            classification = REQUIRES_LLM_REREVIEW if consumed else REQUIRES_PACKET_REBUILD
            violations.append(
                _violation(
                    invariant="QUOTE_SNAPSHOT_MISMATCH",
                    run_id=run_id,
                    ticker=ticker,
                    as_of_date=as_of_date,
                    field=f"company_packets[{ticker}].current_price",
                    source_values={
                        "current_price": current_price,
                        "cap_stage_price": cap_price,
                        "price_as_of_date": current_date,
                        "price_source": current_source,
                        "price_currency": current_currency,
                    },
                    expected_relationship="equal source/date/currency quote identities imply one numeric price",
                    observed_relationship=f"current_price={current_price!r}; cap_stage_price={cap_price!r}",
                    reason="Two materially different prices claim the same quote snapshot identity.",
                    llm_consumed=consumed,
                    repair_classification=classification,
                )
            )

    for context in _nested_analyst_quotes(payload):
        ticker = str(context["ticker"])
        packet = packet_by_ticker.get(ticker)
        if packet is None:
            continue
        packet_price = _as_float(packet.get("current_price"))
        analyst_price = _as_float(context.get("price"))
        if (
            packet_price is None
            or analyst_price is None
            or not _material_quote_difference(packet_price, analyst_price)
        ):
            continue
        violations.append(
            _violation(
                invariant="NESTED_ANALYST_QUOTE_MISMATCH",
                run_id=run_id,
                ticker=ticker,
                as_of_date=as_of_date,
                field=f"company_autonomy_runs[{ticker}].analyst_context.valuation.price",
                source_values={
                    "sector_packet_price": packet_price,
                    "nested_analyst_price": analyst_price,
                    "analyst_report_path": context.get("analysis_report_path"),
                    "analyst_quote_source": context.get("quote_source"),
                    "analyst_quote_as_of_date": context.get("quote_as_of_date"),
                    "analyst_quote_currency": context.get("quote_currency"),
                },
                expected_relationship="nested analyst and sector decision contexts identify one proven quote snapshot",
                observed_relationship=f"sector_packet_price={packet_price!r}; nested_analyst_price={analyst_price!r}",
                reason="LLM underwriting consumed an analyst price that cannot be reconciled to the sector packet quote.",
                llm_consumed=True,
                repair_classification=REQUIRES_LLM_REREVIEW,
            )
        )
    return violations


def audit_payload(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return deterministic violations for an in-memory artifact payload."""

    violations = _sector_payload_violations(payload)
    if isinstance(payload.get("company_packets"), list):
        return violations
    if (
        isinstance(payload.get("valuation"), Mapping)
        and _normalized_text(payload.get("ticker")) is not None
        and _normalized_text(payload.get("as_of_date")) is not None
    ):
        violations.extend(_analyst_payload_violations(payload))
    if any(key in payload for key in ("triage", "deep_reviews", "ranked")):
        violations.extend(_scan_payload_violations(payload))
    if any(
        key in payload
        for key in (
            "analyst_notes",
            "hypotheses_generated",
            "merged_findings",
            "opus_validation",
            "thesis",
        )
    ):
        violations.extend(_research_payload_violations(payload))
    return violations


def _coerce_payload(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        return payload if isinstance(payload, Mapping) else None
    return None


def _latest_manifest_path() -> Path | None:
    configured = os.getenv("VOE_FINANCIAL_INTEGRITY_MANIFEST", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    try:
        from app.config import get_config

        analysis_dir = Path(get_config().outputs_dir) / "analysis"
    except Exception:
        return None
    try:
        candidates = list(analysis_dir.glob(f"{AUDIT_FILENAME_PREFIX}*.json"))
    except OSError:
        return None
    # Audit filenames carry a fixed-width UTC timestamp.  Select the newest
    # canonical file without parsing it: silently falling back to an older
    # valid audit when the newest file is malformed or unreadable would make
    # every downstream decision gate fail open.
    return max(candidates, default=None, key=lambda path: path.name)


def _strict_nonnegative_int(value: Any) -> bool:
    return type(value) is int and value >= 0


def _manifest_artifact_file_identities_are_distinct(
    payload: Mapping[str, Any],
) -> bool:
    """Reject audited path aliases that share one underlying filesystem file."""

    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        return False
    identities: set[tuple[int, int]] = set()
    for record in artifacts:
        if not isinstance(record, Mapping):
            return False
        raw_path = _normalized_text(record.get("path"))
        if raw_path is None:
            return False
        try:
            artifact_path = Path(raw_path).expanduser()
            if _exact_lexical_path(artifact_path) != artifact_path:
                return False
            file_stat = artifact_path.stat()
        except FileNotFoundError:
            # A completed manifest remains structurally usable after an
            # audited artifact disappears so readers can classify that exact
            # record (and anything depending on it) as STALE_AUDIT. Missing
            # bytes cannot alias another current file.
            continue
        except (OSError, RuntimeError, TypeError, ValueError):
            return False
        identity = (file_stat.st_dev, file_stat.st_ino)
        if identity in identities:
            return False
        identities.add(identity)
    return True


def _manifest_census_is_coherent(payload: Mapping[str, Any]) -> bool:
    """Validate that a completed manifest represents a non-empty audit census.

    Product readers treat this file as an authorization boundary, so container
    types alone are insufficient: an empty or internally contradictory
    manifest must never mean "nothing is invalid."  The checks below mirror
    the deterministic shape emitted by :func:`audit_artifact_tree`.
    """

    if payload.get("audit_scope_id") != AUDIT_SCOPE_ID:
        return False

    source_roots = payload.get("source_roots")
    summary = payload.get("summary")
    artifacts = payload.get("artifacts")
    violations = payload.get("violations")
    invalid_run_ids = payload.get("invalid_run_ids")
    if not (
        isinstance(source_roots, list)
        and source_roots
        and isinstance(summary, dict)
        and isinstance(artifacts, list)
        and artifacts
        and isinstance(violations, list)
        and isinstance(invalid_run_ids, list)
    ):
        return False
    if not _manifest_artifact_file_identities_are_distinct(payload):
        return False
    normalized_roots: list[tuple[str, Path]] = []
    declared_root_ids: set[str] = set()
    declared_root_paths: set[str] = set()
    for root in source_roots:
        if not isinstance(root, dict):
            return False
        family = _normalized_text(root.get("family"))
        root_id = _normalized_text(root.get("root_id"))
        root_text = _normalized_text(root.get("path"))
        if (
            family is None
            or not isinstance(root.get("family"), str)
            or root_id is None
            or not isinstance(root.get("root_id"), str)
            or root_text is None
            or not isinstance(root.get("path"), str)
            or CANONICAL_AUDIT_ROOT_IDS.get(family) != root_id
            or root_id in declared_root_ids
        ):
            return False
        root_path = Path(root_text).expanduser()
        if not root_path.is_absolute() or not root_path.is_dir():
            return False
        resolved_root = root_path.resolve()
        normalized_root_text = str(resolved_root)
        if normalized_root_text in declared_root_paths:
            return False
        declared_root_ids.add(root_id)
        declared_root_paths.add(normalized_root_text)
        normalized_roots.append((family, resolved_root))
    if declared_root_ids != set(CANONICAL_AUDIT_ROOT_IDS.values()):
        return False
    expected_roots = _canonical_audit_roots()
    if {family: path for family, path in normalized_roots} != expected_roots:
        return False

    artifacts_scanned = summary.get("artifacts_scanned")
    violation_count = summary.get("violations")
    if (
        not _strict_nonnegative_int(artifacts_scanned)
        or artifacts_scanned != len(artifacts)
        or artifacts_scanned == 0
        or not _strict_nonnegative_int(violation_count)
        or violation_count != len(violations)
    ):
        return False

    artifact_index: dict[tuple[str, str], Mapping[str, Any]] = {}
    artifact_paths: set[str] = set()
    invalid_artifact_run_ids: set[str] = set()
    pass_primary_run_ids: set[str] = set()
    for record in artifacts:
        if not isinstance(record, dict):
            return False
        raw_path = _normalized_text(record.get("path"))
        sha256 = _normalized_text(record.get("sha256"))
        status = record.get("integrity_status")
        decision_eligible = record.get("decision_eligible")
        family = record.get("family")
        run_id_value = record.get("run_id")
        artifact_path = Path(raw_path).expanduser().resolve() if raw_path is not None else None
        normalized_artifact_path = str(artifact_path) if artifact_path is not None else None
        if (
            raw_path is None
            or not isinstance(record.get("path"), str)
            or not Path(raw_path).expanduser().is_absolute()
            or raw_path != normalized_artifact_path
            or sha256 is None
            or not isinstance(record.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", sha256) is None
            or not isinstance(family, str)
            or _normalized_text(family) is None
            or (run_id_value is not None and not isinstance(run_id_value, str))
            or status not in {PASS, INVALID}
            or type(decision_eligible) is not bool
            or decision_eligible is not (status == PASS)
            or normalized_artifact_path in artifact_paths
        ):
            return False
        assert artifact_path is not None
        if not any(
            family == root_family and artifact_path.is_relative_to(root_path)
            for root_family, root_path in normalized_roots
        ):
            return False
        artifact_paths.add(str(artifact_path))
        artifact_index[(str(artifact_path), sha256)] = record
        run_id = _normalized_text(record.get("run_id"))
        if (
            family == "autonomous_sector"
            and artifact_path.name == "autonomous_sector_run.json"
            and status == PASS
        ):
            expected_primary = _canonical_autonomous_primary_path(
                run_id,
                root=expected_roots["autonomous_sector"],
            )
            if (
                run_id is None
                or artifact_path != expected_primary
                or run_id in pass_primary_run_ids
            ):
                return False
            pass_primary_run_ids.add(run_id)
        if status == INVALID and run_id is not None:
            invalid_artifact_run_ids.add(run_id)

    if any(not isinstance(item, dict) for item in violations):
        return False
    actual_invariant_counts = Counter(str(item.get("invariant") or "") for item in violations)
    if "" in actual_invariant_counts:
        return False
    expected_invariant_counts = summary.get("violations_by_invariant")
    if (
        not isinstance(expected_invariant_counts, dict)
        or any(
            not isinstance(key, str) or not _strict_nonnegative_int(value)
            for key, value in expected_invariant_counts.items()
        )
        or dict(actual_invariant_counts) != dict(expected_invariant_counts)
    ):
        return False

    violation_run_ids: set[str] = set()
    violation_tickers: set[str] = set()
    violated_artifacts: set[tuple[str, str]] = set()
    llm_consumed_count = 0
    for violation in violations:
        artifact_path = _normalized_text(violation.get("artifact_path"))
        artifact_sha256 = _normalized_text(violation.get("artifact_sha256"))
        try:
            normalized_violation_path = (
                str(Path(artifact_path).expanduser().resolve())
                if artifact_path is not None
                else None
            )
        except (OSError, RuntimeError):
            return False
        if artifact_path != normalized_violation_path:
            return False
        artifact_key = (normalized_violation_path, artifact_sha256)
        artifact_record = artifact_index.get(artifact_key)
        if artifact_record is None or artifact_record.get("integrity_status") != INVALID:
            return False
        violated_artifacts.add(artifact_key)
        run_id = _normalized_text(violation.get("run_id"))
        ticker = _normalized_text(violation.get("ticker"))
        if run_id is not None:
            violation_run_ids.add(run_id)
        if ticker is not None:
            violation_tickers.add(ticker)
        llm_consumed_count += bool(violation.get("llm_consumed"))

    invalid_artifacts = {
        key for key, record in artifact_index.items() if record.get("integrity_status") == INVALID
    }
    if invalid_artifacts != violated_artifacts:
        return False

    if any(not isinstance(value, str) for value in invalid_run_ids):
        return False
    declared_run_ids = [_normalized_text(value) for value in invalid_run_ids]
    summary_run_ids = summary.get("affected_run_id_values")
    if (
        any(value is None for value in declared_run_ids)
        or len(set(declared_run_ids)) != len(declared_run_ids)
        or not isinstance(summary_run_ids, list)
        or any(not isinstance(value, str) for value in summary_run_ids)
        or declared_run_ids != [_normalized_text(value) for value in summary_run_ids]
        or set(declared_run_ids) != violation_run_ids | invalid_artifact_run_ids
        or not _strict_nonnegative_int(summary.get("affected_run_ids"))
        or summary.get("affected_run_ids") != len(declared_run_ids)
    ):
        return False

    summary_tickers = summary.get("affected_ticker_values")
    if (
        not isinstance(summary_tickers, list)
        or any(not isinstance(value, str) for value in summary_tickers)
        or any(_normalized_text(value) is None for value in summary_tickers)
        or len({_normalized_text(value) for value in summary_tickers}) != len(summary_tickers)
        or {_normalized_text(value) for value in summary_tickers} != violation_tickers
        or not _strict_nonnegative_int(summary.get("affected_tickers"))
        or summary.get("affected_tickers") != len(summary_tickers)
        or not _strict_nonnegative_int(summary.get("tickers_scanned"))
        or not _strict_nonnegative_int(summary.get("llm_consumed_violation_count"))
        or summary.get("llm_consumed_violation_count") != llm_consumed_count
        or summary.get("source_artifacts_rewritten") != 0
    ):
        return False
    return True


@lru_cache(maxsize=8)
def _read_manifest_cached(
    path_text: str,
    expected_sha256: str,
    canonical_root_key: tuple[tuple[str, str], ...],
) -> Mapping[str, Any] | None:
    del canonical_root_key  # Included in the cache key; coherence reads current roots below.
    path = Path(path_text)
    try:
        data = path.read_bytes()
        if _sha256_bytes(data) != expected_sha256:
            return None
        payload = json.loads(data.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("schema_version") != AUDIT_SCHEMA_VERSION or payload.get("complete") is not True:
        return None
    if not isinstance(payload.get("generated_at"), str):
        return None
    if not _manifest_census_is_coherent(payload):
        return None
    return payload


def _load_manifest_snapshot(path: Path | None) -> _ManifestSnapshot | None:
    if path is None or not path.is_file():
        return None
    try:
        resolved_path = path.expanduser().resolve()
        canonical_root_key = _canonical_audit_root_cache_key()
        manifest_sha256 = _current_sha256(resolved_path)
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
    payload = _read_manifest_cached(
        str(resolved_path),
        manifest_sha256,
        canonical_root_key,
    )
    if payload is None:
        return None
    return _ManifestSnapshot(
        path=resolved_path,
        sha256=manifest_sha256,
        canonical_root_key=canonical_root_key,
        payload=payload,
    )


def _load_manifest(path: Path | None) -> Mapping[str, Any] | None:
    snapshot = _load_manifest_snapshot(path)
    if snapshot is None or not _manifest_snapshot_is_current(snapshot, path):
        return None
    return snapshot.payload


def financial_integrity_manifest_is_usable(
    manifest_path: str | Path | None = None,
) -> bool:
    """Whether the canonical audit manifest is readable and authorizing.

    Current decision-product readers must check this before selecting rows.
    A missing, unreadable, malformed, incomplete, empty-census, or wrong-schema
    manifest is an unavailable audit, never an empty invalidation set.
    """

    selected = active_financial_integrity_manifest_path(manifest_path)
    snapshot = _load_manifest_snapshot(selected)
    return snapshot is not None and _manifest_snapshot_is_current(
        snapshot,
        manifest_path,
    )


@lru_cache(maxsize=8)
def _manifest_artifact_indexes(
    manifest_path: str,
    manifest_sha256: str,
    canonical_root_key: tuple[tuple[str, str], ...],
) -> dict[str, Mapping[str, Any]]:
    manifest = _read_manifest_cached(
        manifest_path,
        manifest_sha256,
        canonical_root_key,
    )
    by_path: dict[str, Mapping[str, Any]] = {}
    artifacts = manifest.get("artifacts") if manifest is not None else None
    if isinstance(artifacts, list):
        for item in artifacts:
            if not isinstance(item, dict):
                continue
            raw_path = _normalized_text(item.get("path"))
            if raw_path:
                by_path[raw_path] = item
    return by_path


def _exact_lexical_path(path: str | Path) -> Path | None:
    """Return one normalized absolute path only when it is not an alias."""

    try:
        expanded = Path(path).expanduser()
        lexical = Path(os.path.abspath(os.fspath(expanded)))
        resolved = expanded.resolve()
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
    return lexical if lexical == resolved else None


def active_financial_integrity_manifest_path(
    manifest_path: str | Path | None = None,
) -> Path | None:
    """Return the configured/discovered manifest path, if auditing is active.

    An explicitly configured path is returned even when missing or malformed
    so callers can fail closed instead of silently disabling their gate.
    """

    if manifest_path is not None:
        return Path(manifest_path).expanduser().resolve()
    configured = os.getenv("VOE_FINANCIAL_INTEGRITY_MANIFEST", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return _latest_manifest_path()


def _manifest_snapshot_is_current(
    snapshot: _ManifestSnapshot,
    manifest_path: str | Path | None = None,
) -> bool:
    """Require the same active manifest bytes and canonical roots as loaded."""

    selected = active_financial_integrity_manifest_path(manifest_path)
    if selected is None:
        return False
    try:
        if selected.expanduser().resolve() != snapshot.path:
            return False
        if _canonical_audit_root_cache_key() != snapshot.canonical_root_key:
            return False
        manifest_unchanged = _current_sha256(snapshot.path) == snapshot.sha256
        distinct_artifacts = _manifest_artifact_file_identities_are_distinct(snapshot.payload)
        return manifest_unchanged and distinct_artifacts
    except (OSError, RuntimeError, TypeError, ValueError):
        return False


def _manifest_root_for_family(
    manifest: Mapping[str, Any],
    family: str,
) -> Path | None:
    roots = manifest.get("source_roots")
    if not isinstance(roots, list):
        return None
    matches = [
        Path(str(item["path"])).expanduser().resolve()
        for item in roots
        if isinstance(item, dict) and item.get("family") == family
    ]
    return matches[0] if len(matches) == 1 else None


def _canonical_autonomous_primary_path(
    run_id: str | None,
    *,
    root: Path,
) -> Path | None:
    """Return the sole canonical primary path for one serialized run identity."""

    normalized_run_id = _normalized_text(run_id)
    if (
        normalized_run_id is None
        or normalized_run_id in {".", ".."}
        or Path(normalized_run_id).name != normalized_run_id
    ):
        return None
    return root / normalized_run_id / "autonomous_sector_run.json"


def _single_canonical_primary_record(
    records: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
    manifest: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Require exactly one primary record at the serialized run's canonical path."""

    root = _manifest_root_for_family(manifest, "autonomous_sector")
    expected = _canonical_autonomous_primary_path(run_id, root=root) if root is not None else None
    primaries = [
        record
        for record in records
        if record.get("family") == "autonomous_sector"
        and Path(str(record.get("path") or "")).name == "autonomous_sector_run.json"
    ]
    if expected is None or len(primaries) != 1:
        return None
    primary = primaries[0]
    raw_path = _normalized_text(primary.get("path"))
    if raw_path is None:
        return None
    try:
        path = Path(raw_path).expanduser()
        if raw_path != str(path) or _exact_lexical_path(path) != path or path != expected:
            return None
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
    return primary


def _canonical_run_report_bytes(payload: Mapping[str, Any]) -> bytes | None:
    """Render the only report bytes that may be authorized for a run artifact."""

    try:
        from app.autonomous.sector_contract import AutonomousSectorFinancialRunArtifact
        from app.autonomous.sector_report import render_autonomous_sector_report

        artifact = AutonomousSectorFinancialRunArtifact.from_dict(dict(payload))
        return render_autonomous_sector_report(artifact).encode("utf-8")
    except (KeyError, TypeError, ValueError):
        return None


def _validated_run_authorization_records(
    authorization_path: Path,
    *,
    manifest: Mapping[str, Any],
    expected_run_id: str | None = None,
) -> dict[str, Mapping[str, Any]] | None:
    """Validate a post-write run authorization against exact current bytes."""

    root = _manifest_root_for_family(manifest, "autonomous_sector")
    if root is None:
        return None
    resolved_authorization = authorization_path.expanduser().resolve()
    if (
        resolved_authorization.name != RUN_AUTHORIZATION_FILENAME
        or not resolved_authorization.is_file()
        or not resolved_authorization.is_relative_to(root)
    ):
        return None
    try:
        payload = json.loads(resolved_authorization.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    run_id = _normalized_text(payload.get("run_id"))
    if (
        payload.get("schema_version") != RUN_AUTHORIZATION_SCHEMA_VERSION
        or payload.get("complete") is not True
        or not isinstance(payload.get("generated_at"), str)
        or run_id is None
        or (expected_run_id is not None and run_id != expected_run_id)
        or resolved_authorization.parent.name != run_id
        or resolved_authorization.parent.parent != root
    ):
        return None
    raw_records = payload.get("artifacts")
    if not isinstance(raw_records, list) or len(raw_records) != 2:
        return None
    expected_names = {
        "autonomous_sector_run.json": "artifact_json",
        "autonomous_sector_report.md": "report_markdown",
    }
    validated: dict[str, Mapping[str, Any]] = {}
    artifact_payload: Mapping[str, Any] | None = None
    report_bytes: bytes | None = None
    for record in raw_records:
        if not isinstance(record, dict):
            return None
        raw_path = _normalized_text(record.get("path"))
        sha256 = _normalized_text(record.get("sha256"))
        kind = _normalized_text(record.get("kind"))
        if raw_path is None or sha256 is None or kind is None:
            return None
        path = Path(raw_path).expanduser().resolve()
        if (
            raw_path != str(path)
            or path.parent != resolved_authorization.parent
            or expected_names.get(path.name) != kind
            or kind in validated
            or re.fullmatch(r"[0-9a-f]{64}", sha256) is None
            or not path.is_file()
        ):
            return None
        try:
            data = path.read_bytes()
        except OSError:
            return None
        if _sha256_bytes(data) != sha256:
            return None
        if kind == "artifact_json":
            try:
                parsed_payload = json.loads(data.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return None
            if (
                not isinstance(parsed_payload, dict)
                or _normalized_text(parsed_payload.get("run_id")) != run_id
                or audit_payload(parsed_payload)
            ):
                return None
            artifact_payload = parsed_payload
        else:
            report_bytes = data
        validated[kind] = record
    if (
        set(validated) != set(expected_names.values())
        or artifact_payload is None
        or report_bytes is None
        or report_bytes != _canonical_run_report_bytes(artifact_payload)
    ):
        return None
    return validated


def write_run_financial_authorization(
    artifact_path: str | Path,
    report_path: str | Path,
    *,
    generated_at: datetime | None = None,
    published_parent: str | Path | None = None,
) -> Path:
    """Audit exact post-write bytes and atomically persist their authorization."""

    artifact = Path(artifact_path).expanduser().resolve()
    report = Path(report_path).expanduser().resolve()
    if (
        artifact.name != "autonomous_sector_run.json"
        or report.name != "autonomous_sector_report.md"
        or artifact.parent != report.parent
    ):
        raise ValueError("Run authorization requires the canonical artifact/report pair.")
    try:
        artifact_bytes = artifact.read_bytes()
        report_bytes = report.read_bytes()
        artifact_payload = json.loads(artifact_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot authorize unreadable run output: {exc}") from exc
    if not isinstance(artifact_payload, dict):
        raise RuntimeError("Cannot authorize a run artifact whose JSON root is not an object.")
    run_id = _normalized_text(artifact_payload.get("run_id"))
    violations = audit_payload(artifact_payload)
    if run_id is None or violations:
        invariants = sorted({str(item.get("invariant") or "") for item in violations})
        raise RuntimeError(
            "Cannot authorize financially invalid post-write run bytes"
            + (f": {', '.join(invariants)}" if invariants else ".")
        )
    expected_report_bytes = _canonical_run_report_bytes(artifact_payload)
    if (
        artifact.parent.name != run_id
        or expected_report_bytes is None
        or report_bytes != expected_report_bytes
    ):
        raise RuntimeError("Run output paths/content do not match the serialized run identity.")
    published_dir = (
        Path(published_parent).expanduser().resolve()
        if published_parent is not None
        else artifact.parent
    )
    if published_dir.name != run_id:
        raise RuntimeError("Published run directory does not match the serialized run identity.")
    now = generated_at or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    authorization = {
        "schema_version": RUN_AUTHORIZATION_SCHEMA_VERSION,
        "generated_at": now.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "complete": True,
        "run_id": run_id,
        "artifacts": [
            {
                "kind": "artifact_json",
                "path": str(published_dir / artifact.name),
                "sha256": _sha256_bytes(artifact_bytes),
            },
            {
                "kind": "report_markdown",
                "path": str(published_dir / report.name),
                "sha256": _sha256_bytes(report_bytes),
            },
        ],
    }
    destination = artifact.parent / RUN_AUTHORIZATION_FILENAME
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(authorization, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def _postwrite_records_for_run(
    run_id: str,
    *,
    manifest: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]] | None:
    root = _manifest_root_for_family(manifest, "autonomous_sector")
    if root is None:
        return None
    authorization_path = root / run_id / RUN_AUTHORIZATION_FILENAME
    return _validated_run_authorization_records(
        authorization_path,
        manifest=manifest,
        expected_run_id=run_id,
    )


def _typed_authorization_path(artifact_path: Path) -> Path:
    return artifact_path.with_name(f"{artifact_path.stem}{TYPED_AUTHORIZATION_SUFFIX}")


def _typed_artifact_root(artifact_type: str) -> Path | None:
    """Resolve the bounded runtime root for one supported typed artifact."""

    if artifact_type not in {
        "grade_status_calibration",
        "discovery_calibration",
    }:
        return None
    from app.config import get_config

    return Path(get_config().calibration_dir).expanduser().resolve()


_GRADE_STATUS_CALIBRATION_FIELDS = frozenset(
    {
        "run_id",
        "report_family",
        "as_of_date",
        "generated_at",
        "headline_hit_metric",
        "hit_metrics",
        "avoid_sign_inverted",
        "source_run_ids",
        "source_outcomes",
        "by_grade",
        "by_status",
        "by_decision_basis",
        "overall",
    }
)

_DISCOVERY_CALIBRATION_FIELDS = frozenset(
    {
        "run_id",
        "report_family",
        "target_run_id",
        "generated_at",
        "discovery_run_ids",
        "candidate_count",
        "outcomes_total",
        "excluded_outcomes_total",
        "matched_outcomes",
        "closed_outcomes_total",
        "source_run_ids",
        "source_outcomes",
        "source_candidates",
        "average_closed_return_pct",
        "hit_rate_by_stage",
        "hit_rate_by_whale_fit_bucket",
        "hit_rate_by_evidence_strength_bucket",
        "top_reason_counts",
        "top_gap_counts",
        "threshold_suggestions",
    }
)


def _grade_status_calibration_payload_is_authorizable(
    payload: Mapping[str, Any],
    *,
    expected_run_id: str,
) -> bool:
    from app.calibration.calibration_report import recompute_grade_status_calibration_claims
    from app.outcomes.lineage import serialized_outcome_binding_is_decision_eligible

    if (
        set(payload) != _GRADE_STATUS_CALIBRATION_FIELDS
        or _normalized_text(payload.get("run_id")) != expected_run_id
        or payload.get("report_family") != "grade_status_calibration"
        or _normalized_text(payload.get("as_of_date")) is None
        or not isinstance(payload.get("overall"), dict)
    ):
        return False
    source_run_ids = payload.get("source_run_ids")
    if not isinstance(source_run_ids, list):
        return False
    normalized_source_run_ids = [_normalized_text(run_id) for run_id in source_run_ids]
    if (
        any(run_id is None for run_id in normalized_source_run_ids)
        or len(set(normalized_source_run_ids)) != len(normalized_source_run_ids)
        or normalized_source_run_ids != sorted(normalized_source_run_ids)
    ):
        return False
    source_outcomes = payload.get("source_outcomes")
    if not isinstance(source_outcomes, list):
        return False
    source_keys: list[tuple[str, str, int, str]] = []
    for binding in source_outcomes:
        if not isinstance(binding, Mapping) or not serialized_outcome_binding_is_decision_eligible(
            binding
        ):
            return False
        state = binding.get("outcome_state")
        if not isinstance(state, Mapping):
            return False
        run_id = _normalized_text(state.get("run_id"))
        ticker = _normalized_text(state.get("ticker"))
        fingerprint = _normalized_text(binding.get("financial_integrity_fingerprint"))
        outcome_id = state.get("id")
        if (
            run_id is None
            or ticker is None
            or fingerprint is None
            or not isinstance(outcome_id, int)
        ):
            return False
        source_keys.append((run_id, ticker.upper(), outcome_id, fingerprint))
    if source_keys != sorted(source_keys) or len(set(source_keys)) != len(source_keys):
        return False
    bound_run_ids = sorted({key[0] for key in source_keys})
    recomputed = recompute_grade_status_calibration_claims(source_outcomes)
    return (
        bound_run_ids == normalized_source_run_ids
        and recomputed is not None
        and payload.get("headline_hit_metric") == "excess_return_pct>0"
        and payload.get("hit_metrics")
        == ["excess_return_pct>0", "realized_return_pct>0", "reached_buy_target"]
        and payload.get("avoid_sign_inverted") is True
        and all(payload.get(field) == value for field, value in recomputed.items())
    )


def _discovery_calibration_payload_is_authorizable(
    payload: Mapping[str, Any],
    *,
    expected_run_id: str,
) -> bool:
    from app.discovery.calibration import recompute_discovery_calibration_claims
    from app.discovery.lineage import (
        serialized_discovery_candidate_binding_is_structurally_valid,
    )

    if (
        set(payload) != _DISCOVERY_CALIBRATION_FIELDS
        or _normalized_text(payload.get("run_id")) != expected_run_id
        or payload.get("report_family") != "discovery_calibration"
        or not isinstance(payload.get("hit_rate_by_stage"), dict)
        or not isinstance(payload.get("threshold_suggestions"), list)
    ):
        return False
    if not _outcome_source_bindings_are_authorizable(payload):
        return False
    source_candidates = payload.get("source_candidates")
    discovery_run_ids = payload.get("discovery_run_ids")
    if not isinstance(source_candidates, list) or not isinstance(discovery_run_ids, list):
        return False
    candidate_keys: list[tuple[str, str, int, str]] = []
    for binding in source_candidates:
        if not serialized_discovery_candidate_binding_is_structurally_valid(binding):
            return False
        state = binding["candidate_state"]
        candidate_keys.append(
            (
                str(state["run_id"]),
                str(state["ticker"]),
                int(state["id"]),
                str(binding["candidate_state_sha256"]),
            )
        )
    if candidate_keys != sorted(candidate_keys) or len(candidate_keys) != len(set(candidate_keys)):
        return False
    normalized_discovery_run_ids = [_normalized_text(run_id) for run_id in discovery_run_ids]
    if (
        any(run_id is None for run_id in normalized_discovery_run_ids)
        or len(set(normalized_discovery_run_ids)) != len(normalized_discovery_run_ids)
        or normalized_discovery_run_ids != sorted(normalized_discovery_run_ids)
    ):
        return False
    bound_discovery_run_ids = sorted({candidate_key[0] for candidate_key in candidate_keys})
    recomputed = recompute_discovery_calibration_claims(
        payload.get("source_outcomes"),
        source_candidates,
    )
    return (
        bound_discovery_run_ids == normalized_discovery_run_ids
        and recomputed is not None
        and all(payload.get(field) == value for field, value in recomputed.items())
        and (
            (payload.get("target_run_id") is None and expected_run_id.startswith("calibration_"))
            or payload.get("target_run_id") == expected_run_id
        )
    )


def _outcome_source_bindings_are_authorizable(payload: Mapping[str, Any]) -> bool:
    from app.outcomes.lineage import serialized_outcome_binding_is_decision_eligible

    source_run_ids = payload.get("source_run_ids")
    if not isinstance(source_run_ids, list):
        return False
    normalized_source_run_ids = [_normalized_text(run_id) for run_id in source_run_ids]
    if (
        any(run_id is None for run_id in normalized_source_run_ids)
        or len(set(normalized_source_run_ids)) != len(normalized_source_run_ids)
        or normalized_source_run_ids != sorted(normalized_source_run_ids)
    ):
        return False
    source_outcomes = payload.get("source_outcomes")
    if not isinstance(source_outcomes, list):
        return False
    source_keys: list[tuple[str, str, int, str]] = []
    for binding in source_outcomes:
        if not isinstance(binding, Mapping) or not serialized_outcome_binding_is_decision_eligible(
            binding
        ):
            return False
        state = binding.get("outcome_state")
        if not isinstance(state, Mapping):
            return False
        run_id = _normalized_text(state.get("run_id"))
        ticker = _normalized_text(state.get("ticker"))
        fingerprint = _normalized_text(binding.get("financial_integrity_fingerprint"))
        outcome_id = state.get("id")
        if (
            run_id is None
            or ticker is None
            or fingerprint is None
            or not isinstance(outcome_id, int)
        ):
            return False
        source_keys.append((run_id, ticker.upper(), outcome_id, fingerprint))
    if source_keys != sorted(source_keys) or len(set(source_keys)) != len(source_keys):
        return False
    bound_run_ids = sorted({key[0] for key in source_keys})
    return bound_run_ids == normalized_source_run_ids


def _typed_payload_is_authorizable(
    payload: Mapping[str, Any],
    *,
    artifact_type: str,
    expected_run_id: str,
) -> bool:
    if artifact_type == "grade_status_calibration":
        return _grade_status_calibration_payload_is_authorizable(
            payload,
            expected_run_id=expected_run_id,
        )
    if artifact_type == "discovery_calibration":
        return _discovery_calibration_payload_is_authorizable(
            payload,
            expected_run_id=expected_run_id,
        )
    return False


def _typed_report_bytes_are_authorizable(
    payload: Mapping[str, Any],
    report_bytes: bytes,
    *,
    artifact_type: str,
) -> bool:
    try:
        if artifact_type == "grade_status_calibration":
            from app.calibration.calibration_report import _to_markdown

            expected = _to_markdown(payload)
        elif artifact_type == "discovery_calibration":
            from app.discovery.calibration import _to_markdown

            expected = _to_markdown(dict(payload))
        else:
            return False
    except (KeyError, TypeError, ValueError):
        return False
    return report_bytes == expected.encode("utf-8")


def _validated_typed_authorization_records(
    authorization_path: Path,
    *,
    manifest: Mapping[str, Any],
    expected_artifact_path: Path | None = None,
) -> dict[str, Mapping[str, Any]] | None:
    """Validate an exact typed post-write authorization against current bytes."""

    del manifest  # A usable manifest binds this process to canonical runtime roots.
    resolved_authorization = authorization_path.expanduser().resolve()
    try:
        payload = json.loads(resolved_authorization.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    artifact_type = _normalized_text(payload.get("artifact_type"))
    run_id = _normalized_text(payload.get("run_id"))
    root = _typed_artifact_root(str(artifact_type or ""))
    if (
        payload.get("schema_version") != TYPED_AUTHORIZATION_SCHEMA_VERSION
        or payload.get("complete") is not True
        or not isinstance(payload.get("generated_at"), str)
        or artifact_type is None
        or run_id is None
        or root is None
        or resolved_authorization.parent != root
        or resolved_authorization.name != f"{run_id}{TYPED_AUTHORIZATION_SUFFIX}"
    ):
        return None
    raw_records = payload.get("artifacts")
    if not isinstance(raw_records, list) or len(raw_records) != 2:
        return None
    expected_names = {
        f"{run_id}.json": "artifact_json",
        f"{run_id}.md": "report_markdown",
    }
    validated: dict[str, Mapping[str, Any]] = {}
    artifact_payload: Mapping[str, Any] | None = None
    report_bytes: bytes | None = None
    for record in raw_records:
        if not isinstance(record, dict):
            return None
        raw_path = _normalized_text(record.get("path"))
        sha256 = _normalized_text(record.get("sha256"))
        kind = _normalized_text(record.get("kind"))
        if raw_path is None or sha256 is None or kind is None:
            return None
        path = Path(raw_path).expanduser().resolve()
        if (
            raw_path != str(path)
            or path.parent != root
            or expected_names.get(path.name) != kind
            or kind in validated
            or re.fullmatch(r"[0-9a-f]{64}", sha256) is None
            or not path.is_file()
        ):
            return None
        try:
            data = path.read_bytes()
        except OSError:
            return None
        if _sha256_bytes(data) != sha256:
            return None
        if kind == "artifact_json":
            try:
                parsed = json.loads(data.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return None
            if not isinstance(parsed, dict):
                return None
            artifact_payload = parsed
        else:
            if not data.strip():
                return None
            report_bytes = data
        validated[kind] = record
    if (
        set(validated) != set(expected_names.values())
        or artifact_payload is None
        or report_bytes is None
        or (
            expected_artifact_path is not None
            and str(expected_artifact_path.expanduser().resolve())
            not in {str(record["path"]) for record in validated.values()}
        )
        or not _typed_payload_is_authorizable(
            artifact_payload,
            artifact_type=artifact_type,
            expected_run_id=run_id,
        )
        or not _typed_report_bytes_are_authorizable(
            artifact_payload,
            report_bytes,
            artifact_type=artifact_type,
        )
    ):
        return None
    return validated


def write_typed_financial_authorization(
    artifact_path: str | Path,
    report_path: str | Path,
    *,
    artifact_type: str,
    generated_at: datetime | None = None,
) -> Path:
    """Atomically authorize an exact supported post-write JSON/Markdown pair."""

    artifact = Path(artifact_path).expanduser().resolve()
    report = Path(report_path).expanduser().resolve()
    root = _typed_artifact_root(artifact_type)
    if (
        root is None
        or artifact.parent != root
        or report.parent != root
        or artifact.suffix.lower() != ".json"
        or report.suffix.lower() != ".md"
        or artifact.stem != report.stem
    ):
        raise ValueError("Typed authorization requires a supported canonical artifact pair.")
    selected_manifest = active_financial_integrity_manifest_path()
    manifest_snapshot = _load_manifest_snapshot(selected_manifest)
    if manifest_snapshot is None:
        raise RuntimeError(
            "Cannot authorize typed output without a usable canonical audit manifest."
        )
    manifest = manifest_snapshot.payload
    try:
        artifact_bytes = artifact.read_bytes()
        report_bytes = report.read_bytes()
        artifact_payload = json.loads(artifact_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot authorize unreadable typed output: {exc}") from exc
    if not isinstance(artifact_payload, dict):
        raise RuntimeError("Typed artifact JSON root must be an object.")
    run_id = _normalized_text(artifact_payload.get("run_id"))
    if (
        run_id is None
        or artifact.name != f"{run_id}.json"
        or report.name != f"{run_id}.md"
        or not report_bytes.strip()
        or not _typed_payload_is_authorizable(
            artifact_payload,
            artifact_type=artifact_type,
            expected_run_id=run_id,
        )
        or not _typed_report_bytes_are_authorizable(
            artifact_payload,
            report_bytes,
            artifact_type=artifact_type,
        )
    ):
        raise RuntimeError("Typed output paths/content/source authorization are inconsistent.")
    now = generated_at or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    authorization = {
        "schema_version": TYPED_AUTHORIZATION_SCHEMA_VERSION,
        "generated_at": now.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "complete": True,
        "artifact_type": artifact_type,
        "run_id": run_id,
        "artifacts": [
            {
                "kind": "artifact_json",
                "path": str(artifact),
                "sha256": _sha256_bytes(artifact_bytes),
            },
            {
                "kind": "report_markdown",
                "path": str(report),
                "sha256": _sha256_bytes(report_bytes),
            },
        ],
    }
    destination = _typed_authorization_path(artifact)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(authorization, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    if _validated_typed_authorization_records(
        destination,
        manifest=manifest,
        expected_artifact_path=artifact,
    ) is None or not _manifest_snapshot_is_current(manifest_snapshot):
        destination.unlink(missing_ok=True)
        raise RuntimeError("Typed post-write authorization failed exact validation.")
    return destination


def authorized_artifact_bytes(
    path: str | Path,
    manifest_path: str | Path | None = None,
) -> tuple[str, bytes | None]:
    """Authorize and return the exact bytes read from one opened artifact.

    Callers that will render or parse a decision artifact should consume these
    returned bytes instead of authorizing a pathname and reopening it later.
    """

    artifact_path = _exact_lexical_path(path)
    if artifact_path is None:
        return UNAUDITED, None
    selected_manifest = active_financial_integrity_manifest_path(manifest_path)
    manifest_snapshot = _load_manifest_snapshot(selected_manifest)
    if manifest_snapshot is None:
        return UNAUDITED, None
    manifest = manifest_snapshot.payload
    by_path = _manifest_artifact_indexes(
        str(manifest_snapshot.path),
        manifest_snapshot.sha256,
        manifest_snapshot.canonical_root_key,
    )
    record = by_path.get(str(artifact_path))
    try:
        with artifact_path.open("rb") as handle:
            data = handle.read()
    except OSError:
        if not _manifest_snapshot_is_current(manifest_snapshot, manifest_path):
            return UNAUDITED, None
        return (STALE_AUDIT, None) if isinstance(record, dict) else (UNAUDITED, None)
    if _exact_lexical_path(path) != artifact_path:
        return UNAUDITED, None
    if not _manifest_snapshot_is_current(manifest_snapshot, manifest_path):
        return UNAUDITED, None
    current_hash = _sha256_bytes(data)
    if isinstance(record, dict):
        if record.get("sha256") != current_hash:
            return STALE_AUDIT, None
        if record.get("integrity_status") == INVALID:
            return INVALID, None
        dependency_status = _authorized_artifact_dependency_status(
            artifact_path,
            data,
            record=record,
            artifact_records=by_path,
            manifest_snapshot=manifest_snapshot,
            manifest_path=manifest_path,
        )
        if dependency_status != PASS:
            return dependency_status, None
        if not _manifest_snapshot_is_current(manifest_snapshot, manifest_path):
            return UNAUDITED, None
        return PASS, data

    postwrite_records = _validated_run_authorization_records(
        artifact_path.parent / RUN_AUTHORIZATION_FILENAME,
        manifest=manifest,
        expected_run_id=artifact_path.parent.name,
    )
    if not _manifest_snapshot_is_current(manifest_snapshot, manifest_path):
        return UNAUDITED, None
    if postwrite_records is not None:
        for postwrite_record in postwrite_records.values():
            if (
                postwrite_record.get("path") == str(artifact_path)
                and postwrite_record.get("sha256") == current_hash
            ):
                if not _manifest_snapshot_is_current(
                    manifest_snapshot,
                    manifest_path,
                ):
                    return UNAUDITED, None
                return PASS, data
    typed_records = _validated_typed_authorization_records(
        _typed_authorization_path(artifact_path),
        manifest=manifest,
        expected_artifact_path=artifact_path,
    )
    if not _manifest_snapshot_is_current(manifest_snapshot, manifest_path):
        return UNAUDITED, None
    if typed_records is not None:
        for typed_record in typed_records.values():
            if (
                typed_record.get("path") == str(artifact_path)
                and typed_record.get("sha256") == current_hash
            ):
                if not _manifest_snapshot_is_current(
                    manifest_snapshot,
                    manifest_path,
                ):
                    return UNAUDITED, None
                return PASS, data
    return UNAUDITED, None


def _authorized_artifact_dependency_status(
    artifact_path: Path,
    artifact_bytes: bytes,
    *,
    record: Mapping[str, Any],
    artifact_records: Mapping[str, Mapping[str, Any]],
    manifest_snapshot: _ManifestSnapshot,
    manifest_path: str | Path | None,
) -> str:
    """Revalidate exact mutable dependencies of an audited rendered artifact.

    A manifest record proves only that the rendered bytes were valid at audit
    time. Reports and digests remain publishable only while their canonical
    sources still match the exact bytes and decision state audited with them.
    """

    if (
        record.get("family") == "autonomous_sector"
        and artifact_path.name == "autonomous_sector_report.md"
    ):
        source_path = artifact_path.with_name("autonomous_sector_run.json")
        source_record = artifact_records.get(str(source_path))
        if not isinstance(source_record, Mapping):
            return STALE_AUDIT
        if (
            source_record.get("family") != "autonomous_sector"
            or source_record.get("integrity_status") != PASS
            or source_record.get("decision_eligible") is not True
        ):
            return INVALID
        if _exact_lexical_path(source_path) != source_path:
            return STALE_AUDIT
        if not _manifest_snapshot_is_current(manifest_snapshot, manifest_path):
            return UNAUDITED
        try:
            source_bytes = source_path.read_bytes()
        except OSError:
            return STALE_AUDIT
        if _sha256_bytes(source_bytes) != source_record.get("sha256"):
            return STALE_AUDIT
        try:
            source_payload = json.loads(source_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return INVALID
        if not isinstance(source_payload, Mapping):
            return INVALID
        if (
            _normalized_text(source_payload.get("run_id"))
            != _normalized_text(source_record.get("run_id"))
            or _canonical_run_report_bytes(source_payload) != artifact_bytes
        ):
            return INVALID
        try:
            if _current_sha256(source_path) != source_record.get("sha256"):
                return STALE_AUDIT
        except OSError:
            return STALE_AUDIT
        if _exact_lexical_path(source_path) != source_path:
            return STALE_AUDIT
        if not _manifest_snapshot_is_current(manifest_snapshot, manifest_path):
            return UNAUDITED
        return PASS

    if record.get("family") != "watchlist_report" or artifact_path.suffix.lower() != ".md":
        return PASS

    digest_path = artifact_path
    digest_bytes = artifact_bytes
    try:
        markdown = digest_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return INVALID
    if _is_canonical_blocked_digest(markdown):
        return PASS
    decision_tickers = _digest_decision_tickers(markdown)
    lineage_path = digest_path.with_suffix(".lineage.json").resolve()
    lineage_record = artifact_records.get(str(lineage_path))
    has_rendered_state = _DIGEST_RENDERED_STATE_MARKER in markdown
    if not has_rendered_state:
        return INVALID
    if not isinstance(lineage_record, Mapping):
        return STALE_AUDIT
    if (
        lineage_record.get("family") != "watchlist_report"
        or lineage_record.get("integrity_status") != PASS
        or lineage_record.get("decision_eligible") is not True
    ):
        return INVALID

    try:
        lineage_bytes = lineage_path.read_bytes()
    except OSError:
        return STALE_AUDIT
    if _sha256_bytes(lineage_bytes) != lineage_record.get("sha256"):
        return STALE_AUDIT
    try:
        lineage_payload = json.loads(lineage_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return INVALID
    if not isinstance(lineage_payload, Mapping):
        return INVALID
    if (
        lineage_payload.get("schema_version") != DIGEST_LINEAGE_SCHEMA_VERSION
        or lineage_payload.get("digest_path") != str(digest_path)
        or lineage_payload.get("digest_sha256") != _sha256_bytes(digest_bytes)
    ):
        return INVALID
    bound_manifest_text = _normalized_text(lineage_payload.get("manifest_path"))
    bound_manifest_sha256 = _normalized_text(lineage_payload.get("manifest_sha256"))
    if (
        bound_manifest_text is None
        or bound_manifest_sha256 is None
        or re.fullmatch(r"[0-9a-f]{64}", bound_manifest_sha256) is None
    ):
        return INVALID
    try:
        bound_manifest_path = Path(bound_manifest_text).expanduser().resolve()
        if (
            bound_manifest_text != str(bound_manifest_path)
            or _current_sha256(bound_manifest_path) != bound_manifest_sha256
            or _load_manifest_snapshot(bound_manifest_path) is None
        ):
            return STALE_AUDIT
    except (OSError, RuntimeError):
        return STALE_AUDIT

    rows = lineage_payload.get("rows")
    if not isinstance(rows, list):
        return INVALID
    row_tickers: set[str] = set()
    source_paths: set[Path] = set()
    from app.watchlist.lineage import authorized_watchlist_decision_binding

    for row in rows:
        if not isinstance(row, Mapping):
            return INVALID
        ticker = _normalized_text(row.get("ticker"))
        run_id = _normalized_text(row.get("source_run_id"))
        source_path_text = _normalized_text(row.get("source_artifact_path"))
        source_sha256 = _normalized_text(row.get("source_artifact_sha256"))
        source_decision_fingerprint = _normalized_text(row.get("source_decision_fingerprint"))
        decision_state = row.get("decision_state")
        if (
            ticker is None
            or run_id is None
            or source_path_text is None
            or source_sha256 is None
            or source_decision_fingerprint is None
            or not isinstance(decision_state, Mapping)
        ):
            return INVALID
        try:
            source_path = Path(source_path_text).expanduser().resolve()
        except (OSError, RuntimeError):
            return INVALID
        if source_path_text != str(source_path):
            return INVALID
        binding = authorized_watchlist_decision_binding(
            decision_state,
            manifest_snapshot.path,
            ticker=ticker,
        )
        if binding is None:
            return STALE_AUDIT
        if (
            binding.get("source_run_id") != run_id
            or binding.get("source_artifact_path") != str(source_path)
            or binding.get("source_artifact_sha256") != source_sha256
            or binding.get("source_decision_fingerprint") != source_decision_fingerprint
        ):
            return INVALID
        row_tickers.add(ticker.upper())
        source_paths.add(source_path)

    if row_tickers != decision_tickers:
        return INVALID
    try:
        if _current_sha256(lineage_path) != lineage_record.get("sha256"):
            return STALE_AUDIT
        if _current_sha256(bound_manifest_path) != bound_manifest_sha256:
            return STALE_AUDIT
        for source_path in source_paths:
            source_record = artifact_records.get(str(source_path))
            if (
                not isinstance(source_record, Mapping)
                or source_record.get("integrity_status") != PASS
                or _current_sha256(source_path) != source_record.get("sha256")
            ):
                return STALE_AUDIT
    except OSError:
        return STALE_AUDIT
    if not _manifest_snapshot_is_current(manifest_snapshot, manifest_path):
        return UNAUDITED
    return PASS


def artifact_decision_eligibility(
    path_or_payload: str | Path | Mapping[str, Any] | Any,
    manifest_path: str | Path | None = None,
) -> str:
    """Return PASS, INVALID, UNAUDITED, or STALE_AUDIT.

    In-memory payloads are checked directly and are suitable for pre-write
    publication gates.  Filesystem artifacts require a completed audit
    manifest; their current SHA-256 must match the recorded hash.
    """

    payload = _coerce_payload(path_or_payload)
    if payload is not None:
        try:
            _payload_sha256(payload)
        except (TypeError, ValueError):
            return INVALID
        return INVALID if audit_payload(payload) else PASS

    status, _data = authorized_artifact_bytes(path_or_payload, manifest_path)
    return status


def invalid_run_ids_from_active_manifest(
    manifest_path: str | Path | None = None,
) -> set[str]:
    """Return invalid run IDs from the active, completed audit manifest.

    Consumers must apply this set *after* their normal latest-row selection.
    Filtering candidates before MAX(id) selection would resurrect an older
    decision that the invalid newer row had already superseded.
    """

    selected = active_financial_integrity_manifest_path(manifest_path)
    if selected is None or not selected.is_file():
        return set()
    snapshot = _load_manifest_snapshot(selected)
    if snapshot is None:
        return set()
    invalid_ids = set(
        _invalid_run_ids_cached(
            str(snapshot.path),
            snapshot.sha256,
            snapshot.canonical_root_key,
        )
    )
    return invalid_ids if _manifest_snapshot_is_current(snapshot, manifest_path) else set()


@lru_cache(maxsize=8)
def _invalid_run_ids_cached(
    manifest_path: str,
    manifest_sha256: str,
    canonical_root_key: tuple[tuple[str, str], ...],
) -> frozenset[str]:
    manifest = _read_manifest_cached(
        manifest_path,
        manifest_sha256,
        canonical_root_key,
    )
    if manifest is None:
        return frozenset()
    declared = manifest.get("invalid_run_ids")
    invalid_ids: set[str] = (
        {str(value).strip() for value in declared if str(value).strip()}
        if isinstance(declared, list)
        else set()
    )
    artifacts = manifest.get("artifacts")
    if isinstance(artifacts, list):
        for item in artifacts:
            if not isinstance(item, dict) or item.get("integrity_status") != INVALID:
                continue
            run_id = _normalized_text(item.get("run_id"))
            if run_id:
                invalid_ids.add(run_id)
    violations = manifest.get("violations")
    if isinstance(violations, list):
        for item in violations:
            if not isinstance(item, dict):
                continue
            run_id = _normalized_text(item.get("run_id"))
            if run_id:
                invalid_ids.add(run_id)
    return frozenset(invalid_ids)


@lru_cache(maxsize=8)
def _manifest_run_artifact_index(
    manifest_path: str,
    manifest_sha256: str,
    canonical_root_key: tuple[tuple[str, str], ...],
) -> dict[str, tuple[Mapping[str, Any], ...]]:
    manifest = _read_manifest_cached(
        manifest_path,
        manifest_sha256,
        canonical_root_key,
    )
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    artifacts = manifest.get("artifacts") if manifest is not None else None
    if isinstance(artifacts, list):
        for record in artifacts:
            if not isinstance(record, dict):
                continue
            run_id = _normalized_text(record.get("run_id"))
            if run_id is not None:
                grouped.setdefault(run_id, []).append(record)
    return {run_id: tuple(records) for run_id, records in grouped.items()}


def run_id_is_decision_eligible(
    run_id: str | None,
    manifest_path: str | Path | None = None,
) -> bool:
    """Return whether a run is positively authorized by current audited bytes.

    Absence from the invalid-run list is not authorization.  A current product
    row may use a source run only when the canonical manifest contains at least
    one record for that exact run ID, every such record was audited ``PASS``,
    and every recorded file still matches its audited SHA-256.  This makes an
    unlisted, deleted, or post-audit-mutated run fail closed.
    """

    normalized_run_id = _normalized_text(run_id)
    selected = active_financial_integrity_manifest_path(manifest_path)
    snapshot = _load_manifest_snapshot(selected)
    if normalized_run_id is None or snapshot is None:
        return False
    manifest = snapshot.payload
    if normalized_run_id in {
        str(value).strip() for value in manifest.get("invalid_run_ids", []) if str(value).strip()
    }:
        return False
    records = _manifest_run_artifact_index(
        str(snapshot.path),
        snapshot.sha256,
        snapshot.canonical_root_key,
    ).get(normalized_run_id, ())
    if not records:
        postwrite_records = _postwrite_records_for_run(
            normalized_run_id,
            manifest=manifest,
        )
        return postwrite_records is not None and _manifest_snapshot_is_current(
            snapshot,
            manifest_path,
        )
    if (
        _single_canonical_primary_record(
            records,
            run_id=normalized_run_id,
            manifest=manifest,
        )
        is None
    ):
        return False
    for record in records:
        if record.get("integrity_status") != PASS or record.get("decision_eligible") is not True:
            return False
        raw_path = _normalized_text(record.get("path"))
        if raw_path is None:
            return False
        artifact_path = Path(raw_path).expanduser()
        try:
            if not artifact_path.is_file() or _current_sha256(artifact_path) != record.get(
                "sha256"
            ):
                return False
        except OSError:
            return False
    return _manifest_snapshot_is_current(snapshot, manifest_path)


def authorized_run_artifact_binding(
    run_id: str | None,
    manifest_path: str | Path | None = None,
) -> dict[str, str] | None:
    """Return the exact authorized primary artifact binding for ``run_id``."""

    normalized_run_id = _normalized_text(run_id)
    selected = active_financial_integrity_manifest_path(manifest_path)
    snapshot = _load_manifest_snapshot(selected)
    if normalized_run_id is None or snapshot is None:
        return None
    manifest = snapshot.payload
    if not run_id_is_decision_eligible(normalized_run_id, selected):
        return None
    records = _manifest_run_artifact_index(
        str(snapshot.path),
        snapshot.sha256,
        snapshot.canonical_root_key,
    ).get(normalized_run_id, ())
    candidate_records: Sequence[Mapping[str, Any]]
    if records:
        candidate_records = records
    else:
        postwrite = _postwrite_records_for_run(normalized_run_id, manifest=manifest)
        if postwrite is None:
            return None
        candidate_records = (
            {
                **postwrite["artifact_json"],
                "family": "autonomous_sector",
            },
        )
    primary = _single_canonical_primary_record(
        candidate_records,
        run_id=normalized_run_id,
        manifest=manifest,
    )
    if not isinstance(primary, Mapping):
        return None
    path = Path(str(primary.get("path") or "")).expanduser().resolve()
    sha256 = _normalized_text(primary.get("sha256"))
    if sha256 is None or not path.is_file() or _current_sha256(path) != sha256:
        return None
    if not _manifest_snapshot_is_current(snapshot, manifest_path):
        return None
    return {
        "source_run_id": normalized_run_id,
        "source_artifact_path": str(path),
        "source_artifact_sha256": sha256,
    }


def authorized_run_ticker_artifact_binding(
    run_id: str | None,
    ticker: str | None,
    manifest_path: str | Path | None = None,
) -> dict[str, str] | None:
    """Return a run binding only when its exact bytes contain ``ticker``.

    Run-level authorization alone cannot prove that a mutable database row was
    sourced from that run.  Current watchlist, outcome, digest, and calibration
    rows therefore need both the exact run binding and membership of their
    ticker in the authorized primary artifact.
    """

    normalized_ticker = _normalized_text(ticker)
    binding = authorized_run_artifact_binding(run_id, manifest_path)
    if normalized_ticker is None or binding is None:
        return None
    status, artifact_bytes = authorized_artifact_bytes(
        binding["source_artifact_path"],
        manifest_path,
    )
    if (
        status != PASS
        or artifact_bytes is None
        or _sha256_bytes(artifact_bytes) != binding["source_artifact_sha256"]
    ):
        return None
    try:
        payload = json.loads(artifact_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    payload_run_id = _normalized_text(payload.get("run_id"))
    if payload_run_id != binding["source_run_id"]:
        return None
    if normalized_ticker.upper() not in _collect_tickers(payload):
        return None
    return binding


def run_ticker_is_decision_eligible(
    run_id: str | None,
    ticker: str | None,
    manifest_path: str | Path | None = None,
) -> bool:
    """Whether exact authorized run bytes positively contain ``ticker``."""

    return authorized_run_ticker_artifact_binding(run_id, ticker, manifest_path) is not None


def _discover_files(roots: Sequence[tuple[str, Path]]) -> list[tuple[str, Path]]:
    found: dict[Path, tuple[str, Path]] = {}
    found_identities: dict[tuple[int, int], tuple[str, Path]] = {}
    for family, root in roots:
        root = root.expanduser().resolve()
        if not root.exists():
            continue
        paths = [root] if root.is_file() else root.rglob("*")
        for path in paths:
            if not path.is_file() or path.suffix.lower() not in {".json", ".md"}:
                continue
            try:
                lexical_path = Path(os.path.abspath(os.fspath(path.expanduser())))
                resolved_path = path.resolve()
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Financial-integrity audit could not normalize artifact path {path}: {exc}"
                ) from exc
            previous = found.get(resolved_path)
            if previous is not None:
                previous_family, previous_path = previous
                raise ValueError(
                    "Financial-integrity audit rejects duplicate artifact aliases: "
                    f"{previous_family}:{previous_path} and {family}:{lexical_path} "
                    f"resolve to {resolved_path}."
                )
            if lexical_path != resolved_path:
                raise ValueError(
                    "Financial-integrity audit rejects artifact path aliases: "
                    f"{family}:{lexical_path} resolves to {resolved_path}."
                )
            try:
                file_stat = lexical_path.stat()
            except OSError as exc:
                raise ValueError(
                    f"Financial-integrity audit could not identify artifact path {path}: {exc}"
                ) from exc
            identity = (file_stat.st_dev, file_stat.st_ino)
            previous_identity = found_identities.get(identity)
            if previous_identity is not None:
                previous_family, previous_path = previous_identity
                raise ValueError(
                    "Financial-integrity audit rejects duplicate artifact file identities: "
                    f"{previous_family}:{previous_path} and {family}:{lexical_path} "
                    "refer to the same underlying file."
                )
            found[resolved_path] = (family, lexical_path)
            found_identities[identity] = (family, lexical_path)
    return sorted(
        ((family, path) for _resolved, (family, path) in found.items()),
        key=lambda item: str(item[1]),
    )


def _collect_tickers(value: Any) -> set[str]:
    tickers: set[str] = set()
    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            for key, item in current.items():
                if key in {"ticker", "primary_ticker", "selected_ticker"} and isinstance(item, str):
                    normalized = item.strip().upper()
                    if normalized:
                        tickers.add(normalized)
                elif key in {
                    "tickers",
                    "selected_tickers",
                    "loaded_tickers",
                    "survivors",
                } and isinstance(item, list):
                    tickers.update(
                        token.strip().upper()
                        for token in item
                        if isinstance(token, str) and token.strip()
                    )
                stack.append(item)
        elif isinstance(current, list):
            stack.extend(current)
    return tickers


def _record_date(path: Path, payload: Mapping[str, Any] | None) -> str | None:
    if payload is not None:
        for key in ("as_of_date", "run_date", "date"):
            value = _normalized_text(payload.get(key))
            if value and _DATE_RE.match(value[:10]):
                return value[:10]
    match = _DATE_RE.search(path.name)
    return match.group("date") if match else None


def _analyst_info(
    payload: Mapping[str, Any],
) -> tuple[str, str, float | None, tuple[str, ...]] | None:
    ticker = str(payload.get("ticker") or "").strip().upper()
    as_of_date = str(payload.get("as_of_date") or "").strip()
    valuation = payload.get("valuation")
    if not ticker or not as_of_date or not isinstance(valuation, dict):
        return None
    raw_price = valuation.get("price")
    price = _as_float(raw_price)
    if price is None:
        raw_price = valuation.get("current_price")
        price = _as_float(raw_price)
    decision_fields = (
        "base_case_value",
        "bear_case_value",
        "bull_case_value",
        "margin_of_safety",
        "market_cap",
        "dcf_base",
        "epv_adjusted",
        "graham_value",
    )
    if raw_price is None and not any(valuation.get(field) is not None for field in decision_fields):
        return None
    required_text_fields = (
        "price_source",
        "price_as_of_date",
        "price_currency",
        "price_unit",
        "price_basis",
        "quote_snapshot_id",
    )
    missing = [
        field for field in required_text_fields if _normalized_text(valuation.get(field)) is None
    ]
    if price is None or price <= 0:
        missing.append("price_finite_positive")
    snapshot_id = _normalized_text(valuation.get("quote_snapshot_id"))
    if snapshot_id is not None and re.fullmatch(r"[0-9a-f]{64}", snapshot_id) is None:
        missing.append("quote_snapshot_id_valid_sha256")
    price_as_of_date = _normalized_text(valuation.get("price_as_of_date"))
    if price_as_of_date is not None:
        try:
            if date.fromisoformat(price_as_of_date[:10]) > date.fromisoformat(as_of_date[:10]):
                missing.append("price_as_of_date_not_future")
        except ValueError:
            missing.append("price_as_of_date_valid_date")
    price_basis = str(valuation.get("price_basis") or "").strip().upper()
    if price_basis and price_basis not in {"UNADJUSTED", "SPLIT_ADJUSTED"}:
        missing.append("price_basis_supported")
    if _normalized_text(valuation.get("price_currency")) not in {None, "USD"}:
        missing.append("price_currency_usd")
    if _normalized_text(valuation.get("price_unit")) not in {None, "USD_per_share"}:
        missing.append("price_unit_usd_per_share")
    split_factor = _as_float(valuation.get("split_adjustment_factor"))
    if split_factor is None or split_factor <= 0:
        missing.append("split_adjustment_factor")
    split_effective_date = _normalized_text(valuation.get("split_effective_date"))
    if "split_effective_date" not in valuation:
        missing.append("split_effective_date")
    if price_basis == "SPLIT_ADJUSTED":
        raw_unadjusted_price = _as_float(valuation.get("raw_price"))
        if raw_unadjusted_price is None or raw_unadjusted_price <= 0:
            missing.append("raw_price")
        if split_effective_date is None:
            missing.append("split_effective_date_required_for_adjusted_basis")
        elif not _date_is_valid_at_or_before(split_effective_date, as_of_date):
            missing.append("split_effective_date_valid_nonfuture")
        if (
            price is not None
            and price > 0
            and raw_unadjusted_price is not None
            and split_factor is not None
            and split_factor > 0
            and not _close(raw_unadjusted_price / price, split_factor)
        ):
            missing.append("split_adjustment_reconciles")
    elif price_basis == "UNADJUSTED":
        raw_unadjusted_price = _as_float(valuation.get("raw_price"))
        if split_factor is not None and not _close(split_factor, 1.0):
            missing.append("unadjusted_split_factor_is_one")
        if (
            price is not None
            and raw_unadjusted_price is not None
            and not _close(price, raw_unadjusted_price)
        ):
            missing.append("unadjusted_raw_price_reconciles")
        if split_effective_date is not None and not _date_is_valid_at_or_before(
            split_effective_date, as_of_date
        ):
            missing.append("split_effective_date_valid_nonfuture")
    if (
        price is not None
        and price > 0
        and snapshot_id is not None
        and not any(
            field in missing
            for field in (
                "price_source",
                "price_as_of_date",
                "price_currency",
                "price_unit",
                "price_basis",
                "quote_snapshot_id",
                "quote_snapshot_id_valid_sha256",
                "split_adjustment_factor",
                "split_effective_date",
                "price_basis_supported",
                "price_currency_usd",
                "price_unit_usd_per_share",
                "price_as_of_date_not_future",
                "price_as_of_date_valid_date",
                "raw_price",
                "split_effective_date_required_for_adjusted_basis",
                "split_effective_date_valid_nonfuture",
                "split_adjustment_reconciles",
                "unadjusted_split_factor_is_one",
                "unadjusted_raw_price_reconciles",
            )
        )
    ):
        from app.autonomous.financial_integrity import stable_quote_hash

        expected_snapshot_id = stable_quote_hash(
            ticker=ticker,
            price=price,
            as_of_date=price_as_of_date,
            currency=valuation.get("price_currency"),
            source=valuation.get("price_source"),
            source_url=valuation.get("price_source_url"),
            price_basis=price_basis,
            raw_price=valuation.get("raw_price"),
            split_adjustment_factor=split_factor,
            split_effective_date=valuation.get("split_effective_date"),
        )
        if snapshot_id != expected_snapshot_id:
            missing.append("quote_snapshot_id_reconciles")
    return ticker, as_of_date, price, tuple(missing)


def _analyst_payload_violations(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    info = _analyst_info(payload)
    if info is None:
        return []
    ticker, as_of_date, price, missing = info
    valuation = payload.get("valuation")
    assert isinstance(valuation, Mapping)
    findings: list[dict[str, Any]] = []
    if missing:
        findings.append(
            _violation(
                invariant="ANALYST_QUOTE_PROVENANCE_MISSING",
                ticker=ticker,
                run_id=_normalized_text(payload.get("run_id")),
                as_of_date=as_of_date,
                field="valuation.price",
                source_values={
                    "analyst_price": price,
                    "missing_fields": list(missing),
                    "price_source": valuation.get("price_source"),
                    "price_as_of_date": valuation.get("price_as_of_date"),
                    "price_currency": valuation.get("price_currency"),
                    "quote_snapshot_id": valuation.get("quote_snapshot_id"),
                    "price_basis": valuation.get("price_basis"),
                    "split_adjustment_factor": valuation.get("split_adjustment_factor"),
                    "split_effective_date": valuation.get("split_effective_date"),
                },
                expected_relationship=(
                    "a decision price carries source, date, currency, unit, stable snapshot "
                    "identity, and explicit split-adjustment lineage"
                ),
                observed_relationship=f"missing_or_invalid={','.join(missing)}",
                reason=(
                    "The standalone analyst artifact contains an investment price whose "
                    "historical quote and split identity cannot be proven."
                ),
                llm_consumed=True,
                repair_classification=MISSING_PROVENANCE_UNREPAIRABLE,
            )
        )
    nonfinite_paths = _nonfinite_financial_value_paths(valuation, prefix="valuation")
    if nonfinite_paths:
        findings.append(
            _violation(
                invariant="NONFINITE_FINANCIAL_VALUE",
                ticker=ticker,
                run_id=_normalized_text(payload.get("run_id")),
                as_of_date=as_of_date,
                field="analyst_valuation",
                source_values={"nonfinite_paths": nonfinite_paths},
                expected_relationship="all investment-relevant analyst values are finite",
                observed_relationship=f"nonfinite_paths={','.join(nonfinite_paths)}",
                reason="The analyst artifact contains NaN or infinite financial values.",
                llm_consumed=True,
                repair_classification=REQUIRES_LLM_REREVIEW,
            )
        )
    return findings


def _date_is_valid_at_or_before(value: Any, upper_bound: Any) -> bool:
    value_text = _normalized_text(value)
    upper_text = _normalized_text(upper_bound)
    if value_text is None or upper_text is None:
        return False
    try:
        return date.fromisoformat(value_text[:10]) <= date.fromisoformat(upper_text[:10])
    except ValueError:
        return False


def _scan_financial_value_paths(value: Any, *, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if (
                str(key).lower() in _SCAN_FINANCIAL_KEYS
                and isinstance(item, (int, float))
                and not isinstance(item, bool)
            ):
                found.append(path)
            found.extend(_scan_financial_value_paths(item, prefix=path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_scan_financial_value_paths(item, prefix=f"{prefix}[{index}]"))
    elif isinstance(value, str) and _SCAN_FINANCIAL_TEXT_RE.search(value):
        found.append(prefix or "text")
    return found


def _nonfinite_financial_value_paths(value: Any, *, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if (
                str(key).lower() in _SCAN_FINANCIAL_KEYS
                and isinstance(item, (int, float))
                and not isinstance(item, bool)
                and not math.isfinite(float(item))
            ):
                found.append(path)
            found.extend(_nonfinite_financial_value_paths(item, prefix=path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_nonfinite_financial_value_paths(item, prefix=f"{prefix}[{index}]"))
    return found


def _decision_context(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "financial_integrity"}


def _structured_quote_prices(value: Any) -> dict[str, list[float]]:
    prices: dict[str, list[float]] = {}

    def visit(current: Any, inherited_ticker: str | None = None) -> None:
        if isinstance(current, Mapping):
            ticker = _normalized_text(current.get("ticker")) or inherited_ticker
            normalized_ticker = ticker.upper() if ticker is not None else None
            if normalized_ticker is not None:
                for key in ("current_price", "price", "scorecard_price"):
                    if key not in current:
                        continue
                    price = _as_float(current.get(key))
                    if price is not None and price > 0:
                        prices.setdefault(normalized_ticker, []).append(price)
            for key, item in current.items():
                if key != "financial_integrity":
                    visit(item, normalized_ticker)
        elif isinstance(current, list):
            for item in current:
                visit(item, inherited_ticker)

    visit(value)
    return prices


def _scan_quote_contract_is_complete(
    payload: Mapping[str, Any],
    *,
    audit_as_of_date: str | None = None,
) -> bool:
    integrity = payload.get("financial_integrity")
    if not isinstance(integrity, Mapping) or integrity.get("status") != PASS:
        return False
    snapshots = integrity.get("quote_snapshots")
    values = list(snapshots.values()) if isinstance(snapshots, Mapping) else snapshots
    if not isinstance(values, list) or not values:
        return False
    decision_context = _decision_context(payload)
    required_tickers = _collect_tickers(decision_context)
    expected_prices = _structured_quote_prices(decision_context)
    if not required_tickers:
        return False
    snapshot_by_ticker: dict[str, tuple[Mapping[str, Any], float]] = {}
    reference_date = _normalized_text(payload.get("as_of_date")) or _normalized_text(
        audit_as_of_date
    )
    for item in values:
        if not isinstance(item, Mapping):
            return False
        if any(
            _normalized_text(item.get(field)) is None
            for field in (
                "ticker",
                "currency",
                "as_of_date",
                "source",
                "price_unit",
                "price_basis",
                "quote_snapshot_id",
            )
        ):
            return False
        price = _as_float(item.get("price"))
        ticker = str(item.get("ticker") or "").strip().upper()
        snapshot_id = _normalized_text(item.get("quote_snapshot_id"))
        split_factor = _as_float(item.get("split_adjustment_factor"))
        price_basis = item.get("price_basis")
        split_effective_date = _normalized_text(item.get("split_effective_date"))
        raw_price = _as_float(item.get("raw_price"))
        if (
            price is None
            or price <= 0
            or not ticker
            or ticker in snapshot_by_ticker
            or item.get("currency") != "USD"
            or item.get("price_unit") != "USD_per_share"
            or price_basis not in {"UNADJUSTED", "SPLIT_ADJUSTED"}
            or snapshot_id is None
            or re.fullmatch(r"[0-9a-f]{64}", snapshot_id) is None
            or split_factor is None
            or split_factor <= 0
            or "split_effective_date" not in item
        ):
            return False
        quote_date = _normalized_text(item.get("as_of_date"))
        if (
            quote_date is None
            or not _date_is_valid_at_or_before(quote_date, quote_date)
            or (
                reference_date is not None
                and not _date_is_valid_at_or_before(quote_date, reference_date)
            )
        ):
            return False
        if price_basis == "UNADJUSTED":
            if not _close(split_factor, 1.0):
                return False
            if raw_price is not None and not _close(raw_price, price):
                return False
            if split_effective_date is not None and (
                reference_date is None
                or not _date_is_valid_at_or_before(split_effective_date, reference_date)
            ):
                return False
        else:
            if (
                raw_price is None
                or raw_price <= 0
                or split_effective_date is None
                or reference_date is None
                or not _date_is_valid_at_or_before(split_effective_date, reference_date)
                or not _close(raw_price / price, split_factor)
            ):
                return False
        from app.autonomous.financial_integrity import stable_quote_hash

        if snapshot_id != stable_quote_hash(item):
            return False
        snapshot_by_ticker[ticker] = (item, price)
    if set(snapshot_by_ticker) != required_tickers:
        return False
    for ticker, prices in expected_prices.items():
        snapshot = snapshot_by_ticker.get(ticker)
        if snapshot is None or any(not _close(price, snapshot[1]) for price in prices):
            return False
    return True


def _scan_payload_violations(
    payload: Mapping[str, Any],
    *,
    audit_as_of_date: str | None = None,
) -> list[dict[str, Any]]:
    financial_paths = _scan_financial_value_paths(_decision_context(payload))
    nonfinite_paths = _nonfinite_financial_value_paths(payload)
    if not financial_paths and not nonfinite_paths:
        return []
    triage = payload.get("triage")
    llm_consumed = bool(
        isinstance(triage, Mapping)
        and (
            _normalized_text(triage.get("full_prompt")) is not None
            or _normalized_text(triage.get("prompt_preview")) is not None
            or triage.get("survivors")
        )
    ) or bool(payload.get("deep_reviews"))
    findings: list[dict[str, Any]] = []
    quote_contract_complete = _scan_quote_contract_is_complete(
        payload,
        audit_as_of_date=audit_as_of_date,
    )
    if not quote_contract_complete:
        findings.append(
            _violation(
                invariant="SCAN_FINANCIAL_PROVENANCE_MISSING",
                ticker=None,
                run_id=_normalized_text(payload.get("run_id")),
                as_of_date=_normalized_text(payload.get("as_of_date")),
                field="financial_decision_context",
                source_values={
                    "financial_value_paths": financial_paths[:25],
                    "financial_value_path_count": len(financial_paths),
                    "financial_integrity_status": (
                        payload.get("financial_integrity", {}).get("status")
                        if isinstance(payload.get("financial_integrity"), Mapping)
                        else None
                    ),
                },
                expected_relationship=(
                    "a scan carrying investment prices or valuation metrics has a PASS "
                    "integrity attestation and complete per-ticker quote snapshots"
                ),
                observed_relationship=(
                    f"financial_values_present={len(financial_paths)}; "
                    "complete_quote_contract=false"
                ),
                reason=(
                    "The scan artifact cannot prove which quote, currency, as-of date, and "
                    "split basis its decision context used."
                ),
                llm_consumed=llm_consumed,
                repair_classification=MISSING_PROVENANCE_UNREPAIRABLE,
            )
        )
    if nonfinite_paths:
        findings.append(
            _violation(
                invariant="NONFINITE_FINANCIAL_VALUE",
                ticker=None,
                run_id=_normalized_text(payload.get("run_id")),
                as_of_date=_normalized_text(payload.get("as_of_date"))
                or _normalized_text(audit_as_of_date),
                field="scan_financial_context",
                source_values={"nonfinite_paths": nonfinite_paths[:25]},
                expected_relationship="all investment-relevant scan values are finite",
                observed_relationship=f"nonfinite_paths={','.join(nonfinite_paths[:25])}",
                reason="The scan artifact contains NaN or infinite financial values.",
                llm_consumed=llm_consumed,
                repair_classification=(
                    REQUIRES_LLM_REREVIEW if llm_consumed else REQUIRES_PACKET_REBUILD
                ),
            )
        )
    return findings


def _research_payload_violations(
    payload: Mapping[str, Any],
    *,
    audit_as_of_date: str | None = None,
) -> list[dict[str, Any]]:
    financial_paths = _scan_financial_value_paths(_decision_context(payload))
    nonfinite_paths = _nonfinite_financial_value_paths(payload)
    if not financial_paths and not nonfinite_paths:
        return []
    ticker = _normalized_text(payload.get("ticker"))
    llm_consumed = _llm_consumed(payload) or any(
        bool(payload.get(key))
        for key in (
            "analyst_notes",
            "hypotheses_generated",
            "merged_findings",
            "opus_validation",
            "thesis",
        )
    )
    findings: list[dict[str, Any]] = []
    if not _scan_quote_contract_is_complete(payload, audit_as_of_date=audit_as_of_date):
        findings.append(
            _violation(
                invariant="RESEARCH_FINANCIAL_PROVENANCE_MISSING",
                ticker=ticker,
                run_id=_normalized_text(payload.get("run_id")),
                as_of_date=_normalized_text(payload.get("as_of_date"))
                or _normalized_text(audit_as_of_date),
                field="research_financial_context",
                source_values={
                    "financial_value_paths": financial_paths[:25],
                    "financial_value_path_count": len(financial_paths),
                },
                expected_relationship=(
                    "research carrying price or valuation values has a PASS integrity "
                    "attestation and an exact canonical quote snapshot for each ticker"
                ),
                observed_relationship="complete_quote_contract=false",
                reason=(
                    "The research artifact cannot prove the quote and split lineage used "
                    "by its investment conclusions."
                ),
                llm_consumed=llm_consumed,
                repair_classification=MISSING_PROVENANCE_UNREPAIRABLE,
            )
        )
    if nonfinite_paths:
        findings.append(
            _violation(
                invariant="NONFINITE_FINANCIAL_VALUE",
                ticker=ticker,
                run_id=_normalized_text(payload.get("run_id")),
                as_of_date=_normalized_text(payload.get("as_of_date"))
                or _normalized_text(audit_as_of_date),
                field="research_financial_context",
                source_values={"nonfinite_paths": nonfinite_paths[:25]},
                expected_relationship="all investment-relevant research values are finite",
                observed_relationship=f"nonfinite_paths={','.join(nonfinite_paths[:25])}",
                reason="The research artifact contains NaN or infinite financial values.",
                llm_consumed=llm_consumed,
                repair_classification=(
                    REQUIRES_LLM_REREVIEW if llm_consumed else REQUIRES_PACKET_REBUILD
                ),
            )
        )
    return findings


def _classification_for(violations: Iterable[Mapping[str, Any]]) -> str | None:
    values = {str(item.get("repair_classification")) for item in violations}
    for classification in (
        MISSING_PROVENANCE_UNREPAIRABLE,
        REQUIRES_LLM_REREVIEW,
        REQUIRES_PACKET_REBUILD,
        SAFE_DETERMINISTIC_RERENDER,
    ):
        if classification in values:
            return classification
    return None


def _digest_decision_tickers(markdown: str) -> set[str]:
    """Return ticker identities carried by decision-bearing digest rows."""

    tickers: set[str] = set()
    ticker_column: int | None = None
    for line in markdown.splitlines():
        stripped = line.strip()
        imperative = re.match(
            r"^AT TARGET\s+([A-Z][A-Z0-9.-]{0,14}):",
            stripped,
        )
        if imperative is not None:
            tickers.add(imperative.group(1))
        if not (stripped.startswith("|") and stripped.endswith("|")):
            ticker_column = None
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if "Ticker" in cells:
            ticker_column = cells.index("Ticker")
            continue
        if ticker_column is None or ticker_column >= len(cells):
            continue
        candidate = cells[ticker_column].strip().upper()
        if re.fullmatch(r"[A-Z][A-Z0-9.-]{0,14}", candidate):
            tickers.add(candidate)
    return tickers


def _is_canonical_blocked_digest(markdown: str) -> bool:
    """Recognize only the exact non-decision digest emitted while audit is blocked."""

    match = re.fullmatch(
        (
            r"# IVI Watchlist Daily Digest\n\n"
            r"- Generated: ([^\n]+)\n\n"
            f"{re.escape(_FINANCIAL_INTEGRITY_BLOCKED_DIGEST_MESSAGE)}"
            r"\n?"
        ),
        markdown,
    )
    if match is None:
        return False
    try:
        generated_at = datetime.fromisoformat(match.group(1).replace("Z", "+00:00"))
    except ValueError:
        return False
    return generated_at.tzinfo is not None


def _digest_lineage_findings(
    digest_path: Path,
    *,
    records: Mapping[Path, Mapping[str, Any]],
    payloads: Mapping[Path, Mapping[str, Any]],
    violations_by_path: Mapping[Path, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Validate a digest against exact source artifacts, independent of dates."""

    from app.outcomes.lineage import emitted_decision_claim
    from app.watchlist.lineage import watchlist_row_matches_source_decision

    try:
        markdown_bytes = digest_path.read_bytes()
        markdown = markdown_bytes.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return [
            _violation(
                invariant="DIGEST_SOURCE_LINEAGE_MISSING",
                ticker=None,
                field="watchlist_report",
                source_values={"error": f"{type(exc).__name__}: {exc}"},
                expected_relationship=(
                    "every decision-bearing digest row has exact machine-readable source lineage"
                ),
                observed_relationship="digest could not be read for lineage validation",
                reason="The digest decision rows cannot be bound to exact source bytes.",
                llm_consumed=False,
                repair_classification=MISSING_PROVENANCE_UNREPAIRABLE,
            )
        ]
    lineage_path = digest_path.with_suffix(".lineage.json").resolve()
    if _is_canonical_blocked_digest(markdown):
        return []
    decision_tickers = _digest_decision_tickers(markdown)

    structural_errors: list[str] = []
    marker_pattern = re.compile(
        rf"<!--\s*{re.escape(_DIGEST_RENDERED_STATE_MARKER)}([A-Za-z0-9+/=]+)\s*-->"
    )
    marker_matches = list(marker_pattern.finditer(markdown))
    marker_state_sha256: str | None = None
    marker_markdown_sha256: str | None = None
    visible_markdown_sha256: str | None = None
    if len(marker_matches) != 1:
        structural_errors.append("rendered_state_marker")
    else:
        marker_match = marker_matches[0]
        if markdown[marker_match.end() :].strip():
            structural_errors.append("rendered_state_marker_position")
        visible_markdown_sha256 = _sha256_bytes(markdown[: marker_match.start()].encode("utf-8"))
        try:
            rendered_payload = json.loads(
                base64.b64decode(marker_match.group(1), validate=True).decode("utf-8")
            )
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
            rendered_payload = None
        if not isinstance(rendered_payload, Mapping):
            structural_errors.append("rendered_state_payload")
        else:
            marker_state_sha256 = _normalized_text(rendered_payload.get("rendered_state_sha256"))
            marker_markdown_sha256 = _normalized_text(
                rendered_payload.get("rendered_markdown_sha256")
            )
            if (
                rendered_payload.get("schema_version") != DIGEST_RENDERED_STATE_SCHEMA_VERSION
                or re.fullmatch(r"[0-9a-f]{64}", marker_state_sha256 or "") is None
                or re.fullmatch(r"[0-9a-f]{64}", marker_markdown_sha256 or "") is None
            ):
                structural_errors.append("rendered_state_structure")
            elif marker_markdown_sha256 != visible_markdown_sha256:
                structural_errors.append("rendered_markdown_sha256")

    lineage_payload = payloads.get(lineage_path)
    if lineage_payload is None:
        return [
            _violation(
                invariant="DIGEST_SOURCE_LINEAGE_MISSING",
                ticker=None,
                as_of_date=_record_date(digest_path, None),
                field="watchlist_report",
                source_values={
                    "decision_tickers": sorted(decision_tickers),
                    "required_lineage_path": str(lineage_path),
                },
                expected_relationship=(
                    "every decision-bearing digest row names its source run, "
                    "canonical artifact path, and exact artifact SHA-256"
                ),
                observed_relationship="required digest lineage sidecar is absent or unreadable",
                reason="Ticker/date coincidence is not proof of source lineage.",
                llm_consumed=False,
                repair_classification=MISSING_PROVENANCE_UNREPAIRABLE,
            )
        ]

    if lineage_payload.get("schema_version") != DIGEST_LINEAGE_SCHEMA_VERSION:
        structural_errors.append("schema_version")
    if lineage_payload.get("digest_path") != str(digest_path.resolve()):
        structural_errors.append("digest_path")
    digest_sha256 = _sha256_bytes(markdown_bytes)
    if lineage_payload.get("digest_sha256") != digest_sha256:
        structural_errors.append("digest_sha256")
    sidecar_state_sha256 = _normalized_text(lineage_payload.get("rendered_state_sha256"))
    sidecar_markdown_sha256 = _normalized_text(lineage_payload.get("rendered_markdown_sha256"))
    rendered_state = lineage_payload.get("rendered_state")
    required_state_fields = {
        "watchlist_rows",
        "latest_prices",
        "history_rows",
        "review_rows",
        "cheapness",
        "open_dispositions",
        "held_exit_rows",
        "data_health",
    }
    if (
        not isinstance(rendered_state, Mapping)
        or set(rendered_state) != required_state_fields
        or re.fullmatch(r"[0-9a-f]{64}", sidecar_state_sha256 or "") is None
        or _payload_sha256(rendered_state) != sidecar_state_sha256
        or sidecar_state_sha256 != marker_state_sha256
    ):
        structural_errors.append("rendered_state_sidecar")
        rendered_state = {}
    if (
        re.fullmatch(r"[0-9a-f]{64}", sidecar_markdown_sha256 or "") is None
        or sidecar_markdown_sha256 != marker_markdown_sha256
        or sidecar_markdown_sha256 != visible_markdown_sha256
    ):
        structural_errors.append("rendered_markdown_sidecar")

    list_state_fields = (
        "watchlist_rows",
        "latest_prices",
        "history_rows",
        "review_rows",
        "open_dispositions",
        "held_exit_rows",
    )
    for field in list_state_fields:
        value = rendered_state.get(field)
        if not isinstance(value, list) or any(not isinstance(row, Mapping) for row in value):
            structural_errors.append(f"rendered_state.{field}")
    cheapness_state = rendered_state.get("cheapness")
    if not isinstance(cheapness_state, Mapping) or any(
        not isinstance(value, Mapping) for value in cheapness_state.values()
    ):
        structural_errors.append("rendered_state.cheapness")
    data_health_state = rendered_state.get("data_health")
    if (
        not isinstance(data_health_state, Mapping)
        or set(data_health_state) != {"checks", "blocking"}
        or not isinstance(data_health_state.get("checks"), list)
        or any(not isinstance(check, Mapping) for check in data_health_state.get("checks", []))
        or type(data_health_state.get("blocking")) is not bool
    ):
        structural_errors.append("rendered_state.data_health")

    raw_manifest_path = _normalized_text(lineage_payload.get("manifest_path"))
    manifest_sha256 = _normalized_text(lineage_payload.get("manifest_sha256"))
    if raw_manifest_path is None or re.fullmatch(r"[0-9a-f]{64}", manifest_sha256 or "") is None:
        structural_errors.append("manifest_binding")
    else:
        try:
            manifest_path = Path(raw_manifest_path).expanduser().resolve()
            manifest_current_sha256 = _current_sha256(manifest_path)
        except (OSError, RuntimeError):
            manifest_path = None
            manifest_current_sha256 = None
        if (
            manifest_path is None
            or raw_manifest_path != str(manifest_path)
            or manifest_current_sha256 != manifest_sha256
            or _load_manifest(manifest_path) is None
        ):
            structural_errors.append("manifest_binding")

    raw_rows = lineage_payload.get("rows")
    if not isinstance(raw_rows, list):
        structural_errors.append("rows")
        raw_rows = []

    watchlist_state_rows = (
        rendered_state.get("watchlist_rows")
        if isinstance(rendered_state.get("watchlist_rows"), list)
        else []
    )
    watchlist_state_by_identity: dict[tuple[str, str], Mapping[str, Any]] = {}
    for index, state_row in enumerate(watchlist_state_rows):
        if not isinstance(state_row, Mapping):
            continue
        state_ticker = _normalized_text(state_row.get("ticker"))
        state_run_id = _normalized_text(state_row.get("source_run_id"))
        identity = (
            state_ticker.upper() if state_ticker is not None else "",
            state_run_id or "",
        )
        if not all(identity) or identity in watchlist_state_by_identity:
            structural_errors.append(f"rendered_state.watchlist_rows[{index}].identity")
            continue
        watchlist_state_by_identity[identity] = state_row

    rows_by_ticker: dict[str, list[Mapping[str, Any]]] = {}
    row_identities: list[tuple[str, str]] = []
    for index, raw_row in enumerate(raw_rows):
        if not isinstance(raw_row, Mapping):
            structural_errors.append(f"rows[{index}]")
            continue
        ticker = _normalized_text(raw_row.get("ticker"))
        run_id = _normalized_text(raw_row.get("source_run_id"))
        raw_source_path = _normalized_text(raw_row.get("source_artifact_path"))
        source_sha256 = _normalized_text(raw_row.get("source_artifact_sha256"))
        source_decision_fingerprint = _normalized_text(raw_row.get("source_decision_fingerprint"))
        decision_state = raw_row.get("decision_state")
        decision_state_sha256 = _normalized_text(raw_row.get("decision_state_sha256"))
        if (
            ticker is None
            or run_id is None
            or raw_source_path is None
            or source_sha256 is None
            or re.fullmatch(r"[0-9a-f]{64}", source_sha256) is None
            or source_decision_fingerprint is None
            or re.fullmatch(r"[0-9a-f]{64}", source_decision_fingerprint) is None
            or not isinstance(decision_state, Mapping)
            or decision_state_sha256 is None
            or re.fullmatch(r"[0-9a-f]{64}", decision_state_sha256) is None
        ):
            structural_errors.append(f"rows[{index}].required_fields")
            continue
        if (
            _payload_sha256(decision_state) != decision_state_sha256
            or _normalized_text(decision_state.get("ticker")) != ticker
            or _normalized_text(decision_state.get("source_run_id")) != run_id
            or set(decision_state) != set(DIGEST_DECISION_STATE_FIELDS)
        ):
            structural_errors.append(f"rows[{index}].decision_state")
            continue
        state_row = watchlist_state_by_identity.get((ticker.upper(), run_id))
        expected_decision_state = (
            {field: state_row.get(field) for field in DIGEST_DECISION_STATE_FIELDS}
            if state_row is not None
            else None
        )
        if expected_decision_state != dict(decision_state):
            structural_errors.append(f"rows[{index}].rendered_state_mismatch")
            continue
        try:
            source_path = Path(raw_source_path).expanduser().resolve()
        except (OSError, RuntimeError):
            structural_errors.append(f"rows[{index}].source_artifact_path")
            continue
        if raw_source_path != str(source_path):
            structural_errors.append(f"rows[{index}].source_artifact_path_canonical")
            continue
        rows_by_ticker.setdefault(ticker.upper(), []).append(raw_row)
        row_identities.append((ticker.upper(), run_id))
    if row_identities != sorted(row_identities) or len(row_identities) != len(set(row_identities)):
        structural_errors.append("rows.order_or_identity")
    if set(row_identities) != set(watchlist_state_by_identity):
        structural_errors.append("rows.rendered_state_coverage")

    watchlist_state_tickers = {ticker for ticker, _ in watchlist_state_by_identity}
    held_exit_state = (
        rendered_state.get("held_exit_rows")
        if isinstance(rendered_state.get("held_exit_rows"), list)
        else []
    )
    if held_exit_state:
        structural_errors.append("rendered_state.held_exit_rows_unbound")
    history_state = (
        rendered_state.get("history_rows")
        if isinstance(rendered_state.get("history_rows"), list)
        else []
    )
    if history_state:
        structural_errors.append("rendered_state.history_rows_unbound")

    findings: list[dict[str, Any]] = []
    if structural_errors:
        findings.append(
            _violation(
                invariant="DIGEST_SOURCE_LINEAGE_INVALID",
                ticker=None,
                as_of_date=_record_date(digest_path, None),
                field="watchlist_report",
                source_values={
                    "lineage_path": str(lineage_path),
                    "structural_errors": sorted(set(structural_errors)),
                },
                expected_relationship=(
                    "digest lineage is schema-valid and hash-bound to the exact digest bytes"
                ),
                observed_relationship="lineage sidecar failed structural validation",
                reason="The digest/source authorization binding is incomplete or stale.",
                llm_consumed=False,
                repair_classification=MISSING_PROVENANCE_UNREPAIRABLE,
            )
        )

    source_tickers = decision_tickers | watchlist_state_tickers
    for ticker in sorted(source_tickers):
        rows = rows_by_ticker.get(ticker, [])
        if len(rows) != 1:
            findings.append(
                _violation(
                    invariant="DIGEST_SOURCE_LINEAGE_MISSING",
                    ticker=ticker,
                    as_of_date=_record_date(digest_path, None),
                    field="watchlist_report",
                    source_values={
                        "lineage_path": str(lineage_path),
                        "matching_rows": len(rows),
                    },
                    expected_relationship=(
                        "each decision-bearing digest ticker has exactly one exact source binding"
                    ),
                    observed_relationship=f"matching_lineage_rows={len(rows)}",
                    reason="The rendered decision cannot be attributed to one authorized run.",
                    llm_consumed=False,
                    repair_classification=MISSING_PROVENANCE_UNREPAIRABLE,
                )
            )
            continue
        row = rows[0]
        run_id = str(row["source_run_id"]).strip()
        raw_source_path = str(row["source_artifact_path"]).strip()
        source_path = Path(raw_source_path).expanduser().resolve()
        source_sha256 = str(row["source_artifact_sha256"]).strip()
        source_decision_fingerprint = str(row["source_decision_fingerprint"]).strip()
        source_record = records.get(source_path)
        source_payload = payloads.get(source_path)
        source_violations = violations_by_path.get(source_path, ())
        source_tickers = _collect_tickers(source_payload) if source_payload is not None else set()
        emitted_claim = (
            emitted_decision_claim(source_payload, ticker) if source_payload is not None else None
        )
        decision_state = row.get("decision_state")
        stated_grade = (
            str(decision_state.get("conviction_grade") or "").strip().upper()
            if isinstance(decision_state, Mapping)
            else ""
        )
        stated_grade = {
            "WATCH": "WATCHLIST_ONLY",
            "WATCHLIST": "WATCHLIST_ONLY",
        }.get(stated_grade, stated_grade)
        source_authorized = (
            source_record is not None
            and source_record.get("family") == "autonomous_sector"
            and source_path.name == "autonomous_sector_run.json"
            and source_record.get("run_id") == run_id
            and source_record.get("sha256") == source_sha256
            and not source_violations
            and ticker in source_tickers
            and emitted_claim is not None
            and emitted_claim.get("expected_grade") == stated_grade
            and emitted_claim.get("source_decision_fingerprint") == source_decision_fingerprint
            and isinstance(decision_state, Mapping)
            and watchlist_row_matches_source_decision(
                decision_state,
                source_payload,
                ticker=ticker,
            )
        )
        if source_authorized:
            continue
        findings.append(
            _violation(
                invariant="INVALID_SOURCE_RUN_IN_CURRENT_REPORT",
                ticker=ticker,
                as_of_date=_record_date(digest_path, None),
                field="watchlist_report",
                source_values={
                    "source_run_id": run_id,
                    "source_artifact_path": raw_source_path,
                    "source_artifact_sha256": source_sha256,
                    "source_decision_fingerprint": source_decision_fingerprint,
                    "source_record_present": source_record is not None,
                    "source_record_family": (
                        source_record.get("family") if source_record is not None else None
                    ),
                    "source_record_run_id": (
                        source_record.get("run_id") if source_record is not None else None
                    ),
                    "source_record_sha256": (
                        source_record.get("sha256") if source_record is not None else None
                    ),
                    "source_tickers": sorted(source_tickers),
                },
                expected_relationship=(
                    "the digest row points to a current PASS autonomous-sector artifact "
                    "with matching run ID, ticker, path, and SHA-256"
                ),
                observed_relationship="exact source authorization failed",
                reason="The current report propagated an invalid or unaudited decision state.",
                llm_consumed=any(bool(item.get("llm_consumed")) for item in source_violations),
                repair_classification=(
                    _classification_for(source_violations) or MISSING_PROVENANCE_UNREPAIRABLE
                ),
            )
        )
    return findings


def _render_markdown(manifest: Mapping[str, Any]) -> str:
    summary = manifest["summary"]
    lines = [
        "# Deterministic Financial-Integrity Audit",
        "",
        f"Generated: {manifest['generated_at']}",
        "",
        "This report is read-only. No historical source artifact was rewritten.",
        "",
        "## Scope",
        "",
        f"- Artifacts scanned: {summary['artifacts_scanned']}",
        f"- Tickers scanned: {summary['tickers_scanned']}",
        f"- Violations: {summary['violations']}",
        f"- Affected run IDs: {summary['affected_run_ids']}",
        f"- Affected tickers: {summary['affected_tickers']}",
        f"- Earliest date: {summary['earliest_date'] or 'n/a'}",
        f"- Latest date: {summary['latest_date'] or 'n/a'}",
        "",
        "## Violations by invariant",
        "",
    ]
    counts = summary.get("violations_by_invariant") or {}
    if counts:
        lines.extend(f"- {key}: {value}" for key, value in sorted(counts.items()))
    else:
        lines.append("- None")
    lines.extend(["", "## Findings", ""])
    violations = manifest.get("violations") or []
    if not violations:
        lines.append("No deterministic financial-integrity violations were detected.")
    for index, item in enumerate(violations, start=1):
        lines.extend(
            [
                f"### {index}. {item['invariant']}",
                "",
                f"- Artifact: `{item['artifact_path']}`",
                f"- Run: {item.get('run_id') or 'n/a'}",
                f"- Ticker: {item.get('ticker') or 'n/a'}",
                f"- Date: {item.get('as_of_date') or 'n/a'}",
                f"- Field: `{item['field']}`",
                f"- LLM consumed invalid value: {'yes' if item['llm_consumed'] else 'no'}",
                f"- Repair classification: {item['repair_classification']}",
                f"- Expected: {item['expected_relationship']}",
                f"- Observed: {item['observed_relationship']}",
                f"- Reason: {item['reason']}",
                f"- Required action: {item['invalidation_action']}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def audit_artifact_tree(
    *,
    runs_root: str | Path,
    analyst_outputs_root: str | Path,
    analysis_dir: str | Path,
    scans_root: str | Path | None = None,
    research_outputs_root: str | Path | None = None,
    watchlist_report_roots: Sequence[str | Path] = (),
    generated_at: datetime | None = None,
) -> AuditReportPaths:
    """Audit source trees and write one timestamped JSON/Markdown report pair."""

    now = generated_at or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now = now.astimezone(timezone.utc)
    generated_iso = now.isoformat().replace("+00:00", "Z")
    if scans_root is None or research_outputs_root is None or len(watchlist_report_roots) != 1:
        raise ValueError(
            "A canonical financial-integrity audit requires exactly one root for each "
            "of autonomous_sector, analyst_output, scan, research_output, and "
            "watchlist_report."
        )
    roots: list[tuple[str, Path]] = [
        ("autonomous_sector", Path(runs_root)),
        ("analyst_output", Path(analyst_outputs_root)),
        ("scan", Path(scans_root)),
        ("research_output", Path(research_outputs_root)),
        ("watchlist_report", Path(watchlist_report_roots[0])),
    ]
    resolved_root_paths = [path.expanduser().resolve() for _family, path in roots]
    if len(set(resolved_root_paths)) != len(resolved_root_paths):
        raise ValueError("Canonical financial-integrity audit roots must be distinct.")
    missing_roots = [str(path) for path in resolved_root_paths if not path.is_dir()]
    if missing_roots:
        raise ValueError(
            "Canonical financial-integrity audit roots must already exist: "
            + ", ".join(missing_roots)
        )
    resolved_roots = {family: path.expanduser().resolve() for family, path in roots}
    expected_roots = _canonical_audit_roots()
    if resolved_roots != expected_roots:
        mismatches = [
            f"{family}: expected {expected_roots[family]}, got {resolved_roots.get(family)}"
            for family in CANONICAL_AUDIT_ROOT_IDS
            if resolved_roots.get(family) != expected_roots[family]
        ]
        raise ValueError(
            "Canonical financial-integrity audit roots must match the active "
            "runtime scope: " + "; ".join(mismatches)
        )
    discovered = _discover_files(roots)

    records: dict[Path, dict[str, Any]] = {}
    payloads: dict[Path, Mapping[str, Any]] = {}
    all_tickers: set[str] = set()
    all_dates: set[str] = set()
    violations_by_path: dict[Path, list[dict[str, Any]]] = {}

    for family, path in discovered:
        data = path.read_bytes()
        payload: Mapping[str, Any] | None = None
        parse_error: str | None = None
        if path.suffix.lower() == ".json":
            try:
                parsed = json.loads(data.decode("utf-8"))
                if not isinstance(parsed, dict):
                    raise ValueError(f"artifact root is {type(parsed).__name__}, expected object")
                payload = parsed
                payloads[path] = payload
                all_tickers.update(_collect_tickers(payload))
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                parse_error = f"{type(exc).__name__}: {exc}"
                violations_by_path[path] = [
                    _violation(
                        invariant="UNREADABLE_ARTIFACT",
                        ticker=None,
                        field="artifact",
                        source_values={"parse_error": parse_error},
                        expected_relationship="artifact is a UTF-8 JSON object",
                        observed_relationship=parse_error,
                        reason="The artifact cannot be deterministically inspected.",
                        llm_consumed=False,
                        repair_classification=MISSING_PROVENANCE_UNREPAIRABLE,
                    )
                ]
        record_date = _record_date(path, payload)
        if record_date:
            all_dates.add(record_date)
        records[path] = {
            "path": str(path),
            "family": family,
            "sha256": _sha256_bytes(data),
            "size_bytes": len(data),
            "run_id": _normalized_text(payload.get("run_id")) if payload else None,
            "ticker": _normalized_text(payload.get("ticker")) if payload else None,
            "as_of_date": record_date,
            "parse_error": parse_error,
            "integrity_status": PASS,
            "decision_eligible": True,
            "violation_invariants": [],
            "repair_classification": None,
            "llm_consumed": _llm_consumed(payload) if payload else False,
            "source_artifact_path": None,
        }

    for path, payload in payloads.items():
        direct = _sector_payload_violations(payload)
        family = records[path]["family"]
        if family == "analyst_output":
            direct.extend(_analyst_payload_violations(payload))
        elif family == "scan":
            direct.extend(
                _scan_payload_violations(
                    payload,
                    audit_as_of_date=records[path]["as_of_date"],
                )
            )
        elif family == "research_output":
            direct.extend(
                _research_payload_violations(
                    payload,
                    audit_as_of_date=records[path]["as_of_date"],
                )
            )
        if direct:
            violations_by_path.setdefault(path, []).extend(direct)

    primary_paths_by_run_id: dict[str, list[Path]] = {}
    autonomous_primary_paths: list[Path] = []
    for path, payload in payloads.items():
        if (
            records[path]["family"] != "autonomous_sector"
            or path.name != "autonomous_sector_run.json"
        ):
            continue
        autonomous_primary_paths.append(path)
        run_id = _normalized_text(payload.get("run_id"))
        if run_id is not None:
            primary_paths_by_run_id.setdefault(run_id, []).append(path)

    autonomous_root = resolved_roots["autonomous_sector"]
    for path in autonomous_primary_paths:
        record = records[path]
        run_id = _normalized_text(payloads[path].get("run_id"))
        identity_paths = primary_paths_by_run_id.get(run_id, []) if run_id is not None else [path]
        expected_path = _canonical_autonomous_primary_path(run_id, root=autonomous_root)
        if run_id is not None and len(identity_paths) == 1 and path == expected_path:
            continue
        violations_by_path.setdefault(path, []).append(
            _violation(
                invariant="AUTONOMOUS_RUN_PRIMARY_IDENTITY_INVALID",
                ticker=None,
                run_id=run_id,
                as_of_date=record.get("as_of_date"),
                field="run_id",
                source_values={
                    "serialized_run_id": run_id,
                    "expected_primary_path": (
                        str(expected_path) if expected_path is not None else None
                    ),
                    "observed_primary_paths": sorted(str(item) for item in identity_paths),
                    "observed_primary_count": len(identity_paths),
                },
                expected_relationship=(
                    "one autonomous-sector run ID maps to exactly one primary artifact at "
                    "<autonomous_sector_root>/<run_id>/autonomous_sector_run.json"
                ),
                observed_relationship=(
                    "the serialized run identity is missing, duplicated, or stored at a "
                    "noncanonical primary path"
                ),
                reason=(
                    "Path order must never choose between multiple or mislocated primary "
                    "artifacts for one decision identity."
                ),
                llm_consumed=bool(record.get("llm_consumed")),
                repair_classification=MISSING_PROVENANCE_UNREPAIRABLE,
            )
        )

    # A contradictory nested analyst quote invalidates the analyst artifact
    # itself when its quote has no source/date/currency provenance.
    analyst_by_key: dict[tuple[str, str], list[tuple[Path, float, bool]]] = {}
    for path, payload in payloads.items():
        if records[path]["family"] != "analyst_output" or path.name != "analysis_report.json":
            continue
        info = _analyst_info(payload)
        if info is not None:
            ticker, as_of_date, price, missing = info
            provenance = not missing
            analyst_by_key.setdefault((ticker, as_of_date), []).append((path, price, provenance))

    for run_path, run_violations in list(violations_by_path.items()):
        for violation in run_violations:
            if violation.get("invariant") != "NESTED_ANALYST_QUOTE_MISMATCH":
                continue
            ticker = str(violation.get("ticker") or "")
            as_of_date = str(violation.get("as_of_date") or "")
            source_values = violation.get("source_values") or {}
            reported_path = source_values.get("analyst_report_path")
            candidates: list[tuple[Path, float, bool]] = []
            if reported_path:
                resolved = Path(str(reported_path)).expanduser().resolve()
                if resolved in payloads:
                    info = _analyst_info(payloads[resolved])
                    if info is not None:
                        candidates.append((resolved, info[2], info[3]))
            candidates.extend(analyst_by_key.get((ticker, as_of_date), []))
            seen_paths: set[Path] = set()
            for analyst_path, analyst_price, provenance in candidates:
                if analyst_path in seen_paths or provenance:
                    continue
                seen_paths.add(analyst_path)
                if any(
                    item.get("invariant") == "ANALYST_QUOTE_PROVENANCE_MISSING"
                    for item in violations_by_path.get(analyst_path, [])
                ):
                    continue
                violations_by_path.setdefault(analyst_path, []).append(
                    _violation(
                        invariant="ANALYST_QUOTE_PROVENANCE_MISSING",
                        ticker=ticker,
                        as_of_date=as_of_date,
                        field="valuation.price",
                        source_values={
                            "analyst_price": analyst_price,
                            "contradictory_run_path": str(run_path),
                            "price_source": None,
                            "price_as_of_date": None,
                            "price_currency": None,
                        },
                        expected_relationship="a decision price identifies source, as-of date, currency, and snapshot basis",
                        observed_relationship="contradictory price is present without quote provenance",
                        reason="The historical quote identity cannot be reconstructed from the analyst artifact.",
                        llm_consumed=True,
                        repair_classification=MISSING_PROVENANCE_UNREPAIRABLE,
                    )
                )

    def inherit_source_invalidity(
        source: Path,
        target: Path,
        *,
        expected_relationship: str,
        reason: str,
    ) -> None:
        source_items = violations_by_path.get(source)
        if not source_items or target in violations_by_path:
            return
        records[target]["source_artifact_path"] = str(source)
        violations_by_path[target] = [
            _violation(
                invariant="SOURCE_ARTIFACT_INVALID",
                ticker=records[source].get("ticker"),
                run_id=records[source].get("run_id"),
                as_of_date=records[source].get("as_of_date"),
                field="source_artifact",
                source_values={
                    "source_artifact_path": str(source),
                    "source_sha256": records[source]["sha256"],
                },
                expected_relationship=expected_relationship,
                observed_relationship=(
                    "source artifact has deterministic financial-integrity violations"
                ),
                reason=reason,
                llm_consumed=any(bool(item.get("llm_consumed")) for item in source_items),
                repair_classification=_classification_for(source_items)
                or MISSING_PROVENANCE_UNREPAIRABLE,
            )
        ]

    # Propagate only along explicit artifact dependencies.  In particular, an
    # analyst report is downstream of its evidence bundle; it is never a
    # source for that bundle merely because both files share a directory.
    for source in list(violations_by_path):
        source_record = records[source]
        if (
            source_record["family"] == "autonomous_sector"
            and source.name == "autonomous_sector_run.json"
        ):
            target = source.with_name("autonomous_sector_report.md")
            if target in records:
                inherit_source_invalidity(
                    source,
                    target,
                    expected_relationship=(
                        "the sector report renders a financially valid autonomous-sector run"
                    ),
                    reason="This sector report inherits the invalid source-run decision state.",
                )

    evidence_sources = [
        path
        for path in list(violations_by_path)
        if records[path]["family"] == "analyst_output"
        and path.name == "analysis_evidence_bundle.json"
    ]
    for source in evidence_sources:
        report_json = source.with_name("analysis_report.json")
        if report_json in records:
            inherit_source_invalidity(
                source,
                report_json,
                expected_relationship=(
                    "the analyst report derives from a financially valid evidence bundle"
                ),
                reason="This analyst report inherits invalid evidence-bundle context.",
            )
    report_sources = [
        path
        for path in list(violations_by_path)
        if records[path]["family"] == "analyst_output" and path.name == "analysis_report.json"
    ]
    for source in report_sources:
        report_markdown = source.with_suffix(".md")
        if report_markdown in records:
            inherit_source_invalidity(
                source,
                report_markdown,
                expected_relationship=(
                    "the analyst rendering derives from a financially valid analyst report"
                ),
                reason="This analyst rendering inherits the invalid report decision state.",
            )

    # Scan directories can contain many unrelated dates/sectors, so propagate
    # invalidity only to the JSON artifact's explicitly matching Markdown
    # companion rather than quarantining every sibling in the directory.
    for source in list(violations_by_path):
        source_record = records[source]
        if source_record["family"] != "scan" or source.suffix.lower() != ".json":
            continue
        companion_names = {f"{source.stem.removesuffix('_artifacts')}.md"}
        source_payload = payloads.get(source)
        artifact_paths = (
            source_payload.get("artifact_paths") if isinstance(source_payload, Mapping) else None
        )
        if isinstance(artifact_paths, Mapping):
            markdown_path = _normalized_text(artifact_paths.get("markdown"))
            if markdown_path is not None:
                companion_names.add(Path(markdown_path).name)
        for companion in records:
            if (
                companion == source
                or records[companion]["family"] != "scan"
                or companion.parent != source.parent
                or companion.name not in companion_names
            ):
                continue
            inherit_source_invalidity(
                source,
                companion,
                expected_relationship=(
                    "a scan rendering derives from a financially valid, provenance-complete "
                    "source artifact"
                ),
                reason="This scan report inherits the invalid source decision state.",
            )

    # Research JSON files are authoritative; only their exact or prefix-bound
    # Markdown rendering inherits invalidity.  Unrelated same-directory reports
    # remain independently auditable.
    for source in list(violations_by_path):
        source_record = records[source]
        if source_record["family"] != "research_output" or source.suffix.lower() != ".json":
            continue
        companion_paths: set[Path] = set()
        source_payload = payloads.get(source)
        if isinstance(source_payload, Mapping):
            report_path = _normalized_text(source_payload.get("report_path"))
            if report_path is not None:
                candidate = Path(report_path).expanduser()
                if not candidate.is_absolute():
                    candidate = Path.cwd() / candidate
                companion_paths.add(candidate.resolve())
        companion_paths.update(
            candidate
            for candidate in records
            if records[candidate]["family"] == "research_output"
            and candidate.parent == source.parent
            and candidate.suffix.lower() == ".md"
            and (candidate.stem == source.stem or candidate.name.startswith(f"{source.stem}_"))
        )
        for companion in companion_paths:
            if companion not in records or records[companion]["family"] != "research_output":
                continue
            inherit_source_invalidity(
                source,
                companion,
                expected_relationship=(
                    "a research rendering derives from a financially valid, "
                    "provenance-complete source artifact"
                ),
                reason="This research report inherits the invalid source decision state.",
            )

    # Digest publication is authorized by an exact per-ticker source binding.
    # Date/ticker coincidence is intentionally irrelevant: a current digest
    # may carry a decision from a much older source run.
    for path, record in records.items():
        if record["family"] != "watchlist_report" or path.suffix.lower() != ".md":
            continue
        findings = _digest_lineage_findings(
            path,
            records=records,
            payloads=payloads,
            violations_by_path=violations_by_path,
        )
        if findings:
            violations_by_path.setdefault(path, []).extend(findings)

    flat_violations: list[dict[str, Any]] = []
    for path, items in violations_by_path.items():
        record = records[path]
        record["integrity_status"] = INVALID
        record["decision_eligible"] = False
        record["violation_invariants"] = sorted({str(item["invariant"]) for item in items})
        record["repair_classification"] = _classification_for(items)
        record["llm_consumed"] = any(bool(item.get("llm_consumed")) for item in items)
        for item in items:
            flat_violations.append(
                {"artifact_path": str(path), "artifact_sha256": record["sha256"], **item}
            )

    invariant_counts = Counter(str(item["invariant"]) for item in flat_violations)
    affected_runs = sorted({str(item["run_id"]) for item in flat_violations if item.get("run_id")})
    affected_tickers = sorted(
        {str(item["ticker"]) for item in flat_violations if item.get("ticker")}
    )
    manifest: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "audit_scope_id": AUDIT_SCOPE_ID,
        "generated_at": generated_iso,
        "complete": True,
        "source_roots": [
            {
                "family": family,
                "root_id": CANONICAL_AUDIT_ROOT_IDS[family],
                "path": str(path.expanduser().resolve()),
            }
            for family, path in roots
        ],
        "summary": {
            "artifacts_scanned": len(records),
            "tickers_scanned": len(all_tickers),
            "violations": len(flat_violations),
            "violations_by_invariant": dict(sorted(invariant_counts.items())),
            "affected_run_ids": len(affected_runs),
            "affected_tickers": len(affected_tickers),
            "affected_run_id_values": affected_runs,
            "affected_ticker_values": affected_tickers,
            "earliest_date": min(all_dates) if all_dates else None,
            "latest_date": max(all_dates) if all_dates else None,
            "llm_consumed_violation_count": sum(
                bool(item.get("llm_consumed")) for item in flat_violations
            ),
            "source_artifacts_rewritten": 0,
        },
        "invalid_run_ids": affected_runs,
        "artifacts": [records[path] for path in sorted(records, key=str)],
        "violations": sorted(
            flat_violations,
            key=lambda item: (
                str(item.get("artifact_path") or ""),
                str(item.get("ticker") or ""),
                str(item.get("invariant") or ""),
            ),
        ),
    }

    output_dir = Path(analysis_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = now.strftime("%Y%m%dT%H%M%S%fZ")
    base = output_dir / f"{AUDIT_FILENAME_PREFIX}{timestamp}"
    manifest_path = base.with_suffix(".json")
    report_path = base.with_suffix(".md")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report_path.write_text(_render_markdown(manifest), encoding="utf-8")
    return AuditReportPaths(manifest_json=manifest_path, report_markdown=report_path)


__all__ = [
    "AUDIT_SCOPE_ID",
    "AUDIT_SCHEMA_VERSION",
    "AuditReportPaths",
    "CANONICAL_AUDIT_ROOT_IDS",
    "ELIGIBILITY_STATES",
    "INVALID",
    "MISSING_PROVENANCE_UNREPAIRABLE",
    "PASS",
    "REPAIR_CLASSIFICATIONS",
    "REQUIRES_LLM_REREVIEW",
    "REQUIRES_PACKET_REBUILD",
    "RUN_AUTHORIZATION_FILENAME",
    "RUN_AUTHORIZATION_SCHEMA_VERSION",
    "SAFE_DETERMINISTIC_RERENDER",
    "STALE_AUDIT",
    "TYPED_AUTHORIZATION_SCHEMA_VERSION",
    "TYPED_AUTHORIZATION_SUFFIX",
    "UNAUDITED",
    "active_financial_integrity_manifest_path",
    "authorized_artifact_bytes",
    "authorized_run_artifact_binding",
    "authorized_run_ticker_artifact_binding",
    "artifact_decision_eligibility",
    "audit_artifact_tree",
    "audit_payload",
    "financial_integrity_manifest_is_usable",
    "invalid_run_ids_from_active_manifest",
    "run_id_is_decision_eligible",
    "run_ticker_is_decision_eligible",
    "write_run_financial_authorization",
    "write_typed_financial_authorization",
]
