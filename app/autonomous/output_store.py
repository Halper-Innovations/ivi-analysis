"""Persistence helpers for autonomous analyst run artifacts."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

from app.autonomous.artifact_financial_audit import (
    PASS as FINANCIAL_INTEGRITY_PASS,
    artifact_decision_eligibility,
    write_run_financial_authorization,
)
from app.autonomous.financial_integrity import (
    FinancialIntegrityScope,
    require_unchanged_financial_integrity_scope,
)
from app.autonomous.sector_contract import AutonomousSectorFinancialRunArtifact
from app.autonomous.sector_report import render_autonomous_sector_report
from app.autonomous.run_report import render_autonomous_run_report
from app.autonomous.run_contract import AutonomousRunArtifact
from app.config import ensure_directories, get_config
from app.valuation.lineage import (
    bind_authorized_valuation_rows,
    valuation_records_have_writer_receipts,
    valuation_source_record,
)


PRODUCT_SUCCESS_STATUS = "COMPLETED"
_CONTROLLED_VALUATION_METHODS = frozenset(
    {
        "owner_earnings",
        "dcf",
        "epv",
        "graham",
        "ncav",
        "ev_ebit",
        "fcf_yield",
        "tangible_floor",
        "scorecard",
        "roic",
        "capital_structure",
        "reverse_dcf",
        "dcf_adjusted",
        "epv_adjusted",
        "tech_adjustment",
    }
)


@dataclass(frozen=True)
class AutonomousRunPaths:
    artifact_json: Path
    report_md: Path


@dataclass(frozen=True)
class AutonomousSectorRunPaths:
    artifact_json: Path
    report_md: Path
    authorization_json: Path | None = None


@dataclass(frozen=True)
class AutonomousSectorDiagnosticPaths:
    artifact_json: Path


def _require_publication_financial_integrity_binding(
    binding: Any,
    *,
    packets: Sequence[Any] | None = None,
    scenarios: Sequence[Any] | None = None,
) -> None:
    """Reject a product write unless its original full scope still matches."""

    if not isinstance(binding, Mapping):
        raise RuntimeError(
            "Refusing to persist autonomous product artifact without an immutable "
            "financial-integrity run binding."
        )
    if str(binding.get("schema_version") or "") != "financial_integrity_run_binding_v1":
        raise RuntimeError("Refusing malformed financial-integrity run binding.")
    expected = str(binding.get("scope_fingerprint") or "").strip()
    context = str(binding.get("context") or "").strip()
    run_as_of_date = str(binding.get("run_as_of_date") or "").strip()
    stored_packets = binding.get("packets")
    stored_scenarios = binding.get("scenarios")
    if (
        not expected
        or not context
        or not run_as_of_date
        or not isinstance(stored_packets, list)
        or not isinstance(stored_scenarios, list)
    ):
        raise RuntimeError("Refusing incomplete financial-integrity run binding.")
    require_unchanged_financial_integrity_scope(
        FinancialIntegrityScope(
            context=context,
            run_as_of_date=run_as_of_date,
            packets=tuple(stored_packets),
            scenarios=tuple(stored_scenarios),
        ),
        expected_scope_fingerprint=expected,
    )
    if packets is not None or scenarios is not None:
        require_unchanged_financial_integrity_scope(
            FinancialIntegrityScope(
                context=context,
                run_as_of_date=run_as_of_date,
                packets=tuple(packets or ()),
                scenarios=tuple(scenarios or ()),
            ),
            expected_scope_fingerprint=expected,
        )


def _successful_tool_call_count(tool_calls: list) -> int:
    return len([call for call in tool_calls if getattr(call, "status", None) == "OK"])


def _successful_artifact_tool_call_count(
    artifact: AutonomousRunArtifact | AutonomousSectorFinancialRunArtifact,
) -> int:
    count = _successful_tool_call_count(list(getattr(artifact, "tool_calls", []) or []))
    if (
        not isinstance(artifact, AutonomousSectorFinancialRunArtifact)
        or artifact.pipeline_version != "v2"
    ):
        return count
    for run in artifact.company_autonomy_runs:
        if not isinstance(run, dict):
            continue
        nested = run.get("artifact")
        nested_calls = nested.get("tool_calls") if isinstance(nested, dict) else None
        if isinstance(nested_calls, list):
            count += len(
                [
                    call
                    for call in nested_calls
                    if isinstance(call, dict) and str(call.get("status") or "").upper() == "OK"
                ]
            )
        else:
            count += int(run.get("tool_calls") or 0)
    validation = artifact.selection_validation
    if validation is not None:
        count += _successful_tool_call_count(list(validation.tool_calls or []))
    return count


def _controlled_valuation_source_records(
    *,
    artifact: AutonomousSectorFinancialRunArtifact,
    run_id: str,
    as_of_date: str,
    tickers: set[str],
    source_records: Sequence[Mapping[str, Any]] | None,
) -> list[dict]:
    """Normalize exact writer/packet records without rereading mutable DB rows."""

    if not tickers:
        return []
    embedded_records: list[Mapping[str, Any]] = []
    for packet in artifact.company_packets:
        packet_records = (
            packet.valuation.get("source_records") if isinstance(packet.valuation, dict) else None
        )
        if packet_records is None:
            continue
        if not isinstance(packet_records, list) or any(
            not isinstance(record, Mapping) for record in packet_records
        ):
            raise RuntimeError(
                f"Refusing malformed controlled valuation records for {packet.ticker}"
            )
        packet_ticker = str(packet.ticker or "").strip().upper()
        if any(
            not isinstance(record.get("row"), Mapping)
            or str(record["row"].get("ticker") or "").strip().upper() != packet_ticker
            for record in packet_records
        ):
            raise RuntimeError(
                f"Refusing cross-ticker valuation records for {packet_ticker or 'UNKNOWN'}"
            )
        embedded_records.extend(packet_records)
    repair = (
        artifact.candidate_selection.get("data_gap_repair")
        if isinstance(artifact.candidate_selection, dict)
        else None
    )
    candidate_states = repair.get("candidate_states") if isinstance(repair, dict) else None
    if candidate_states is not None:
        if not isinstance(candidate_states, list):
            raise RuntimeError("Refusing malformed valuation repair candidate states")
        for candidate_state in candidate_states:
            if not isinstance(candidate_state, Mapping):
                raise RuntimeError("Refusing malformed valuation repair candidate state")
            candidate_records = candidate_state.get("valuation_source_records")
            if candidate_records is None:
                continue
            if not isinstance(candidate_records, list) or any(
                not isinstance(record, Mapping) for record in candidate_records
            ):
                raise RuntimeError("Refusing malformed controlled valuation repair records")
            candidate_ticker = str(candidate_state.get("ticker") or "").strip().upper()
            if not candidate_ticker or any(
                not isinstance(record.get("row"), Mapping)
                or str(record["row"].get("ticker") or "").strip().upper() != candidate_ticker
                for record in candidate_records
            ):
                raise RuntimeError(
                    "Refusing valuation repair records not bound to their candidate stage"
                )
            embedded_records.extend(candidate_records)

    def normalize(raw_records: Sequence[Mapping[str, Any]]) -> list[dict]:
        records: list[dict] = []
        identities: set[tuple[str, str, str, str]] = set()
        for raw_record in raw_records:
            if (
                not isinstance(raw_record, Mapping)
                or set(raw_record) != {"schema_version", "row"}
                or not isinstance(raw_record.get("row"), Mapping)
            ):
                raise RuntimeError("Refusing malformed controlled valuation source record")
            row = dict(raw_record["row"])
            record = valuation_source_record(row)
            ticker = str(row.get("ticker") or "").strip().upper()
            method = str(row.get("method") or "").strip()
            identity = (
                ticker,
                str(row.get("as_of_date") or "").strip(),
                method,
                str(row.get("created_at") or "").strip(),
            )
            if (
                record is None
                or record != dict(raw_record)
                or ticker not in tickers
                or identity[1] != as_of_date
                or str(row.get("source_run_id") or "").strip() != run_id
                or method not in _CONTROLLED_VALUATION_METHODS
                or not all(identity)
                or identity in identities
            ):
                raise RuntimeError(
                    "Refusing uncontrolled or malformed valuation source claim for "
                    f"{ticker or 'UNKNOWN'}:{method or 'UNKNOWN'}"
                )
            identities.add(identity)
            records.append(record)
        return sorted(
            records,
            key=lambda record: (
                str(record["row"]["ticker"]).upper(),
                str(record["row"]["as_of_date"]),
                str(record["row"]["method"]),
                str(record["row"]["created_at"]),
            ),
        )

    canonical_records = normalize(embedded_records)
    if canonical_records and not valuation_records_have_writer_receipts(
        run_id,
        canonical_records,
    ):
        raise RuntimeError(
            "Refusing valuation source claims without exact deterministic-writer receipts"
        )
    if source_records is not None:
        # The optional argument is only a transport assertion. It can never
        # introduce or override valuation claims absent from the artifact's
        # exact writer-stage/packet state.
        supplied_records = normalize(list(source_records))
        if supplied_records != canonical_records:
            raise RuntimeError(
                "Refusing valuation source claims that do not exactly match "
                "artifact-embedded writer results"
            )
    return canonical_records


def _validate_autonomous_product_artifact(
    artifact: AutonomousRunArtifact | AutonomousSectorFinancialRunArtifact,
) -> None:
    if (
        isinstance(artifact, AutonomousSectorFinancialRunArtifact)
        and artifact.pipeline_version == "v2"
    ):
        artifact._validate_v2()
    status = str(getattr(artifact, "status", "") or "")
    if status != PRODUCT_SUCCESS_STATUS:
        raise RuntimeError(
            f"Refusing to persist autonomous research artifact with non-success status {status!r}."
        )
    if (
        isinstance(artifact, AutonomousSectorFinancialRunArtifact)
        and artifact.pipeline_version == "v2"
        and artifact.decision_status != "COMPLETE"
    ):
        raise RuntimeError(
            "Refusing to persist an incomplete v2 sector decision as a product artifact."
        )
    if _successful_artifact_tool_call_count(artifact) <= 0:
        run_id = getattr(getattr(artifact, "request", None), "run_id", None) or getattr(
            artifact, "run_id", "unknown"
        )
        raise RuntimeError(
            "Refusing to persist autonomous research artifact without successful tool calls "
            f"for run {run_id}."
        )


def persist_autonomous_run(artifact: AutonomousRunArtifact) -> AutonomousRunPaths:
    """Write the v1 autonomous run JSON and Markdown report under data/outputs/runs."""

    _validate_autonomous_product_artifact(artifact)
    _require_publication_financial_integrity_binding(
        artifact.request.candidate_scope.get("financial_integrity_binding")
    )
    cfg = get_config()
    ensure_directories(cfg)
    run_dir = cfg.runs_dir / "autonomous" / artifact.request.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = run_dir / "autonomous_run.json"
    report_path = run_dir / "autonomous_research_report.md"
    artifact_path.write_text(
        json.dumps(artifact.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    report_path.write_text(render_autonomous_run_report(artifact), encoding="utf-8")
    return AutonomousRunPaths(artifact_json=artifact_path, report_md=report_path)


def persist_autonomous_sector_run(
    artifact: AutonomousSectorFinancialRunArtifact,
    *,
    valuation_source_records: Sequence[Mapping[str, Any]] | None = None,
) -> AutonomousSectorRunPaths:
    """Write the autonomous sector financial run JSON artifact under data/outputs/runs."""

    _validate_autonomous_product_artifact(artifact)
    if artifact.pipeline_version == "v1":
        _require_publication_financial_integrity_binding(
            artifact.candidate_selection.get("financial_integrity_binding"),
            packets=artifact.company_packets,
            scenarios=artifact.expected_return_scenarios,
        )
    cfg = get_config()
    authorized_tickers = {
        str(packet.ticker).strip().upper()
        for packet in artifact.company_packets
        if str(packet.ticker).strip()
        and str(packet.financial_integrity_status or "").strip().upper() == FINANCIAL_INTEGRITY_PASS
    }
    artifact_payload = artifact.to_dict()
    artifact_payload["valuation_source_records"] = _controlled_valuation_source_records(
        artifact=artifact,
        run_id=artifact.run_id,
        as_of_date=artifact.as_of_date,
        tickers=authorized_tickers,
        source_records=valuation_source_records,
    )
    integrity_status = artifact_decision_eligibility(artifact_payload)
    if integrity_status != FINANCIAL_INTEGRITY_PASS:
        raise RuntimeError(
            "Refusing to publish autonomous sector product artifact: "
            f"INVALID_FINANCIAL_INPUT ({integrity_status}) for run {artifact.run_id}."
        )
    ensure_directories(cfg)
    sector_runs_root = cfg.runs_dir / "autonomous_sector"
    sector_runs_root.mkdir(parents=True, exist_ok=True)
    run_dir = sector_runs_root / artifact.run_id
    if run_dir.exists():
        raise RuntimeError(
            "Refusing to replace an existing autonomous-sector product directory "
            f"for run {artifact.run_id}."
        )
    staging_container = sector_runs_root / f".{artifact.run_id}.{uuid4().hex}.staging"
    staging_run_dir = staging_container / artifact.run_id
    staging_run_dir.mkdir(parents=True)
    artifact_path = staging_run_dir / "autonomous_sector_run.json"
    report_path = staging_run_dir / "autonomous_sector_report.md"
    artifact_text = json.dumps(artifact_payload, indent=2, sort_keys=True)
    report_text = render_autonomous_sector_report(artifact)
    published = False
    try:
        artifact_path.write_text(artifact_text, encoding="utf-8")
        report_path.write_text(report_text, encoding="utf-8")
        write_run_financial_authorization(
            artifact_path,
            report_path,
            published_parent=run_dir,
        )
        staging_run_dir.replace(run_dir)
        published = True
        bind_authorized_valuation_rows(
            run_id=artifact.run_id,
            tickers=authorized_tickers,
            as_of_date=artifact.as_of_date,
            cfg=cfg,
        )
    except BaseException:
        if published and run_dir.exists() and not staging_run_dir.exists():
            run_dir.replace(staging_run_dir)
        shutil.rmtree(staging_container, ignore_errors=True)
        raise
    try:
        staging_container.rmdir()
    except OSError:
        pass
    return AutonomousSectorRunPaths(
        artifact_json=run_dir / artifact_path.name,
        report_md=run_dir / report_path.name,
        authorization_json=run_dir / "financial_integrity_authorization.json",
    )


def persist_autonomous_sector_diagnostic(
    artifact: AutonomousSectorFinancialRunArtifact,
) -> AutonomousSectorDiagnosticPaths:
    """Persist an interrupted or incomplete sector attempt outside product outputs."""

    if artifact.pipeline_version == "v2":
        artifact._validate_v2()
    execution_status = str(artifact.execution_status or artifact.status or "")
    is_incomplete = artifact.decision_status == "INCOMPLETE"
    if execution_status == PRODUCT_SUCCESS_STATUS and not is_incomplete:
        raise RuntimeError(
            "Refusing to persist a completed sector decision as a diagnostic artifact."
        )
    cfg = get_config()
    ensure_directories(cfg)
    run_dir = cfg.runs_dir / "autonomous_sector_diagnostics" / artifact.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = run_dir / "autonomous_sector_diagnostic.json"
    artifact_path.write_text(
        json.dumps(artifact.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return AutonomousSectorDiagnosticPaths(artifact_json=artifact_path)


def persist_autonomous_sector_attempt_diagnostic(
    *,
    sector: str,
    market_cap_focus: str,
    objective: str,
    as_of_date: str | None,
    pipeline_version: str,
    candidate_selection: dict | None,
    error: BaseException,
    last_completed_stage: str = "CANDIDATE_SELECTION",
    artifact_snapshot: dict | None = None,
) -> AutonomousSectorDiagnosticPaths:
    """Persist a v2 exception, optionally retaining the last valid artifact."""

    normalized_pipeline = str(pipeline_version or "").strip().lower()
    if normalized_pipeline != "v2":
        raise RuntimeError("Attempt diagnostics are reserved for v2 execution failures.")
    created_at = datetime.now(timezone.utc)
    sector_token = (
        "".join(char if char.isalnum() else "_" for char in str(sector or "sector").lower()).strip(
            "_"
        )
        or "sector"
    )
    run_id = (
        f"autonomous_sector_attempt_{sector_token}_"
        f"{created_at.strftime('%Y%m%dT%H%M%S')}_{uuid4().hex[:8]}"
    )
    selection = dict(candidate_selection or {})
    admitted = list(
        dict.fromkeys(
            str(ticker).strip().upper()
            for ticker in (
                selection.get("loaded_tickers") or selection.get("selected_tickers") or []
            )
            if str(ticker).strip()
        )
    )
    discovered = list(admitted)
    for field_name in ("requested_tickers", "excluded_tickers"):
        for ticker in selection.get(field_name) or []:
            normalized = str(ticker).strip().upper()
            if normalized and normalized not in discovered:
                discovered.append(normalized)
    for ticker in selection.get("cap_classifications") or {}:
        normalized = str(ticker).strip().upper()
        if normalized and normalized not in discovered:
            discovered.append(normalized)
    admitted_set = set(admitted)
    stage = str(last_completed_stage or "CANDIDATE_SELECTION").strip().upper()
    failure_reason = (
        "RUNTIME_EXCEPTION_BEFORE_ARTIFACT"
        if artifact_snapshot is None and stage == "CANDIDATE_SELECTION"
        else f"RUNTIME_EXCEPTION_AFTER_{stage}"
    )
    payload = {
        "artifact_type": "autonomous_sector_attempt_diagnostic_v2",
        "contract_version": "autonomous_sector_financial_run_v2",
        "pipeline_version": "v2",
        "run_id": run_id,
        "sector": sector,
        "market_cap_focus": market_cap_focus,
        "objective": objective,
        "as_of_date": as_of_date or date.today().isoformat(),
        "created_at": created_at.isoformat(),
        "execution_status": "FAILED",
        "decision_status": "INCOMPLETE",
        "final_verdict": None,
        "selected_ticker": None,
        "admitted_tickers": admitted,
        "candidate_dispositions": [
            {
                "ticker": ticker,
                "terminal_state": ("NEEDS_DATA" if ticker in admitted_set else "OUT_OF_SCOPE"),
                "scope_status": ("IN_SCOPE" if ticker in admitted_set else "OUT_OF_SCOPE"),
                "screen_status": "INCOMPLETE" if ticker in admitted_set else "NOT_RUN",
                "review_status": "FAILED" if ticker in admitted_set else "NOT_REQUIRED",
                "reason_codes": [
                    failure_reason if ticker in admitted_set else "NOT_ADMITTED_BEFORE_EXCEPTION"
                ],
                "last_completed_stage": stage,
            }
            for ticker in discovered
        ],
        "candidate_selection": selection,
        "last_completed_stage": stage,
        "artifact_snapshot": dict(artifact_snapshot or {}),
        "error": {
            "type": type(error).__name__,
            "message": str(error),
        },
    }
    cfg = get_config()
    ensure_directories(cfg)
    run_dir = cfg.runs_dir / "autonomous_sector_diagnostics" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = run_dir / "autonomous_sector_attempt_diagnostic.json"
    artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return AutonomousSectorDiagnosticPaths(artifact_json=artifact_path)


__all__ = [
    "AutonomousRunPaths",
    "AutonomousSectorDiagnosticPaths",
    "AutonomousSectorRunPaths",
    "persist_autonomous_run",
    "persist_autonomous_sector_attempt_diagnostic",
    "persist_autonomous_sector_diagnostic",
    "persist_autonomous_sector_run",
]
