"""Exact financial-scope publication for Alpha JSON/Markdown products."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

from app.alpha.report_writer import (
    AlphaFinancialUnitError,
    derive_alpha_report_financials,
    load_alpha_report_financial_sources,
    render_alpha_report,
)
from app.alpha.schemas import ComparisonRound, SectorAlphaReport, TickerSignalPacket
from app.autonomous.financial_integrity import (
    INVALID_FINANCIAL_INPUT,
    NEEDS_DATA,
    FinancialIntegrityGateResult,
    FinancialIntegrityScope,
    FinancialIntegrityViolation,
    InvalidFinancialInputError,
    require_unchanged_financial_integrity_scope,
    validate_financial_integrity_scope,
)
from app.autonomous.v1_financial_context import (
    BoundV1FinancialScope,
    bind_v1_financial_scope,
    financial_input_scenario,
)
from app.config import get_config
from app.db import connect


ALPHA_PUBLICATION_SCHEMA_VERSION = "alpha_financial_publication_v1"
ALPHA_AUTHORIZATION_SCHEMA_VERSION = "alpha_financial_publication_authorization_v1"
ALPHA_AUTHORIZATION_SUFFIX = ".financial_integrity_authorization.json"
_SUPPLEMENTAL_SCHEMA_VERSION = "alpha_report_supplemental_financials_v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json_bytes(value: Any, *, indent: int | None = None) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":") if indent is None else None,
            indent=indent,
            allow_nan=False,
        )
        + ("\n" if indent is not None else "")
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _packet_mapping(
    packets: Mapping[str, TickerSignalPacket] | Sequence[TickerSignalPacket],
) -> dict[str, TickerSignalPacket]:
    values = packets.values() if isinstance(packets, Mapping) else packets
    return {
        str(packet.ticker).strip().upper(): packet
        for packet in values
        if str(packet.ticker).strip()
    }


def _raise_report_packet_mismatch(
    *,
    report: SectorAlphaReport,
    run_as_of_date: str,
    ticker: str,
    field_name: str,
    expected: Any,
    observed: Any,
) -> None:
    raise InvalidFinancialInputError(
        FinancialIntegrityGateResult(
            context=f"alpha_publication:{report.sector}",
            run_as_of_date=run_as_of_date,
            status=INVALID_FINANCIAL_INPUT,
            violations=(
                FinancialIntegrityViolation(
                    code="ALPHA_REPORT_PACKET_MISMATCH",
                    ticker=ticker,
                    field=f"signal_packets.{ticker}.{field_name}",
                    source_values={
                        "canonical_packet_value": expected,
                        "report_value": observed,
                    },
                    expected_relationship="report financial field equals canonical packet",
                    observed_relationship=f"{observed!r} != {expected!r}",
                    reason=(
                        "Alpha report financial fields must come from the exact "
                        "authorized packet scope."
                    ),
                ),
            ),
        )
    )


def _raise_supplemental_unit_error(
    *,
    report: SectorAlphaReport,
    run_as_of_date: str,
    error: AlphaFinancialUnitError,
) -> None:
    row = error.row
    ticker = str(row.get("ticker") or "").strip().upper() or None
    line_item = str(row.get("line_item") or "").strip() or None
    raise InvalidFinancialInputError(
        FinancialIntegrityGateResult(
            context=f"alpha_publication:{report.sector}",
            run_as_of_date=run_as_of_date,
            status=NEEDS_DATA,
            violations=(
                FinancialIntegrityViolation(
                    code="ALPHA_REPORT_FINANCIAL_UNIT_INVALID",
                    ticker=ticker,
                    field=(
                        f"supplemental_companyfacts.{ticker}.{line_item}.units"
                        if ticker and line_item
                        else "supplemental_companyfacts.units"
                    ),
                    source_values={
                        "value": row.get("value"),
                        "units": row.get("units"),
                        "period_end": row.get("period_end"),
                        "filed_date": row.get("filed_date"),
                        "accession": row.get("accession"),
                        "source_url": row.get("source_url"),
                    },
                    expected_relationship="units == 'USD_millions'",
                    observed_relationship=f"units={row.get('units')!r}",
                    reason=(
                        "Alpha report columns labeled $M require an exact "
                        "normalized USD_millions source row."
                    ),
                    terminal_status=NEEDS_DATA,
                ),
            ),
        )
    ) from error


def _bind_report_signal_packets(
    *,
    report: SectorAlphaReport,
    packet_by_ticker: Mapping[str, TickerSignalPacket],
    run_as_of_date: str,
) -> None:
    """Reject financial drift and retain only canonical packet financial fields."""

    extras = sorted(set(report.signal_packets) - set(packet_by_ticker))
    if extras:
        _raise_report_packet_mismatch(
            report=report,
            run_as_of_date=run_as_of_date,
            ticker=extras[0],
            field_name="<ticker>",
            expected="ticker present in canonical packet scope",
            observed="ticker absent from canonical packet scope",
        )
    packet_fields = {item.name for item in fields(TickerSignalPacket)}
    bound_packets: dict[str, dict[str, Any]] = {}
    for ticker, packet in packet_by_ticker.items():
        existing = dict(report.signal_packets.get(ticker, {}))
        canonical = packet.to_summary_dict()
        raw_packet = asdict(packet)
        for field_name in sorted(packet_fields & set(existing)):
            if existing[field_name] != raw_packet[field_name]:
                _raise_report_packet_mismatch(
                    report=report,
                    run_as_of_date=run_as_of_date,
                    ticker=ticker,
                    field_name=field_name,
                    expected=raw_packet[field_name],
                    observed=existing[field_name],
                )
        preserved = {key: value for key, value in existing.items() if key not in packet_fields}
        preserved.update(canonical)
        bound_packets[ticker] = preserved
    report.signal_packets = bound_packets


def _report_years_by_ticker(report: SectorAlphaReport) -> dict[str, int]:
    years: dict[str, int] = {}
    if report.winner:
        years[str(report.winner).strip().upper()] = 5
    if report.runner_up:
        ticker = str(report.runner_up).strip().upper()
        years[ticker] = max(years.get(ticker, 0), 3)
    return years


def build_alpha_supplemental_scenarios(
    *,
    report: SectorAlphaReport,
    packets: Mapping[str, TickerSignalPacket] | Sequence[TickerSignalPacket],
    source_rows_by_ticker: Mapping[str, list[dict[str, Any]]],
    run_as_of_date: str,
) -> tuple[dict[str, Any], ...]:
    """Bind the exact CompanyFacts rows and derived values rendered in Markdown."""

    packet_by_ticker = _packet_mapping(packets)
    scenarios: list[dict[str, Any]] = []
    for ticker, years in sorted(_report_years_by_ticker(report).items()):
        packet = packet_by_ticker.get(ticker)
        if packet is None:
            raise ValueError(
                f"Alpha report financial publication has no canonical packet for {ticker}"
            )
        rows = list(source_rows_by_ticker.get(ticker, []))
        try:
            rendered_financials = derive_alpha_report_financials(
                rows,
                years=years,
            )
        except AlphaFinancialUnitError as exc:
            _raise_supplemental_unit_error(
                report=report,
                run_as_of_date=run_as_of_date,
                error=exc,
            )
        scenarios.append(
            financial_input_scenario(
                packet,
                financial_inputs={
                    "schema_version": _SUPPLEMENTAL_SCHEMA_VERSION,
                    "report_as_of_date": run_as_of_date,
                    "companyfacts_rows": rows,
                    "rendered_financials": rendered_financials,
                },
            )
        )
    return tuple(scenarios)


@dataclass(frozen=True)
class AlphaPublicationContext:
    report: SectorAlphaReport
    packets: tuple[TickerSignalPacket, ...]
    run_as_of_date: str
    bound_decision_scope: BoundV1FinancialScope
    bound_scope: BoundV1FinancialScope
    source_rows_by_ticker: dict[str, list[dict[str, Any]]]

    def binding(self, *, generated_at: str) -> dict[str, Any]:
        return {
            "schema_version": ALPHA_PUBLICATION_SCHEMA_VERSION,
            "status": "PASS",
            "generated_at": generated_at,
            "decision_scope_fingerprint": (self.bound_decision_scope.expected_scope_fingerprint),
            "publication_scope_fingerprint": self.bound_scope.expected_scope_fingerprint,
            "decision_scope_manifest": {
                "context": self.bound_decision_scope.context,
                "run_as_of_date": self.bound_decision_scope.run_as_of_date,
                "packets": list(self.bound_decision_scope.packets),
                "scenarios": list(self.bound_decision_scope.scenarios),
            },
            "scope_manifest": {
                "context": self.bound_scope.context,
                "run_as_of_date": self.bound_scope.run_as_of_date,
                "packets": list(self.bound_scope.packets),
                "scenarios": list(self.bound_scope.scenarios),
            },
        }


@dataclass(frozen=True)
class AlphaPublicationPaths:
    report_json: Path
    report_markdown: Path
    authorization: Path


@dataclass(frozen=True)
class _FileSnapshot:
    path: Path
    data: bytes
    sha256: str
    device: int
    inode: int


def _read_exact_file_snapshot(path: str | Path) -> _FileSnapshot | None:
    """Read one regular, non-aliased path and bind bytes to its open file."""

    try:
        expanded = Path(path).expanduser()
        lexical = Path(os.path.abspath(os.fspath(expanded)))
        if lexical != expanded.resolve():
            return None
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(lexical, flags)
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            return None
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    except OSError:
        return None
    finally:
        os.close(descriptor)
    data = b"".join(chunks)
    if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino) or after.st_size != len(data):
        return None
    return _FileSnapshot(
        path=lexical,
        data=data,
        sha256=_sha256(data),
        device=after.st_dev,
        inode=after.st_ino,
    )


def _file_snapshot_is_current(snapshot: _FileSnapshot) -> bool:
    current = _read_exact_file_snapshot(snapshot.path)
    return bool(
        current is not None
        and (current.device, current.inode) == (snapshot.device, snapshot.inode)
        and current.sha256 == snapshot.sha256
        and current.data == snapshot.data
    )


def _raise_decision_publication_scope_mismatch(
    *,
    report: SectorAlphaReport,
    run_as_of_date: str,
    decision_scope: BoundV1FinancialScope,
    publication_scope: BoundV1FinancialScope,
) -> None:
    raise InvalidFinancialInputError(
        FinancialIntegrityGateResult(
            context=f"alpha_publication:{report.sector}",
            run_as_of_date=run_as_of_date,
            status=INVALID_FINANCIAL_INPUT,
            violations=(
                FinancialIntegrityViolation(
                    code="ALPHA_DECISION_PUBLICATION_SCOPE_MISMATCH",
                    ticker=None,
                    field="decision_scope.packets",
                    source_values={
                        "decision_scope_fingerprint": (decision_scope.expected_scope_fingerprint),
                        "publication_scope_fingerprint": (
                            publication_scope.expected_scope_fingerprint
                        ),
                        "decision_tickers": [
                            item.get("ticker")
                            for item in decision_scope.packets
                            if isinstance(item, Mapping)
                        ],
                        "publication_tickers": [
                            item.get("ticker")
                            for item in publication_scope.packets
                            if isinstance(item, Mapping)
                        ],
                    },
                    expected_relationship=(
                        "decision and publication scopes contain identical canonical packets"
                    ),
                    observed_relationship="canonical packet payloads differ",
                    reason=(
                        "Alpha publication must be derived from the exact packet "
                        "scope that authorized the selection decision."
                    ),
                ),
            ),
        )
    )


def prepare_alpha_publication(
    *,
    report: SectorAlphaReport,
    packets: Mapping[str, TickerSignalPacket] | Sequence[TickerSignalPacket],
    run_as_of_date: str,
    decision_scope: FinancialIntegrityScope,
) -> AlphaPublicationContext:
    """Freeze the report's supplemental PIT rows into an exact financial scope."""

    packet_by_ticker = _packet_mapping(packets)
    _bind_report_signal_packets(
        report=report,
        packet_by_ticker=packet_by_ticker,
        run_as_of_date=run_as_of_date,
    )
    packet_values = tuple(packet_by_ticker.values())
    try:
        source_rows = load_alpha_report_financial_sources(
            report,
            as_of_date=run_as_of_date,
        )
    except AlphaFinancialUnitError as exc:
        _raise_supplemental_unit_error(
            report=report,
            run_as_of_date=run_as_of_date,
            error=exc,
        )
    # Detach SQLite values and fail closed on NaN/unsupported financial data.
    source_rows = json.loads(_canonical_json_bytes(source_rows).decode("utf-8"))
    scenarios = build_alpha_supplemental_scenarios(
        report=report,
        packets=packet_values,
        source_rows_by_ticker=source_rows,
        run_as_of_date=run_as_of_date,
    )
    bound_scope = bind_v1_financial_scope(
        context=f"alpha_publication:{report.sector}",
        run_as_of_date=run_as_of_date,
        packets=packet_values,
        scenarios=scenarios,
    )
    bound_decision_scope = bind_v1_financial_scope(
        context=decision_scope.context,
        run_as_of_date=decision_scope.run_as_of_date,
        packets=decision_scope.packets,
        scenarios=decision_scope.scenarios,
    )
    if (
        bound_decision_scope.context != f"alpha_scan:{report.sector}"
        or bound_decision_scope.run_as_of_date != run_as_of_date
        or bound_decision_scope.packets != bound_scope.packets
    ):
        _raise_decision_publication_scope_mismatch(
            report=report,
            run_as_of_date=run_as_of_date,
            decision_scope=bound_decision_scope,
            publication_scope=bound_scope,
        )
    return AlphaPublicationContext(
        report=report,
        packets=packet_values,
        run_as_of_date=run_as_of_date,
        bound_decision_scope=bound_decision_scope,
        bound_scope=bound_scope,
        source_rows_by_ticker=source_rows,
    )


def _write_temporary_bytes(destination: Path, data: bytes) -> Path:
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    with temporary.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    return temporary


def _authorization_path(report_json: Path) -> Path:
    return report_json.with_name(f"{report_json.stem}{ALPHA_AUTHORIZATION_SUFFIX}")


def _scope_binding_is_valid(binding: Any) -> bool:
    if not isinstance(binding, Mapping):
        return False
    expected_fields = {
        "schema_version",
        "status",
        "generated_at",
        "decision_scope_fingerprint",
        "publication_scope_fingerprint",
        "report_markdown_sha256",
        "decision_scope_manifest",
        "scope_manifest",
    }
    if (
        set(binding) != expected_fields
        or binding.get("schema_version") != ALPHA_PUBLICATION_SCHEMA_VERSION
        or binding.get("status") != "PASS"
        or not isinstance(binding.get("generated_at"), str)
        or _SHA256_RE.fullmatch(str(binding.get("decision_scope_fingerprint") or "")) is None
        or _SHA256_RE.fullmatch(str(binding.get("publication_scope_fingerprint") or "")) is None
        or _SHA256_RE.fullmatch(str(binding.get("report_markdown_sha256") or "")) is None
    ):
        return False
    decision_manifest = binding.get("decision_scope_manifest")
    manifest = binding.get("scope_manifest")
    required_manifest_fields = {
        "context",
        "run_as_of_date",
        "packets",
        "scenarios",
    }
    if (
        not isinstance(decision_manifest, Mapping)
        or set(decision_manifest) != required_manifest_fields
        or not isinstance(manifest, Mapping)
        or set(manifest) != required_manifest_fields
    ):
        return False
    decision_packets = decision_manifest.get("packets")
    decision_scenarios = decision_manifest.get("scenarios")
    packets = manifest.get("packets")
    scenarios = manifest.get("scenarios")
    if not (
        isinstance(decision_packets, list)
        and isinstance(decision_scenarios, list)
        and isinstance(packets, list)
        and isinstance(scenarios, list)
        and decision_packets == packets
        and decision_manifest.get("run_as_of_date") == manifest.get("run_as_of_date")
    ):
        return False
    decision_result = validate_financial_integrity_scope(
        FinancialIntegrityScope(
            context=str(decision_manifest.get("context") or ""),
            run_as_of_date=str(decision_manifest.get("run_as_of_date") or ""),
            packets=tuple(decision_packets),
            scenarios=tuple(decision_scenarios),
        )
    )
    result = validate_financial_integrity_scope(
        FinancialIntegrityScope(
            context=str(manifest.get("context") or ""),
            run_as_of_date=str(manifest.get("run_as_of_date") or ""),
            packets=tuple(packets),
            scenarios=tuple(scenarios),
        )
    )
    return bool(
        decision_result.passed
        and decision_result.scope_fingerprint == binding.get("decision_scope_fingerprint")
        and result.passed
        and result.scope_fingerprint == binding.get("publication_scope_fingerprint")
    )


def _serialized_report_packets_match_scope(
    report_payload: Mapping[str, Any],
    packet_payloads: list[Any],
) -> bool:
    signal_packets = report_payload.get("signal_packets")
    if not isinstance(signal_packets, Mapping):
        return False
    packet_fields = {item.name for item in fields(TickerSignalPacket)}
    canonical_by_ticker: dict[str, dict[str, Any]] = {}
    for raw_packet in packet_payloads:
        if not isinstance(raw_packet, Mapping):
            return False
        try:
            packet = TickerSignalPacket(**dict(raw_packet))
        except TypeError:
            return False
        ticker = str(packet.ticker).strip().upper()
        if not ticker or ticker in canonical_by_ticker:
            return False
        canonical_by_ticker[ticker] = packet.to_summary_dict()
    if set(signal_packets) != set(canonical_by_ticker):
        return False
    for ticker, canonical in canonical_by_ticker.items():
        observed = signal_packets.get(ticker)
        if not isinstance(observed, Mapping):
            return False
        observed_packet_fields = {
            key: value for key, value in observed.items() if key in packet_fields
        }
        if observed_packet_fields != canonical:
            return False
    return True


def _serialized_report_is_authorizable(
    report_payload: Mapping[str, Any],
    report_markdown: bytes,
) -> bool:
    binding = report_payload.get("financial_integrity_binding")
    if not _scope_binding_is_valid(binding):
        return False
    manifest = binding["scope_manifest"]
    if not _serialized_report_packets_match_scope(
        report_payload,
        manifest["packets"],
    ):
        return False
    rounds = report_payload.get("rounds")
    if not isinstance(rounds, list):
        return False
    try:
        prepared = dict(report_payload)
        prepared["rounds"] = [
            ComparisonRound(**dict(item)) for item in rounds if isinstance(item, Mapping)
        ]
        if len(prepared["rounds"]) != len(rounds):
            return False
        report = SectorAlphaReport(**prepared)
        generated_at = datetime.fromisoformat(str(binding["generated_at"]).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return False

    years_by_ticker = _report_years_by_ticker(report)
    source_rows: dict[str, list[dict[str, Any]]] = {}
    for scenario in manifest["scenarios"]:
        if not isinstance(scenario, Mapping):
            return False
        ticker = str(scenario.get("ticker") or "").strip().upper()
        financial_inputs = scenario.get("financial_inputs")
        if (
            ticker not in years_by_ticker
            or ticker in source_rows
            or not isinstance(financial_inputs, Mapping)
            or financial_inputs.get("schema_version") != _SUPPLEMENTAL_SCHEMA_VERSION
            or financial_inputs.get("report_as_of_date") != manifest["run_as_of_date"]
            or not isinstance(financial_inputs.get("companyfacts_rows"), list)
            or not isinstance(financial_inputs.get("rendered_financials"), list)
        ):
            return False
        rows = financial_inputs["companyfacts_rows"]
        try:
            rendered_financials = derive_alpha_report_financials(
                rows,
                years=years_by_ticker[ticker],
            )
        except AlphaFinancialUnitError:
            return False
        if rendered_financials != financial_inputs["rendered_financials"]:
            return False
        source_rows[ticker] = rows
    if set(source_rows) != set(years_by_ticker):
        return False
    expected_markdown = render_alpha_report(
        report,
        as_of_date=str(manifest["run_as_of_date"]),
        financial_source_rows=source_rows,
        generated_at=generated_at,
    ).encode("utf-8")
    return expected_markdown == report_markdown and binding["report_markdown_sha256"] == _sha256(
        report_markdown
    )


def authorized_alpha_publication_bytes(
    authorization_path: str | Path,
) -> dict[str, bytes] | None:
    """Return exact validated Alpha bytes, or ``None`` when authorization fails."""

    raw_authorization = Path(authorization_path).expanduser()
    authorization = raw_authorization.resolve()
    if raw_authorization.absolute() != authorization or not authorization.name.endswith(
        ALPHA_AUTHORIZATION_SUFFIX
    ):
        return None
    authorization_snapshot = _read_exact_file_snapshot(authorization)
    if authorization_snapshot is None:
        return None
    try:
        authorization_payload = json.loads(authorization_snapshot.data.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(authorization_payload, dict):
        return None
    if (
        authorization_payload.get("schema_version") != ALPHA_AUTHORIZATION_SCHEMA_VERSION
        or authorization_payload.get("complete") is not True
        or not isinstance(authorization_payload.get("generated_at"), str)
        or _SHA256_RE.fullmatch(str(authorization_payload.get("decision_scope_fingerprint") or ""))
        is None
        or _SHA256_RE.fullmatch(str(authorization_payload.get("financial_scope_fingerprint") or ""))
        is None
    ):
        return None
    records = authorization_payload.get("artifacts")
    if not isinstance(records, list) or len(records) != 2:
        return None
    sector = str(authorization_payload.get("sector") or "")
    if (
        not sector
        or Path(sector).name != sector
        or authorization.name != f"alpha_{sector}{ALPHA_AUTHORIZATION_SUFFIX}"
    ):
        return None
    artifacts: dict[str, bytes] = {}
    artifact_names: dict[str, str] = {}
    artifact_snapshots: dict[str, _FileSnapshot] = {}
    for record in records:
        if not isinstance(record, Mapping):
            return None
        kind = str(record.get("kind") or "")
        raw_path = str(record.get("path") or "")
        sha256 = str(record.get("sha256") or "")
        path = Path(raw_path).expanduser().resolve()
        snapshot = _read_exact_file_snapshot(path)
        if (
            kind not in {"artifact_json", "report_markdown"}
            or kind in artifacts
            or raw_path != str(path)
            or path.parent != authorization.parent
            or _SHA256_RE.fullmatch(sha256) is None
            or snapshot is None
        ):
            return None
        if snapshot.sha256 != sha256:
            return None
        artifacts[kind] = snapshot.data
        artifact_names[kind] = path.name
        artifact_snapshots[kind] = snapshot
    try:
        report_payload = json.loads(artifacts["artifact_json"].decode("utf-8"))
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(report_payload, dict):
        return None
    binding = report_payload.get("financial_integrity_binding")
    manifest = binding.get("scope_manifest") if isinstance(binding, Mapping) else None
    decision_manifest = (
        binding.get("decision_scope_manifest") if isinstance(binding, Mapping) else None
    )
    valid = bool(
        report_payload.get("sector") == sector
        and artifact_names.get("artifact_json") == f"alpha_{sector}.json"
        and artifact_names.get("report_markdown") == f"alpha_{sector}_report.md"
        and _serialized_report_is_authorizable(
            report_payload,
            artifacts["report_markdown"],
        )
        and isinstance(manifest, Mapping)
        and isinstance(decision_manifest, Mapping)
        and decision_manifest.get("context") == f"alpha_scan:{sector}"
        and decision_manifest.get("packets") == manifest.get("packets")
        and manifest.get("context") == f"alpha_publication:{sector}"
        and binding["generated_at"] == authorization_payload["generated_at"]
        and binding["decision_scope_fingerprint"]
        == authorization_payload["decision_scope_fingerprint"]
        and binding["publication_scope_fingerprint"]
        == authorization_payload["financial_scope_fingerprint"]
        and binding["report_markdown_sha256"] == _sha256(artifacts["report_markdown"])
    )
    snapshots = (authorization_snapshot, *artifact_snapshots.values())
    if not valid or not all(_file_snapshot_is_current(snapshot) for snapshot in snapshots):
        return None
    return artifacts


def alpha_publication_is_authorized(
    authorization_path: str | Path,
) -> bool:
    """Validate one Alpha commit marker against both exact current artifacts."""

    return authorized_alpha_publication_bytes(authorization_path) is not None


def publish_alpha_report(
    *,
    context: AlphaPublicationContext,
    decision_scope: FinancialIntegrityScope,
    output_dir: Path,
) -> AlphaPublicationPaths:
    """Publish and authorize exact Alpha JSON/Markdown bytes under a DB lock."""

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sector = str(context.report.sector)
    if (
        not sector
        or Path(sector).name != sector
        or decision_scope.run_as_of_date != context.run_as_of_date
    ):
        raise ValueError("Alpha publication identity or decision scope is malformed")
    _bind_report_signal_packets(
        report=context.report,
        packet_by_ticker=_packet_mapping(context.packets),
        run_as_of_date=context.run_as_of_date,
    )
    context.bound_scope.require(
        scenarios=build_alpha_supplemental_scenarios(
            report=context.report,
            packets=context.packets,
            source_rows_by_ticker=context.source_rows_by_ticker,
            run_as_of_date=context.run_as_of_date,
        )
    )
    report_json = output_dir / f"alpha_{sector}.json"
    report_markdown = output_dir / f"alpha_{sector}_report.md"
    authorization = _authorization_path(report_json)

    generated = datetime.now(timezone.utc)
    generated_text = generated.isoformat().replace("+00:00", "Z")
    binding = context.binding(generated_at=generated_text)
    context.report.financial_integrity_binding = binding
    try:
        markdown = render_alpha_report(
            context.report,
            as_of_date=context.run_as_of_date,
            financial_source_rows=context.source_rows_by_ticker,
            generated_at=generated,
        )
    except AlphaFinancialUnitError as exc:
        _raise_supplemental_unit_error(
            report=context.report,
            run_as_of_date=context.run_as_of_date,
            error=exc,
        )
    _bind_report_signal_packets(
        report=context.report,
        packet_by_ticker=_packet_mapping(context.packets),
        run_as_of_date=context.run_as_of_date,
    )
    context.bound_scope.require(
        scenarios=build_alpha_supplemental_scenarios(
            report=context.report,
            packets=context.packets,
            source_rows_by_ticker=context.source_rows_by_ticker,
            run_as_of_date=context.run_as_of_date,
        )
    )
    markdown_bytes = markdown.encode("utf-8")
    binding["report_markdown_sha256"] = _sha256(markdown_bytes)
    report_payload = asdict(context.report)
    report_json_bytes = _canonical_json_bytes(report_payload, indent=2)

    authorization_payload = {
        "schema_version": ALPHA_AUTHORIZATION_SCHEMA_VERSION,
        "generated_at": generated_text,
        "complete": True,
        "sector": sector,
        "decision_scope_fingerprint": (context.bound_decision_scope.expected_scope_fingerprint),
        "financial_scope_fingerprint": context.bound_scope.expected_scope_fingerprint,
        "artifacts": [
            {
                "kind": "artifact_json",
                "path": str(report_json),
                "sha256": _sha256(report_json_bytes),
            },
            {
                "kind": "report_markdown",
                "path": str(report_markdown),
                "sha256": _sha256(markdown_bytes),
            },
        ],
    }
    authorization_bytes = _canonical_json_bytes(authorization_payload, indent=2)
    staged_json = _write_temporary_bytes(report_json, report_json_bytes)
    staged_markdown = _write_temporary_bytes(report_markdown, markdown_bytes)
    staged_authorization = _write_temporary_bytes(authorization, authorization_bytes)
    staged = (staged_json, staged_markdown, staged_authorization)

    conn = connect(get_config().db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        current_sources = load_alpha_report_financial_sources(
            context.report,
            as_of_date=context.run_as_of_date,
            conn=conn,
        )
        current_scenarios = build_alpha_supplemental_scenarios(
            report=context.report,
            packets=context.packets,
            source_rows_by_ticker=current_sources,
            run_as_of_date=context.run_as_of_date,
        )
        require_unchanged_financial_integrity_scope(
            decision_scope,
            expected_scope_fingerprint=(context.bound_decision_scope.expected_scope_fingerprint),
        )
        context.bound_scope.require(scenarios=current_scenarios)

        staged_json.replace(report_json)
        staged_markdown.replace(report_markdown)
        staged_authorization.replace(authorization)
        authorized_bytes = authorized_alpha_publication_bytes(authorization)
        if (
            authorized_bytes is None
            or authorized_bytes.get("artifact_json") != report_json_bytes
            or authorized_bytes.get("report_markdown") != markdown_bytes
        ):
            authorization.unlink(missing_ok=True)
            raise RuntimeError("Alpha post-write authorization failed exact validation")
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()
        for temporary in staged:
            temporary.unlink(missing_ok=True)

    return AlphaPublicationPaths(
        report_json=report_json,
        report_markdown=report_markdown,
        authorization=authorization,
    )


__all__ = [
    "ALPHA_AUTHORIZATION_SCHEMA_VERSION",
    "ALPHA_AUTHORIZATION_SUFFIX",
    "ALPHA_PUBLICATION_SCHEMA_VERSION",
    "AlphaPublicationContext",
    "AlphaPublicationPaths",
    "alpha_publication_is_authorized",
    "authorized_alpha_publication_bytes",
    "build_alpha_supplemental_scenarios",
    "prepare_alpha_publication",
    "publish_alpha_report",
]
