"""Validated production bridge from an accepted census to the v2 scanner.

The bridge deliberately reads the accepted census tables in place.  It does
not copy issuer caps into ``market_caps`` or infer membership from the legacy
sector registry.  A cohort is returned only after the persisted run, its
artifacts, immutable inputs, semantic replay fingerprint, promotion lineage,
and admitted issuer/security contracts all agree.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.autonomous.cap_resolver import CapClassification, band_for_market_cap
from app.sector.canonical_taxonomy import (
    ACTIVE_SECTOR_LABELS,
    CANONICAL_SECTOR_TAXONOMY_VERSION,
    canonical_sector_taxonomy_manifest,
    resolve_canonical_sector,
)
from app.sector.external_industry_taxonomy import external_industry_taxonomy_manifest
from app.universe.us_equity_census import (
    CensusInputPaths,
    _census_policy_manifest,
    _semantic_replay_fingerprint,
    census_input_fingerprint,
)


ACCEPTED_ISSUER_DISPOSITION = "ADMITTED_LARGE_AND_MEGA"
ACCEPTED_SECURITY_DISPOSITION = "ADMITTED_PRIMARY_COMMON_EQUITY"
ACCEPTED_PRIMARY_SECURITY_TYPES = frozenset({"COMMON_EQUITY", "COMMON_EQUITY_EQUIVALENT"})
ACCEPTED_CENSUS_CAP_SOURCE = "accepted_census"
BRIDGE_REQUIRED_ACCEPTANCE_CHECKS = frozenset(
    {
        "actual_llm_calls_zero",
        "actual_llm_cost_zero",
        "admitted_issuer_primary_contract_complete",
        "all_admitted_are_us_domiciled_operating",
        "all_admitted_have_resolved_sector_contract",
        "all_current_us_operating_primary_issuers_have_cap_band_decision",
        "at_most_one_primary_security_per_issuer",
        "every_issuer_has_one_terminal_disposition",
        "every_security_has_one_terminal_disposition",
        "new_large_cap_scan_population_promoted",
        "provider_free_replay_verified",
        "zero_silent_unknown_cap",
        "zero_unreconciled_large_cap_source_rows",
        "zero_unresolved_eligible_large_market_cap_boundaries",
        "zero_unresolved_eligible_large_primary_securities",
        "zero_unresolved_eligible_large_sector_contracts",
    }
)


class AcceptedCensusError(RuntimeError):
    """Base error for a census that cannot safely feed production."""


class AcceptedCensusUnavailableError(AcceptedCensusError):
    """No accepted run exactly matches the requested fixed-as-of cohort."""


class AcceptedCensusDriftError(AcceptedCensusError):
    """A nominally accepted run no longer matches its immutable contract."""


@dataclass(frozen=True, slots=True)
class AcceptedCensusLineage:
    run_id: str
    as_of_date: str
    target_band: str
    input_fingerprint: str
    resume_fingerprint: str
    semantic_output_fingerprint: str
    source_manifest_sha256: str
    policy_manifest_sha256: str
    taxonomy_version: str
    taxonomy_hash: str
    external_industry_taxonomy_version: str
    external_industry_taxonomy_hash: str
    checkpoint_path: str
    cohort_fingerprint: str
    member_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AcceptedCensusMember:
    ticker: str
    issuer_key: str
    security_key: str
    cik: str
    legal_name: str
    exchange_name: str
    exchange_mic: str | None
    source_sector_label: str
    canonical_sector: str
    market_cap_usd: float
    market_cap_as_of_date: str
    market_cap_method: str
    market_cap_source_provider: str
    market_cap_source_url: str
    market_cap_retrieved_at: str
    market_cap_confidence: str
    cap_derivation_json: str
    identity_source_provider: str
    identity_source_url: str
    identity_as_of_date: str
    identity_retrieved_at: str
    identity_confidence: str
    listing_source_url: str
    listing_status_as_of_date: str
    listing_retrieved_at: str

    @property
    def market_cap_mm(self) -> float:
        return self.market_cap_usd / 1_000_000.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_cap_classification(self, *, lineage: AcceptedCensusLineage) -> CapClassification:
        """Expose the accepted issuer-cap evidence without losing provenance."""

        return CapClassification(
            ticker=self.ticker,
            as_of_date=lineage.as_of_date,
            market_cap_mm=self.market_cap_mm,
            cap_source=ACCEPTED_CENSUS_CAP_SOURCE,
            cap_band=band_for_market_cap(self.market_cap_mm),
            cap_effective_as_of_date=self.market_cap_as_of_date,
            cap_source_kind="ACCEPTED_CENSUS",
            cap_source_name=self.market_cap_source_provider,
            cap_source_url=self.market_cap_source_url,
            cap_confidence=self.market_cap_confidence,
            issuer_cik=self.cik,
            issuer_primary_ticker=self.ticker,
            issuer_listed_tickers=(self.ticker,),
            security_role="PRIMARY",
            is_secondary_class=False,
            is_adr=False,
            identity_source=self.identity_source_provider,
            identity_source_url=self.identity_source_url,
            identity_as_of_date=self.identity_as_of_date,
            identity_confidence=self.identity_confidence,
            detail="accepted_registry_first_census_issuer_cap",
            cap_retrieved_at=self.market_cap_retrieved_at,
            cap_method=self.market_cap_method,
            cap_derivation_json=self.cap_derivation_json,
            issuer_key=self.issuer_key,
            security_key=self.security_key,
            census_run_id=lineage.run_id,
            census_input_fingerprint=lineage.input_fingerprint,
            census_semantic_output_fingerprint=lineage.semantic_output_fingerprint,
            census_cohort_fingerprint=lineage.cohort_fingerprint,
        )


@dataclass(frozen=True, slots=True)
class AcceptedCensusCohort:
    lineage: AcceptedCensusLineage
    members: tuple[AcceptedCensusMember, ...]

    def members_for_sector(self, source_sector_label: str) -> tuple[AcceptedCensusMember, ...]:
        sector = str(source_sector_label or "").strip().lower()
        return tuple(member for member in self.members if member.source_sector_label == sector)

    def member_by_ticker(self) -> dict[str, AcceptedCensusMember]:
        return {member.ticker: member for member in self.members}

    def cap_classifications(self) -> dict[str, CapClassification]:
        return {
            member.ticker: member.to_cap_classification(lineage=self.lineage)
            for member in self.members
        }


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _read_json(path: Path, *, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AcceptedCensusDriftError(f"invalid {label} artifact at {path}: {exc}") from exc


def _artifact_root(checkpoint_path: str) -> tuple[Path, Path]:
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_absolute():
        checkpoint = Path.cwd() / checkpoint
    return checkpoint.resolve(), checkpoint.resolve().parent


def _policy_manifest_from_artifact(source_manifest: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    return {
        str(row.get("logical_role") or row.get("file_name") or ""): str(row.get("sha256") or "")
        for row in source_manifest
        if str(row.get("artifact_kind") or "") == "POLICY_CODE"
    }


def _current_policy_manifest() -> dict[str, str]:
    return {str(row["name"]): str(row["sha256"]) for row in _census_policy_manifest()}


def _snapshot_inputs(root: Path) -> tuple[CensusInputPaths, Mapping[str, Any]]:
    snapshots = root / "source_snapshots"
    registry = snapshots / "company_tickers_exchange.json"
    nasdaq = snapshots / "nasdaqlisted.txt"
    other = snapshots / "otherlisted.txt"
    required = (registry, nasdaq, other, snapshots / "database_inputs.json")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise AcceptedCensusDriftError(
            "accepted census immutable snapshots are missing: " + ", ".join(missing)
        )
    companies_pages = tuple(sorted((snapshots / "companiesmarketcap").glob("page_*.html")))
    stockanalysis = snapshots / "stockanalysis" / "screener.html"
    terminal = snapshots / "terminal_cap_evidence.json"
    submissions = snapshots / "sec_submissions"
    fixed_cohort = snapshots / "fixed_cohort"
    inputs = CensusInputPaths(
        sec_registry=registry,
        nasdaq_listed=nasdaq,
        other_listed=other,
        companiesmarketcap_pages=companies_pages,
        stockanalysis_html=stockanalysis if stockanalysis.is_file() else None,
        terminal_cap_evidence=terminal if terminal.is_file() else None,
        submissions_dir=submissions if submissions.is_dir() else None,
        fixed_cohort_dir=fixed_cohort if fixed_cohort.is_dir() else None,
    )
    database_inputs = _read_json(
        snapshots / "database_inputs.json", label="database input snapshot"
    )
    if not isinstance(database_inputs, Mapping):
        raise AcceptedCensusDriftError("database input snapshot must be a JSON object")
    return inputs, database_inputs


def _require_exact_artifacts(
    *,
    run: Mapping[str, Any],
    checkpoint_path: Path,
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    acceptance = _read_json(root / "acceptance_report.json", label="acceptance")
    coverage = _read_json(root / "coverage_summary.json", label="coverage")
    checkpoint = _read_json(checkpoint_path, label="checkpoint")
    artifact_manifest = _read_json(root / "source_manifest.json", label="source manifest")
    discrepancies = _read_json(root / "source_discrepancies.json", label="source discrepancy")
    if not isinstance(acceptance, dict) or not isinstance(coverage, dict):
        raise AcceptedCensusDriftError("acceptance and coverage artifacts must be JSON objects")
    if not isinstance(checkpoint, dict) or not isinstance(discrepancies, dict):
        raise AcceptedCensusDriftError("checkpoint and discrepancy artifacts must be JSON objects")
    if not isinstance(artifact_manifest, list) or not all(
        isinstance(row, dict) for row in artifact_manifest
    ):
        raise AcceptedCensusDriftError("source manifest must be a list of JSON objects")

    run_id = str(run["run_id"])
    as_of_date = str(run["as_of_date"])
    semantic = str(
        json.loads(str(run.get("status_detail_json") or "{}")).get("semantic_output_fingerprint")
        or ""
    )
    expected_pairs = (
        (acceptance.get("schema_version"), "US_EQUITY_CENSUS_ACCEPTANCE_V1", "acceptance schema"),
        (acceptance.get("run_id"), run_id, "acceptance run_id"),
        (acceptance.get("as_of_date"), as_of_date, "acceptance as_of_date"),
        (acceptance.get("status"), "PASSED", "acceptance status"),
        (acceptance.get("provider_free_replay_verified"), True, "acceptance replay"),
        (
            acceptance.get("semantic_output_fingerprint"),
            semantic,
            "acceptance semantic fingerprint",
        ),
        (coverage.get("schema_version"), "US_EQUITY_CENSUS_COVERAGE_V1", "coverage schema"),
        (coverage.get("run_id"), run_id, "coverage run_id"),
        (coverage.get("as_of_date"), as_of_date, "coverage as_of_date"),
        (coverage.get("target_band"), "large_and_mega", "coverage target band"),
        (coverage.get("acceptance_status"), "PASSED", "coverage status"),
        (coverage.get("provider_free_replay_verified"), True, "coverage replay"),
        (coverage.get("semantic_output_fingerprint"), semantic, "coverage semantic fingerprint"),
        (checkpoint.get("run_id"), run_id, "checkpoint run_id"),
        (checkpoint.get("as_of_date"), as_of_date, "checkpoint as_of_date"),
        (checkpoint.get("stage"), "TERMINAL_DISPOSITION", "checkpoint stage"),
        (checkpoint.get("status"), "COMPLETED", "checkpoint status"),
        (checkpoint.get("acceptance_status"), "PASSED", "checkpoint acceptance"),
    )
    mismatches = [label for actual, expected, label in expected_pairs if actual != expected]
    if mismatches:
        raise AcceptedCensusDriftError(
            "accepted census artifact mismatch: " + ", ".join(mismatches)
        )
    checks = acceptance.get("checks")
    required_checks = acceptance.get("required_checks")
    if not isinstance(checks, Mapping) or not isinstance(required_checks, list):
        raise AcceptedCensusDriftError("acceptance check contract is missing")
    failed_checks = [str(name) for name in required_checks if checks.get(str(name)) is not True]
    if failed_checks:
        raise AcceptedCensusDriftError(
            "accepted census required checks failed: " + ", ".join(failed_checks)
        )
    if checks.get("new_large_cap_scan_population_promoted") is not True:
        raise AcceptedCensusDriftError("accepted census population was not promoted")
    required_check_set = {str(name) for name in required_checks}
    declared_check_set = {str(name) for name in checks} - {"production_sector_scan_exercised"}
    if required_check_set != declared_check_set:
        raise AcceptedCensusDriftError("acceptance required-check manifest is incomplete")
    missing_bridge_checks = sorted(BRIDGE_REQUIRED_ACCEPTANCE_CHECKS - required_check_set)
    if missing_bridge_checks:
        raise AcceptedCensusDriftError(
            "acceptance omits bridge-critical checks: " + ", ".join(missing_bridge_checks)
        )
    if (
        acceptance.get("actual_llm_calls") != 0
        or float(acceptance.get("actual_llm_cost_usd") or 0.0) != 0.0
    ):
        raise AcceptedCensusDriftError("accepted census was not provider-free")

    try:
        stored_manifest = json.loads(str(run.get("source_manifest_json") or "[]"))
    except json.JSONDecodeError as exc:
        raise AcceptedCensusDriftError("stored source manifest is invalid JSON") from exc
    if stored_manifest != artifact_manifest:
        raise AcceptedCensusDriftError("stored and exported source manifests differ")
    if _policy_manifest_from_artifact(artifact_manifest) != _current_policy_manifest():
        raise AcceptedCensusDriftError("census policy or taxonomy code hash drifted")
    if coverage.get("taxonomy") != canonical_sector_taxonomy_manifest():
        raise AcceptedCensusDriftError("canonical taxonomy manifest drifted")
    if coverage.get("external_industry_taxonomy") != external_industry_taxonomy_manifest():
        raise AcceptedCensusDriftError("external industry taxonomy manifest drifted")
    return acceptance, coverage, artifact_manifest, discrepancies


def _verify_snapshot_manifest(
    *,
    root: Path,
    inputs: CensusInputPaths,
    source_manifest: Sequence[Mapping[str, Any]],
) -> None:
    """Prove the exported manifest names the exact immutable replay inputs."""

    from app.universe.us_equity_census import _sha256_file, _source_role_files

    actual_rows = {
        str(row.get("logical_role") or ""): row
        for row in source_manifest
        if str(row.get("artifact_kind") or "") == "SOURCE_SNAPSHOT"
    }
    expected_roles = {role for role, _path in _source_role_files(inputs)}
    if set(actual_rows) != expected_roles:
        raise AcceptedCensusDriftError("source snapshot role manifest drifted")
    snapshots_root = (root / "source_snapshots").resolve()
    for role, path in _source_role_files(inputs):
        row = actual_rows[role]
        resolved_path = path.resolve()
        if not resolved_path.is_relative_to(snapshots_root):
            raise AcceptedCensusDriftError(f"source snapshot escapes artifact root: {role}")
        digest = _sha256_file(path)
        if (
            str(row.get("sha256") or "") != digest
            or int(row.get("bytes") or -1) != path.stat().st_size
        ):
            raise AcceptedCensusDriftError(f"source snapshot content drifted: {role}")
        declared_snapshot = Path(str(row.get("snapshot_path") or ""))
        if not declared_snapshot.is_absolute():
            declared_snapshot = Path.cwd() / declared_snapshot
        if declared_snapshot.resolve() != resolved_path:
            raise AcceptedCensusDriftError(f"source snapshot path drifted: {role}")
    database_path = root / "source_snapshots" / "database_inputs.json"
    database_digest = _sha256_file(database_path)
    database_rows = [
        row
        for row in source_manifest
        if str(row.get("snapshot_path") or "").endswith("database_inputs.json")
        and not row.get("artifact_kind")
    ]
    if len(database_rows) != 1 or str(database_rows[0].get("sha256") or "") != database_digest:
        raise AcceptedCensusDriftError("database input snapshot manifest drifted")


def _date_not_after(value: Any, *, as_of_date: str, label: str, required: bool = True) -> str:
    normalized = str(value or "").strip()[:10]
    if required and not normalized:
        raise AcceptedCensusDriftError(f"accepted member is missing {label}")
    if normalized and normalized > as_of_date:
        raise AcceptedCensusDriftError(f"accepted member has future-dated {label}: {normalized}")
    return normalized


def _require_text(row: Mapping[str, Any], field: str, *, ticker: str) -> str:
    value = str(row.get(field) or "").strip()
    if not value:
        raise AcceptedCensusDriftError(f"accepted member {ticker} is missing {field}")
    return value


def _build_members(
    *,
    run_id: str,
    as_of_date: str,
    issuer_rows: Sequence[Mapping[str, Any]],
    security_rows: Sequence[Mapping[str, Any]],
) -> tuple[AcceptedCensusMember, ...]:
    admitted_issuers = [
        row for row in issuer_rows if row.get("terminal_disposition") == ACCEPTED_ISSUER_DISPOSITION
    ]
    admitted_securities = [
        row
        for row in security_rows
        if row.get("terminal_disposition") == ACCEPTED_SECURITY_DISPOSITION
    ]
    if not admitted_issuers:
        raise AcceptedCensusDriftError(f"accepted census {run_id} has an empty cohort")
    security_by_key: dict[str, Mapping[str, Any]] = {}
    for row in admitted_securities:
        key = str(row.get("security_key") or "")
        if not key or key in security_by_key:
            raise AcceptedCensusDriftError("admitted primary security keys are not unique")
        security_by_key[key] = row

    members: list[AcceptedCensusMember] = []
    seen_tickers: set[str] = set()
    for issuer in sorted(admitted_issuers, key=lambda row: str(row.get("primary_ticker") or "")):
        ticker = _require_text(issuer, "primary_ticker", ticker="UNKNOWN").upper()
        issuer_contract = {
            "issuer_type": "OPERATING_ISSUER",
            "operating_status": "OPERATING",
            "is_operating_company": 1,
            "is_us_domiciled": 1,
            "is_us_listed_foreign_issuer": 0,
            "primary_selection_status": "RESOLVED",
            "market_cap_status": "RESOLVED",
            "cap_band": "large_and_mega",
            "cap_band_status": "RESOLVED",
            "sector_status": "RESOLVED",
            "processing_status": "COMPLETED",
        }
        failed = [
            field for field, expected in issuer_contract.items() if issuer.get(field) != expected
        ]
        if failed:
            raise AcceptedCensusDriftError(
                f"accepted issuer {ticker} violates: {', '.join(failed)}"
            )
        security_key = _require_text(issuer, "primary_security_key", ticker=ticker)
        security = security_by_key.get(security_key)
        if security is None:
            raise AcceptedCensusDriftError(
                f"accepted issuer {ticker} lacks its admitted primary security"
            )
        security_contract = {
            "listing_status": "ACTIVE",
            "is_primary_security": 1,
            "is_secondary_class": 0,
            "is_duplicate_listing": 0,
            "security_type_status": "RESOLVED",
            "is_common_equity": 1,
            "is_adr": 0,
            "identity_status": "RESOLVED",
            "processing_status": "COMPLETED",
        }
        security_failed = [
            field
            for field, expected in security_contract.items()
            if security.get(field) != expected
        ]
        if security_failed or security.get("security_type") not in ACCEPTED_PRIMARY_SECURITY_TYPES:
            labels = [*security_failed]
            if security.get("security_type") not in ACCEPTED_PRIMARY_SECURITY_TYPES:
                labels.append("security_type")
            raise AcceptedCensusDriftError(
                f"accepted primary security {ticker} violates: {', '.join(labels)}"
            )
        issuer_key = _require_text(issuer, "issuer_key", ticker=ticker)
        if str(security.get("issuer_key") or "") != issuer_key:
            raise AcceptedCensusDriftError(
                f"accepted primary security {ticker} links to wrong issuer"
            )
        if str(security.get("ticker") or "").strip().upper() != ticker:
            raise AcceptedCensusDriftError(
                f"accepted primary ticker identity mismatch for {ticker}"
            )
        if ticker in seen_tickers:
            raise AcceptedCensusDriftError(f"accepted cohort duplicates ticker {ticker}")
        seen_tickers.add(ticker)

        try:
            market_cap_usd = float(issuer.get("market_cap_usd") or 0.0)
        except (TypeError, ValueError, OverflowError) as exc:
            raise AcceptedCensusDriftError(
                f"accepted issuer {ticker} has invalid market cap"
            ) from exc
        if not math.isfinite(market_cap_usd):
            raise AcceptedCensusDriftError(
                f"accepted issuer {ticker} has non-finite market cap"
            )
        if market_cap_usd < 10_000_000_000.0:
            raise AcceptedCensusDriftError(f"accepted issuer {ticker} is below the large-cap floor")
        market_cap_as_of = _date_not_after(
            issuer.get("market_cap_as_of_date"),
            as_of_date=as_of_date,
            label="market_cap_as_of_date",
        )
        if market_cap_as_of != as_of_date:
            raise AcceptedCensusDriftError(
                f"accepted issuer {ticker} cap is not exact-as-of {as_of_date}"
            )
        identity_as_of = _date_not_after(
            issuer.get("identity_as_of_date"),
            as_of_date=as_of_date,
            label="identity_as_of_date",
        )
        listing_as_of = _date_not_after(
            security.get("listing_status_as_of_date"),
            as_of_date=as_of_date,
            label="listing_status_as_of_date",
        )
        _date_not_after(
            issuer.get("domicile_as_of_date"),
            as_of_date=as_of_date,
            label="domicile_as_of_date",
        )
        _date_not_after(
            issuer.get("primary_selection_as_of_date"),
            as_of_date=as_of_date,
            label="primary_selection_as_of_date",
        )
        _date_not_after(
            issuer.get("sector_as_of_date"),
            as_of_date=as_of_date,
            label="sector_as_of_date",
        )

        source_sector = _require_text(issuer, "source_sector_label", ticker=ticker)
        canonical_sector = _require_text(issuer, "canonical_sector", ticker=ticker)
        resolution = resolve_canonical_sector(source_sector)
        if (
            source_sector not in ACTIVE_SECTOR_LABELS
            or resolution.contract is None
            or resolution.contract.canonical_sector_id != canonical_sector
        ):
            raise AcceptedCensusDriftError(f"accepted issuer {ticker} has taxonomy drift")

        cap_derivation_json = _require_text(issuer, "cap_derivation_json", ticker=ticker)
        try:
            cap_derivation = json.loads(cap_derivation_json)
        except json.JSONDecodeError as exc:
            raise AcceptedCensusDriftError(
                f"accepted issuer {ticker} has invalid cap derivation JSON"
            ) from exc
        cap_derivation_json = _canonical_json(cap_derivation)
        members.append(
            AcceptedCensusMember(
                ticker=ticker,
                issuer_key=issuer_key,
                security_key=security_key,
                cik=_require_text(issuer, "cik", ticker=ticker),
                legal_name=_require_text(issuer, "legal_name", ticker=ticker),
                exchange_name=_require_text(issuer, "primary_exchange_name", ticker=ticker),
                exchange_mic=(str(issuer.get("primary_exchange_mic") or "").strip() or None),
                source_sector_label=source_sector,
                canonical_sector=canonical_sector,
                market_cap_usd=market_cap_usd,
                market_cap_as_of_date=market_cap_as_of,
                market_cap_method=_require_text(issuer, "market_cap_method", ticker=ticker),
                market_cap_source_provider=_require_text(
                    issuer, "market_cap_source_provider", ticker=ticker
                ),
                market_cap_source_url=_require_text(issuer, "market_cap_source_url", ticker=ticker),
                market_cap_retrieved_at=_require_text(
                    issuer, "market_cap_retrieved_at", ticker=ticker
                ),
                market_cap_confidence=_require_text(issuer, "market_cap_confidence", ticker=ticker),
                cap_derivation_json=cap_derivation_json,
                identity_source_provider=_require_text(
                    issuer, "identity_source_provider", ticker=ticker
                ),
                identity_source_url=_require_text(issuer, "identity_source_url", ticker=ticker),
                identity_as_of_date=identity_as_of,
                identity_retrieved_at=_require_text(issuer, "identity_retrieved_at", ticker=ticker),
                identity_confidence=_require_text(issuer, "identity_confidence", ticker=ticker),
                listing_source_url=_require_text(security, "listing_source_url", ticker=ticker),
                listing_status_as_of_date=listing_as_of,
                listing_retrieved_at=_require_text(security, "listing_retrieved_at", ticker=ticker),
            )
        )
    if len(members) != len(admitted_securities):
        raise AcceptedCensusDriftError(
            "admitted issuer and primary-security populations are not one-to-one"
        )
    return tuple(members)


def _verify_promotion(
    conn: sqlite3.Connection,
    *,
    lineage_run_id: str,
    input_fingerprint: str,
    as_of_date: str,
    members: Sequence[AcceptedCensusMember],
) -> None:
    expected = {
        member.ticker: (member.source_sector_label, member.issuer_key) for member in members
    }
    rows = conn.execute(
        "SELECT ticker, inferred_sector, derived_from FROM sector_inference WHERE as_of_date = ?",
        (as_of_date,),
    ).fetchall()
    actual: dict[str, tuple[str, str]] = {}
    for row in rows:
        try:
            derived = json.loads(str(row["derived_from"] or "{}"))
        except json.JSONDecodeError:
            continue
        if derived.get("producer") != "us_equity_census_issuer_deduplication":
            continue
        ticker = str(row["ticker"] or "").strip().upper()
        if (
            derived.get("run_id") != lineage_run_id
            or derived.get("input_fingerprint") != input_fingerprint
            or derived.get("taxonomy_version") != CANONICAL_SECTOR_TAXONOMY_VERSION
        ):
            raise AcceptedCensusDriftError(f"promoted census lineage drifted for {ticker}")
        actual[ticker] = (str(row["inferred_sector"] or ""), str(derived.get("issuer_key") or ""))
    if actual != expected:
        raise AcceptedCensusDriftError("promoted sector population does not equal accepted cohort")


def _validate_run(
    conn: sqlite3.Connection,
    *,
    run: Mapping[str, Any],
) -> AcceptedCensusCohort:
    run_id = str(run["run_id"])
    as_of_date = str(run["as_of_date"])
    input_fingerprint = str(run.get("input_fingerprint") or "")
    resume_fingerprint = str(run.get("resume_fingerprint") or "")
    if not input_fingerprint or input_fingerprint != resume_fingerprint:
        raise AcceptedCensusDriftError(f"accepted census {run_id} fingerprint drifted")
    try:
        status_detail = json.loads(str(run.get("status_detail_json") or "{}"))
    except json.JSONDecodeError as exc:
        raise AcceptedCensusDriftError(
            f"accepted census {run_id} status detail is invalid"
        ) from exc
    semantic = str(status_detail.get("semantic_output_fingerprint") or "")
    if not semantic or status_detail.get("provider_free_replay_verified") is not True:
        raise AcceptedCensusDriftError(f"accepted census {run_id} lacks replay proof")

    checkpoint_path, root = _artifact_root(str(run.get("checkpoint_path") or ""))
    if not checkpoint_path.is_file():
        raise AcceptedCensusDriftError(f"accepted census checkpoint is missing: {checkpoint_path}")
    _acceptance, coverage, source_manifest, discrepancies = _require_exact_artifacts(
        run=run,
        checkpoint_path=checkpoint_path,
        root=root,
    )
    inputs, database_inputs = _snapshot_inputs(root)
    _verify_snapshot_manifest(
        root=root,
        inputs=inputs,
        source_manifest=source_manifest,
    )
    recomputed_input = census_input_fingerprint(
        inputs,
        as_of_date=as_of_date,
        database_inputs=database_inputs,
    )
    if recomputed_input != input_fingerprint:
        raise AcceptedCensusDriftError(f"accepted census {run_id} immutable input drifted")

    issuer_rows = [
        dict(row)
        for row in conn.execute(
            "SELECT * FROM us_equity_census_issuers WHERE run_id = ?", (run_id,)
        )
    ]
    security_rows = [
        dict(row)
        for row in conn.execute(
            "SELECT * FROM us_equity_census_securities WHERE run_id = ?", (run_id,)
        )
    ]
    unmatched_caps = discrepancies.get("unmatched_large_cap_source_rows")
    if not isinstance(unmatched_caps, list):
        raise AcceptedCensusDriftError("source discrepancy artifact lacks unmatched cap ledger")
    recomputed_semantic = _semantic_replay_fingerprint(security_rows, issuer_rows, unmatched_caps)
    if recomputed_semantic != semantic:
        raise AcceptedCensusDriftError(f"accepted census {run_id} semantic output drifted")

    members = _build_members(
        run_id=run_id,
        as_of_date=as_of_date,
        issuer_rows=issuer_rows,
        security_rows=security_rows,
    )
    counts = coverage.get("counts")
    if not isinstance(counts, Mapping) or counts.get("admitted_large_and_mega_issuers") != len(
        members
    ):
        raise AcceptedCensusDriftError("coverage count does not match accepted cohort")
    large_count = sum(member.market_cap_usd < 200_000_000_000.0 for member in members)
    mega_count = sum(member.market_cap_usd >= 200_000_000_000.0 for member in members)
    if (
        counts.get("admitted_large_cap_issuers") != large_count
        or counts.get("admitted_mega_cap_issuers") != mega_count
    ):
        raise AcceptedCensusDriftError("coverage cap-band counts do not match accepted cohort")
    _verify_promotion(
        conn,
        lineage_run_id=run_id,
        input_fingerprint=input_fingerprint,
        as_of_date=as_of_date,
        members=members,
    )

    cohort_fingerprint = _sha256_json([member.to_dict() for member in members])
    policy_manifest = _policy_manifest_from_artifact(source_manifest)
    taxonomy = coverage["taxonomy"]
    external_taxonomy = coverage["external_industry_taxonomy"]
    lineage = AcceptedCensusLineage(
        run_id=run_id,
        as_of_date=as_of_date,
        target_band="large_and_mega",
        input_fingerprint=input_fingerprint,
        resume_fingerprint=resume_fingerprint,
        semantic_output_fingerprint=semantic,
        source_manifest_sha256=_sha256_json(source_manifest),
        policy_manifest_sha256=_sha256_json(policy_manifest),
        taxonomy_version=str(taxonomy["taxonomy_version"]),
        taxonomy_hash=str(taxonomy["taxonomy_hash"]),
        external_industry_taxonomy_version=str(external_taxonomy["taxonomy_version"]),
        external_industry_taxonomy_hash=str(external_taxonomy["taxonomy_hash"]),
        checkpoint_path=str(checkpoint_path),
        cohort_fingerprint=cohort_fingerprint,
        member_count=len(members),
    )
    return AcceptedCensusCohort(lineage=lineage, members=members)


def select_accepted_census_run(
    conn: sqlite3.Connection,
    *,
    as_of_date: str,
    target_band: str = "large_and_mega",
) -> AcceptedCensusCohort:
    """Select and fully validate the newest exact-date accepted census run."""

    if target_band != "large_and_mega":
        raise ValueError("accepted census production bridge currently supports large_and_mega")
    prior_row_factory = conn.row_factory
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT *
            FROM us_equity_census_runs
            WHERE as_of_date = ?
              AND target_band = ?
              AND status = 'COMPLETED'
              AND current_stage = 'TERMINAL_DISPOSITION'
              AND acceptance_status = 'PASSED'
            ORDER BY completed_at DESC, updated_at DESC, run_id DESC
            """,
            (as_of_date, target_band),
        ).fetchall()
        failures: list[str] = []
        for row in rows:
            run = dict(row)
            try:
                return _validate_run(conn, run=run)
            except AcceptedCensusDriftError as exc:
                failures.append(f"{run['run_id']}: {exc}")
        detail = "; ".join(failures[:3]) if failures else "no persisted accepted run"
        raise AcceptedCensusUnavailableError(
            f"no valid accepted {target_band} census for {as_of_date}: {detail}"
        )
    finally:
        conn.row_factory = prior_row_factory


def load_accepted_census_cohort(
    *,
    db_path: str | Path,
    as_of_date: str,
) -> AcceptedCensusCohort:
    """Open ``db_path`` query-only and return the validated cohort."""

    path = Path(db_path).resolve()
    if not path.is_file():
        raise AcceptedCensusUnavailableError(f"census database does not exist: {path}")
    conn = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    try:
        conn.execute("PRAGMA query_only = ON")
        conn.execute("PRAGMA temp_store = MEMORY")
        return select_accepted_census_run(conn, as_of_date=as_of_date)
    finally:
        conn.close()


__all__ = [
    "ACCEPTED_CENSUS_CAP_SOURCE",
    "ACCEPTED_ISSUER_DISPOSITION",
    "ACCEPTED_PRIMARY_SECURITY_TYPES",
    "ACCEPTED_SECURITY_DISPOSITION",
    "AcceptedCensusCohort",
    "AcceptedCensusDriftError",
    "AcceptedCensusError",
    "AcceptedCensusLineage",
    "AcceptedCensusMember",
    "AcceptedCensusUnavailableError",
    "load_accepted_census_cohort",
    "select_accepted_census_run",
]
