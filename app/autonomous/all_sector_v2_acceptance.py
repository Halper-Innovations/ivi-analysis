"""Deterministic release acceptance for an all-sector v2 replay.

The evaluator is intentionally offline and read-only.  It consumes persisted
benchmark/replay JSON plus an optional acceptance-evidence ledger, follows only
explicit local artifact references, and recomputes every supported invariant.
An absent evidence family is ``NOT_EVALUABLE``; it is never silently treated as
passing or reconstructed from unrelated fields.

Run it directly with::

    .venv/bin/python3 -m app.autonomous.all_sector_v2_acceptance \
        --input path/to/benchmark_summary.json \
        --evidence path/to/acceptance_evidence.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.autonomous.sector_contract import (
    AutonomousSectorFinancialRunArtifact,
    SECTOR_CONTRACT_VERSION_V2,
    SECTOR_PIPELINE_VERSION_V2,
    SectorCompanyFinancialPacket,
    SectorExpectedReturnScenario,
    selection_validation_terminal_ledger_fingerprint,
)
from app.autonomous.competitive_frontier import build_competitive_frontier
from app.autonomous.sector_lane_budget import CANONICAL_LANES
from app.autonomous.structural_gate import evaluate_going_concern_filing_text
from app.llm.synthesis_agent import _estimate_cost_usd
from app.sector.canonical_taxonomy import ACTIVE_SECTOR_LABELS


PASS = "PASS"
FAIL = "FAIL"
NOT_EVALUABLE = "NOT_EVALUABLE"

VALID_TERMINAL_STATES = {
    "OUT_OF_SCOPE",
    "SCREENED_OUT",
    "NEEDS_DATA",
    "READY_FOR_UNDERWRITING",
    "UNDERWRITTEN",
}
VALID_SCOPE_STATUSES = {"IN_SCOPE", "OUT_OF_SCOPE"}
PRECISE_NEEDS_DATA_EXCLUSIONS = {
    "",
    "UNKNOWN",
    "DATA_INCOMPLETE",
    "MISSING_DATA",
    "MISSING_FACTS",
    "MISSING_NORMALIZED_FACTS",
    "MISSING_PRICE",
    "MISSING_FILING",
}
GOING_CONCERN_ASSERTION_SUBJECTS = {
    "REGISTRANT",
    "CONSOLIDATED_GROUP",
    "CONSOLIDATED_SUBSIDIARY",
    "INVESTEE",
    "PARTNER",
    "COUNTERPARTY",
    "THIRD_PARTY",
    "UNATTRIBUTED",
}
GOING_CONCERN_ASSERTION_MODES = {
    "AFFIRMATIVE_CURRENT",
    "NEGATED",
    "HISTORICAL_RESOLVED",
    "ACCOUNTING_POLICY",
    "HYPOTHETICAL",
}
GOING_CONCERN_BLOCKABLE_SUBJECTS = {
    "REGISTRANT",
    "CONSOLIDATED_GROUP",
    "CONSOLIDATED_SUBSIDIARY",
}
LANE_INTEGER_FIELDS = (
    "tool_call_attempts",
    "tool_calls_ok",
    "tool_calls_failed",
    "provider_call_attempts",
    "provider_calls_ok",
    "provider_calls_failed",
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "reserved_output_tokens",
    "cost_microdollars",
)
_ARTIFACT_DIGEST_FIELDS = ("artifact_sha256", "sha256", "source_sha256")
_FIXED_COHORT_SCHEMA = "all_sector_v2_fixed_cohort_v1"
_TRUSTED_MANIFEST_POLICY = "CALLER_OR_CONFIG_PINNED_NOT_BENCHMARK_SELF_DECLARED"

# Independently derived from the canonical 2026-07-15 postmortem source artifacts.
# ``build_trusted_postmortem_manifest_hashes`` documents and reproduces every
# identifier.  The omitted-top-ten manifest comes from the persisted rank and
# prompt context in the canonical benchmark; the going-concern manifest binds
# the exact false-positive excerpts plus the two frozen affirmative controls.
TRUSTED_2026_07_15_POSTMORTEM_MANIFESTS: tuple[tuple[str, str], ...] = (
    (
        "sparse_history",
        "436bbd9109f0237de45fe4689ab25b40cc98e4625223d11b6e0aeb489dcf65cf",
    ),
    (
        "cap_price_losses",
        "f6f974daa8d6252cfe86e1a1e6147533c4a56403f4e4049fcea857285c25cc1d",
    ),
    (
        "cached_annual_filings",
        "6d6cb5988b4775f162147984cae32c53e8f2be76f565dff3f11102694469bfd1",
    ),
    (
        "extraction_gaps",
        "3e1a8d5a0b02ffd257a1ce5188fa53a1fb1f85ab6c6abf34db090424199f5347",
    ),
    (
        "previously_unreviewed",
        "7741d5562a4b059956540d8a3c7579c8c2184eb8f9f34970b38d7a1883c4e106",
    ),
    (
        "omitted_top_ten",
        "570920f89c54182a3e2f468b590e1eb4c2879214eacfcf576c1be1a9da3a792a",
    ),
    (
        "going_concern",
        "15f127284c9e691ce1c9b9e3d852f1dd46abf16841b049e6e7954fe3a69f564b",
    ),
)

_GOING_CONCERN_TRUE_CONTROLS: tuple[dict[str, Any], ...] = (
    {
        "ticker": "REGP",
        "subject": "REGISTRANT",
        "filing_text": (
            "<html><body>These conditions raise substantial doubt about our ability "
            "to continue as a going concern.</body></html>"
        ),
        "content_revision": "4bd39566c08c23262daff27c1d01364a4485f604489482f78a59ae5c7024e1f4",
        "accession": "acc-0",
        "form_type": "10-K",
        "filing_date": "2026-02-15",
        "assertion_mode": "AFFIRMATIVE_CURRENT",
        "evidence_ref_id": "sec-filing:REGP:acc-0",
    },
    {
        "ticker": "SUBP",
        "subject": "CONSOLIDATED_SUBSIDIARY",
        "filing_text": (
            "<html><body>These conditions raise substantial doubt about our wholly "
            "owned consolidated subsidiary's ability to continue as a going concern."
            "</body></html>"
        ),
        "content_revision": "a6a1106fc03011133695ef4b8defb5d77837d000cd87b80dce6abbd8a1307ff3",
        "accession": "acc-0",
        "form_type": "10-K",
        "filing_date": "2026-02-15",
        "assertion_mode": "AFFIRMATIVE_CURRENT",
        "evidence_ref_id": "sec-filing:SUBP:acc-0",
    },
)

_GOING_CONCERN_FALSE_FIXTURES: tuple[dict[str, str], ...] = (
    {
        "sector": "biotech",
        "ticker": "IQV",
        "gate_scorecard_as_of": "2026-07-15",
        "content_revision": "16c58e8ab9fd439c17e9d87f74bc7eeb45fbb2f4ba5b865ad63ab6386e3adfd4",
    },
    {
        "sector": "capital_markets",
        "ticker": "ICE",
        "gate_scorecard_as_of": "2026-07-15",
        "content_revision": "c33938d2b1e414f57b47d0046cb8e64696fd2cca6c09e44c3f3d3c7057907811",
    },
    {
        "sector": "consumer_staples",
        "ticker": "KO",
        "gate_scorecard_as_of": "2026-07-15",
        "content_revision": "79926d79012e43634aea644af157c66b23c666bd1bb6621e96014bec32269539",
    },
    {
        "sector": "energy",
        "ticker": "APA",
        "gate_scorecard_as_of": "2026-07-15",
        "content_revision": "0302d9725188b9526d628e2d117eae69d8acd47fe219a73e62eb5c1aa0633322",
    },
    {
        "sector": "enterprise_software",
        "ticker": "PLTR",
        "gate_scorecard_as_of": "2026-04-09",
        "content_revision": "afac3cf43a23b4477839916fde92ebf9b2fc32a531740fd846321cc29e4c80f2",
    },
    {
        "sector": "industrial_tech",
        "ticker": "HPE",
        "gate_scorecard_as_of": "2026-04-04",
        "content_revision": "d38bd7fc805dc1c4db13780793e477e79db7dbdcf1b9ca5c80995196e793cb7b",
    },
    {
        "sector": "medical_devices",
        "ticker": "BSX",
        "gate_scorecard_as_of": "2026-07-15",
        "content_revision": "ca63fc55d722d2e4c63ec233c0f3df70e8159ae53fe88acb45250bb2d0137e67",
    },
    {
        "sector": "semiconductors",
        "ticker": "AVGO",
        "gate_scorecard_as_of": "2026-07-15",
        "content_revision": "a4ed97ef53c17a7e0f31de9e1750c0891e0853090a8dac583ea0adb7b6126fed",
    },
)


@dataclass(frozen=True)
class AcceptanceThresholds:
    """Current production sector contract plus pinned repair-plan counts.

    ``expected_sector_labels=None`` is an explicit compatibility mode for
    historical fixtures whose acceptance contract was count-only.
    """

    sector_count: int = 34
    expected_sector_labels: tuple[str, ...] | None = ACTIVE_SECTOR_LABELS
    security_slots: int = 642
    sparse_history: int = 49
    cap_price_losses: int = 241
    cached_annual_filings: int = 115
    extraction_gaps: int = 7
    previously_unreviewed: int = 553
    omitted_top_ten: int = 36
    going_concern_false_positives: tuple[str, ...] = (
        "APA",
        "KO",
        "IQV",
        "ICE",
        "PLTR",
        "HPE",
        "BSX",
        "AVGO",
    )
    trusted_evidence_manifest_sha256: tuple[tuple[str, str], ...] = (
        TRUSTED_2026_07_15_POSTMORTEM_MANIFESTS
    )
    # Semantic provenance is deliberately separate from the benchmark's
    # identity-only cohort manifest.  A current run cannot certify its own SEC
    # source metadata merely by repeating it in two artifact fields.
    trusted_semantic_manifest_sha256: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class _Bundle:
    payload: dict[str, Any]
    evidence: dict[str, Any]
    data_plane_report: dict[str, Any]
    artifacts: tuple[dict[str, Any], ...]
    artifact_models: tuple[AutonomousSectorFinancialRunArtifact, ...]
    artifact_contract_errors: tuple[str, ...]
    source_path: Path | None
    input_errors: tuple[str, ...]


class _Report:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def add(
        self,
        check_id: str,
        status: str,
        *,
        expected: Any = None,
        actual: Any = None,
        evidence_source: str | None = None,
        details: Mapping[str, Any] | None = None,
        missing_fields: Sequence[str] = (),
    ) -> None:
        row: dict[str, Any] = {"id": check_id, "status": status}
        if expected is not None:
            row["expected"] = expected
        if actual is not None:
            row["actual"] = actual
        if evidence_source:
            row["evidence_source"] = evidence_source
        if details:
            row["details"] = dict(details)
        if missing_fields:
            row["missing_fields"] = list(missing_fields)
        self.checks.append(row)

    def unavailable(
        self,
        check_id: str,
        *,
        expected: Any,
        missing_fields: Sequence[str],
    ) -> None:
        self.add(
            check_id,
            NOT_EVALUABLE,
            expected=expected,
            missing_fields=missing_fields,
        )

    @property
    def status(self) -> str:
        statuses = {row["status"] for row in self.checks}
        if FAIL in statuses:
            return FAIL
        if NOT_EVALUABLE in statuses or not statuses:
            return NOT_EVALUABLE
        return PASS


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _rows(value: Any) -> list[dict[str, Any]] | None:
    if not isinstance(value, list):
        return None
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _upper(value: Any) -> str:
    return str(value or "").strip().upper()


def _lower(value: Any) -> str:
    return str(value or "").strip().lower()


def _normalized_cik(value: Any) -> str | None:
    text = str(value or "").strip()
    digits = "".join(character for character in text if character.isdigit())
    return (digits.lstrip("0") or "0") if digits else None


def _issuer_key(row: Mapping[str, Any]) -> str | None:
    cik = _normalized_cik(row.get("issuer_cik") or row.get("cik"))
    if cik:
        return f"CIK:{cik}"
    token = _upper(row.get("issuer_key") or row.get("issuer_id"))
    return f"ISSUER:{token}" if token else None


def _slot_key(row: Mapping[str, Any]) -> tuple[str, str] | None:
    sector = _lower(row.get("sector"))
    ticker = _upper(row.get("ticker"))
    return (sector, ticker) if sector and ticker else None


def _positive_finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
    )


def _precise_reason(value: Any) -> bool:
    return _upper(value) not in PRECISE_NEEDS_DATA_EXCLUSIONS


def _manifest_sha256(identifiers: Sequence[str]) -> str:
    payload = json.dumps(
        sorted(identifiers),
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_text(value: Any) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def _canonical_identifier(family: str, row: Mapping[str, Any]) -> str | None:
    """Derive one cohort identity from normalized family-specific source fields."""

    sector = _lower(row.get("sector"))
    ticker = _upper(row.get("ticker"))
    if family in {
        "sparse_history",
        "cap_price_losses",
        "cached_annual_filings",
        "extraction_gaps",
        "previously_unreviewed",
        "omitted_top_ten",
    } and (not sector or not ticker):
        return None
    if family == "sparse_history":
        return f"sparse_history:{sector}:{ticker}"
    if family == "cap_price_losses":
        return f"cap_price_loss:{sector}:{ticker}"
    if family in {"cached_annual_filings", "extraction_gaps"}:
        accession = str(
            row.get("accession") or row.get("available_source_accession") or ""
        ).strip()
        if not accession:
            return None
        prefix = (
            "cached_annual_filing"
            if family == "cached_annual_filings"
            else "extraction_gap"
        )
        return f"{prefix}:{sector}:{ticker}:{accession}"
    if family == "previously_unreviewed":
        return f"previously_unreviewed:{sector}:{ticker}"
    if family == "omitted_top_ten":
        try:
            rank = int(row.get("rank"))
        except (TypeError, ValueError):
            return None
        if rank < 1:
            return None
        return f"omitted_top_ten:{sector}:{ticker}:{rank}"
    if family == "going_concern":
        fixture_kind = _lower(row.get("fixture_kind") or row.get("kind"))
        if fixture_kind == "false_positive":
            gate_as_of = str(
                row.get("gate_scorecard_as_of") or row.get("gate_as_of_date") or ""
            ).strip()
            if not sector or not ticker or not gate_as_of:
                return None
            return f"going_concern:false:{sector}:{ticker}:{gate_as_of}"
        if fixture_kind == "true_positive":
            subject = _upper(row.get("subject") or row.get("control_type"))
            revision = str(row.get("content_revision") or "").strip().lower()
            if not ticker or not subject or not revision:
                return None
            return f"going_concern:true:{subject.lower()}:{ticker}:{revision}"
    return None


def build_trusted_postmortem_manifest_hashes(
    postmortem_dir: str | Path,
    *,
    thresholds: AcceptanceThresholds | None = None,
    as_of_date: str = "2026-07-15",
    canonical_benchmark_path: str | Path | None = None,
    going_concern_false_positive_path: str | Path | None = None,
    expected_benchmark_sha256: str | None = None,
    expected_going_concern_csv_sha256: str | None = None,
) -> dict[str, str]:
    """Rebuild trusted cohort manifests from the canonical source ledgers.

    Canonical identifiers are deliberately constructed from stable source
    identity, never from replay output:

    * ``sparse_history:{sector}:{ticker}``
    * ``cap_price_loss:{sector}:{ticker}``
    * ``cached_annual_filing:{sector}:{ticker}:{available_accession}``
    * ``extraction_gap:{sector}:{ticker}:{available_accession}``
    * ``previously_unreviewed:{sector}:{ticker}``

    When the independently pinned benchmark and false-positive CSV are supplied,
    the omitted-top-ten and going-concern manifests are rebuilt as well.
    """

    expected = thresholds or AcceptanceThresholds()
    root = Path(postmortem_dir)

    def read_csv_rows(name: str) -> list[dict[str, str]]:
        with (root / name).open(newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]

    def truthy(value: Any) -> bool:
        return str(value or "").strip().lower() in {"1", "true", "yes"}

    def slot_id(prefix: str, row: Mapping[str, Any]) -> str:
        sector = _lower(row.get("sector"))
        ticker = _upper(row.get("ticker"))
        if not sector or not ticker:
            raise ValueError(f"canonical {prefix} row lacks sector/ticker identity")
        return f"{prefix}:{sector}:{ticker}"

    company_rows = read_csv_rows("company_funnel.csv")
    filing_rows = read_csv_rows("filing_availability_cache_audit.csv")
    sparse_rows = [row for row in company_rows if truthy(row.get("financial_history_excluded"))]
    price_loss_rows = [
        row for row in company_rows if _upper(row.get("data_quality_status")) == "MISSING_PRICE"
    ]
    filing_available_rows = [
        row
        for row in filing_rows
        if _upper(row.get("availability_bucket")) != "NO_VALID_CACHED_ANNUAL_PATH"
    ]

    def csv_positive(value: Any) -> bool:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return False
        return math.isfinite(parsed) and parsed > 0

    extraction_gap_rows = [
        row
        for row in filing_available_rows
        if _upper(row.get("audit_disposition")) == "FILING_EXISTS_EXTRACTION_GAP"
        and not csv_positive(row.get("fallback_full_10k_risk_text_chars"))
    ]
    unreviewed_rows = [
        row
        for row in company_rows
        if truthy(row.get("packet_created"))
        and not truthy(row.get("company_autonomy_reviewed"))
    ]
    selected_rows = [
        *sparse_rows,
        *price_loss_rows,
        *filing_available_rows,
        *extraction_gap_rows,
        *unreviewed_rows,
    ]
    wrong_dates = sorted(
        {
            str(row.get("as_of_date") or "").strip()
            for row in selected_rows
            if str(row.get("as_of_date") or "").strip() != as_of_date
        }
    )
    if wrong_dates:
        raise ValueError(
            f"canonical postmortem rows do not match as-of {as_of_date}: {wrong_dates}"
        )

    def filing_id(prefix: str, row: Mapping[str, Any]) -> str:
        accession = str(row.get("available_source_accession") or "").strip()
        if not accession:
            raise ValueError(f"canonical {prefix} row lacks available accession")
        return f"{slot_id(prefix, row)}:{accession}"

    identifiers = {
        "sparse_history": [slot_id("sparse_history", row) for row in sparse_rows],
        "cap_price_losses": [slot_id("cap_price_loss", row) for row in price_loss_rows],
        "cached_annual_filings": [
            filing_id("cached_annual_filing", row) for row in filing_available_rows
        ],
        "extraction_gaps": [
            filing_id("extraction_gap", row) for row in extraction_gap_rows
        ],
        "previously_unreviewed": [
            slot_id("previously_unreviewed", row) for row in unreviewed_rows
        ],
    }
    expected_counts = {
        "sparse_history": expected.sparse_history,
        "cap_price_losses": expected.cap_price_losses,
        "cached_annual_filings": expected.cached_annual_filings,
        "extraction_gaps": expected.extraction_gaps,
        "previously_unreviewed": expected.previously_unreviewed,
    }

    if canonical_benchmark_path is not None:
        benchmark_path = Path(canonical_benchmark_path)
        benchmark_digest = hashlib.sha256(benchmark_path.read_bytes()).hexdigest()
        if expected_benchmark_sha256 and benchmark_digest != expected_benchmark_sha256:
            raise ValueError("canonical benchmark SHA-256 does not match trusted input")
        benchmark = _read_json(benchmark_path)
        omitted: list[str] = []
        for result in benchmark.get("sector_results") or []:
            if not isinstance(result, Mapping) or not result.get("artifact_path"):
                continue
            artifact_path = _resolve_reference(
                result.get("artifact_path"), source_path=benchmark_path
            )
            if artifact_path is None:
                continue
            artifact = _read_json(artifact_path)
            ranking = [
                dict(item)
                for item in artifact.get("relative_ranking") or []
                if isinstance(item, Mapping) and _upper(item.get("ticker"))
            ]

            def rank_key(item: Mapping[str, Any]) -> tuple[float, str]:
                try:
                    rank = float(item.get("rank"))
                except (TypeError, ValueError):
                    rank = math.inf
                return rank, _upper(item.get("ticker"))

            top_ten = sorted(ranking, key=rank_key)[:10]
            prompt = _mapping(artifact.get("final_decision_prompt_context"))
            scoped = {_upper(item) for item in prompt.get("prompt_scoped_tickers") or []}
            sector = _lower(artifact.get("sector") or result.get("sector"))
            omitted.extend(
                f"omitted_top_ten:{sector}:{_upper(item.get('ticker'))}:{int(float(item.get('rank')))}"
                for item in top_ten
                if _upper(item.get("ticker")) not in scoped
            )
        identifiers["omitted_top_ten"] = omitted
        expected_counts["omitted_top_ten"] = expected.omitted_top_ten

    if going_concern_false_positive_path is not None:
        going_path = Path(going_concern_false_positive_path)
        going_digest = hashlib.sha256(going_path.read_bytes()).hexdigest()
        if (
            expected_going_concern_csv_sha256
            and going_digest != expected_going_concern_csv_sha256
        ):
            raise ValueError("going-concern CSV SHA-256 does not match trusted input")
        with going_path.open(
            newline="", encoding="utf-8"
        ) as handle:
            false_rows = [dict(row) for row in csv.DictReader(handle)]
        wrong_classifications = [
            _upper(row.get("ticker"))
            for row in false_rows
            if _upper(row.get("postmortem_classification"))
            != "FALSE_POSITIVE_ISSUER_ATTRIBUTION_OR_SUBSTRING"
        ]
        if wrong_classifications:
            raise ValueError(
                "going-concern source contains non-false-positive rows: "
                f"{wrong_classifications}"
            )
        going_rows = [
            {
                **row,
                "fixture_kind": "false_positive",
                "content_revision": _sha256_text(row.get("matched_context")),
            }
            for row in false_rows
        ] + [
            {**row, "fixture_kind": "true_positive"}
            for row in _GOING_CONCERN_TRUE_CONTROLS
        ]
        production_false = {
            _upper(row.get("ticker")): row for row in _GOING_CONCERN_FALSE_FIXTURES
        }
        if set(expected.going_concern_false_positives) == set(production_false):
            source_false = {_upper(row.get("ticker")): row for row in going_rows[:-2]}
            mismatches = [
                ticker
                for ticker, trusted in production_false.items()
                if ticker not in source_false
                or _lower(source_false[ticker].get("sector")) != trusted["sector"]
                or str(source_false[ticker].get("gate_scorecard_as_of") or "")
                != trusted["gate_scorecard_as_of"]
                or str(source_false[ticker].get("content_revision") or "")
                != trusted["content_revision"]
            ]
            if mismatches:
                raise ValueError(
                    "going-concern source does not match frozen fixture content: "
                    f"{mismatches}"
                )
        identifiers["going_concern"] = [
            identifier
            for row in going_rows
            if (identifier := _canonical_identifier("going_concern", row)) is not None
        ]
        expected_counts["going_concern"] = (
            len(expected.going_concern_false_positives)
            + len(_GOING_CONCERN_TRUE_CONTROLS)
        )
    for family, values in identifiers.items():
        if len(values) != expected_counts[family] or len(values) != len(set(values)):
            raise ValueError(
                f"canonical {family} ledger mismatch: rows={len(values)} "
                f"unique={len(set(values))} expected={expected_counts[family]}"
            )
    return {family: _manifest_sha256(values) for family, values in identifiers.items()}


def load_trusted_manifest_hashes(path: str | Path) -> tuple[tuple[str, str], ...]:
    """Load caller-pinned expected hashes, never hashes embedded in a benchmark."""

    payload = _read_json(Path(path))
    if payload.get("artifact_type") != "all_sector_v2_trusted_acceptance_manifest_v1":
        raise ValueError("unsupported trusted acceptance manifest artifact_type")
    raw = payload.get("evidence_manifest_sha256")
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("trusted acceptance manifest requires evidence_manifest_sha256")
    normalized: list[tuple[str, str]] = []
    for family, value in sorted(raw.items()):
        digest = str(value or "").strip().lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"invalid trusted SHA-256 for {family}")
        normalized.append((str(family), digest))
    return tuple(normalized)


def load_trusted_semantic_manifest_hashes(
    path: str | Path,
) -> tuple[tuple[str, str], ...]:
    """Load optional caller-pinned semantic provenance hashes."""

    payload = _read_json(Path(path))
    if payload.get("artifact_type") != "all_sector_v2_trusted_acceptance_manifest_v1":
        raise ValueError("unsupported trusted acceptance manifest artifact_type")
    raw = payload.get("semantic_manifest_sha256")
    if raw is None:
        return ()
    if not isinstance(raw, Mapping):
        raise ValueError("trusted semantic manifest must be a SHA-256 map")
    normalized: list[tuple[str, str]] = []
    for family, value in sorted(raw.items()):
        digest = str(value or "").strip().lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"invalid trusted semantic SHA-256 for {family}")
        normalized.append((str(family), digest))
    return tuple(normalized)


def _cohort_manifest_binding(
    bundle: _Bundle,
    *,
    thresholds: AcceptanceThresholds,
    family: str,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[str, dict[str, Any]]:
    fixed = _mapping(bundle.payload.get("fixed_cohort"))
    manifests = _mapping(fixed.get("evidence_manifest_sha256"))
    declared = str(manifests.get(family) or "").strip().lower()
    trusted_manifests = dict(thresholds.trusted_evidence_manifest_sha256)
    expected = str(trusted_manifests.get(family) or "").strip().lower()
    supplied = [str(row.get("canonical_id") or "").strip() for row in rows]
    identifiers = [_canonical_identifier(family, row) for row in rows]
    missing = [index for index, value in enumerate(supplied) if not value]
    unresolvable = [index for index, value in enumerate(identifiers) if not value]
    identity_mismatches = [
        {
            "row_index": index,
            "supplied": supplied[index] or None,
            "derived": derived,
        }
        for index, derived in enumerate(identifiers)
        if derived is not None and supplied[index] != derived
    ]
    present = [value for value in identifiers if value]
    duplicates = sorted(
        value for value in set(present) if present.count(value) > 1
    )
    actual = _manifest_sha256(present) if not missing and not duplicates else None
    details = {
        "family": family,
        "trusted_manifest_sha256": expected or None,
        "declared_manifest_sha256": declared or None,
        "actual_manifest_sha256": actual,
        "canonical_identifier_count": len(present),
        "missing_canonical_id_rows": missing[:30],
        "unresolvable_identity_rows": unresolvable[:30],
        "canonical_id_mismatches": identity_mismatches[:30],
        "duplicate_canonical_ids": duplicates[:30],
    }
    if not expected:
        details["reason"] = "no independently trusted manifest is configured"
        return NOT_EVALUABLE, details
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        details["invalid_expected_manifest"] = True
        return FAIL, details
    if declared and declared != expected:
        details["declared_trusted_mismatch"] = True
        return FAIL, details
    if missing or unresolvable or identity_mismatches or duplicates or actual != expected:
        return FAIL, details
    return PASS, details


def _manifest_bound_status(
    *,
    content_passed: bool,
    binding_status: str,
    proof_status: str = PASS,
) -> str:
    if not content_passed or binding_status == FAIL or proof_status == FAIL:
        return FAIL
    if binding_status == NOT_EVALUABLE or proof_status == NOT_EVALUABLE:
        return NOT_EVALUABLE
    return PASS


def _artifact_by_sector(bundle: _Bundle, sector: str) -> dict[str, Any] | None:
    matches = [
        artifact
        for artifact in bundle.artifacts
        if _lower(artifact.get("sector")) == _lower(sector)
    ]
    return matches[0] if len(matches) == 1 else None


def _repair_state(
    artifact: Mapping[str, Any], ticker: str
) -> dict[str, Any] | None:
    repair = _mapping(_mapping(artifact.get("candidate_selection")).get("data_gap_repair"))
    matches = [
        dict(row)
        for row in repair.get("candidate_states") or []
        if isinstance(row, Mapping) and _upper(row.get("ticker")) == _upper(ticker)
    ]
    return matches[0] if len(matches) == 1 else None


def _stage_result(state: Mapping[str, Any], stage: str) -> dict[str, Any]:
    stage_states = _mapping(state.get("stage_states"))
    summary = _mapping(stage_states.get(stage))
    if stage == "FACTS_AVAILABILITY":
        detailed = _mapping(_mapping(state.get("packet_inputs")).get("facts"))
    elif stage == "PRICE":
        detailed = _mapping(_mapping(state.get("packet_inputs")).get("price"))
    else:
        detailed = {}
    return {**summary, **detailed}


def _row_artifact_proof(
    bundle: _Bundle,
    *,
    family: str,
    row: Mapping[str, Any],
) -> tuple[str, str]:
    sector = _lower(row.get("sector"))
    ticker = _upper(row.get("ticker"))
    artifact = _artifact_by_sector(bundle, sector)
    if artifact is None:
        return NOT_EVALUABLE, f"{sector}:{ticker}:sector-artifact-not-unique"

    if family == "previously_unreviewed":
        matches = [
            candidate
            for candidate in artifact.get("candidate_dispositions") or []
            if isinstance(candidate, Mapping)
            and _upper(candidate.get("ticker")) == ticker
        ]
        if len(matches) != 1:
            return NOT_EVALUABLE, f"{sector}:{ticker}:disposition-not-found"
        expected_state = _upper(row.get("terminal_state"))
        actual_state = _upper(matches[0].get("terminal_state"))
        return (
            (PASS, f"{sector}:{ticker}:disposition:{actual_state}")
            if bool(row.get("visible")) and actual_state == expected_state
            else (FAIL, f"{sector}:{ticker}:disposition-mismatch:{actual_state}")
        )

    if family == "omitted_top_ten":
        selection = _mapping(artifact.get("candidate_selection"))
        persisted = [_upper(item) for item in selection.get("deterministic_prompt_tickers") or []]
        pre_rank = _mapping(selection.get("deterministic_pre_rank"))
        ranked = [
            _upper(item.get("ticker"))
            for item in pre_rank.get("candidates") or []
            if isinstance(item, Mapping)
        ]
        top_tickers = [_upper(item) for item in pre_rank.get("top_tickers") or []]
        packet_tickers = [
            _upper(item.get("ticker"))
            for item in artifact.get("company_packets") or []
            if isinstance(item, Mapping)
        ]
        final_scoped = [
            _upper(item)
            for item in _mapping(artifact.get("final_decision_prompt_context")).get(
                "prompt_scoped_tickers"
            )
            or []
        ]
        try:
            rebuilt = build_competitive_frontier(
                [
                    SectorCompanyFinancialPacket.from_dict(item)
                    for item in artifact.get("company_packets") or []
                    if isinstance(item, Mapping)
                ],
                [
                    SectorExpectedReturnScenario.from_dict(item)
                    for item in artifact.get("expected_return_scenarios") or []
                    if isinstance(item, Mapping)
                ],
                top_n=int(pre_rank.get("top_n") or 25),
                batch_size=int(pre_rank.get("batch_size") or 3),
            )
        except (TypeError, ValueError, KeyError) as exc:
            return FAIL, f"{sector}:{ticker}:rank-rebuild-failed:{type(exc).__name__}"
        rebuilt_candidates = [item.ticker for item in rebuilt.candidates]
        if (
            not persisted
            or not ranked
            or not packet_tickers
            or not isinstance(
                _mapping(artifact.get("final_decision_prompt_context")).get(
                    "prompt_scoped_tickers"
                ),
                list,
            )
        ):
            return NOT_EVALUABLE, f"{sector}:{ticker}:rank-context-not-bound"
        if (
            persisted != top_tickers
            or ranked != rebuilt_candidates
            or set(ranked) != set(packet_tickers)
            or any(item not in packet_tickers for item in final_scoped)
        ):
            return FAIL, f"{sector}:{ticker}:rank-context-inconsistent"
        actual_rank = ranked.index(ticker) + 1 if ticker in ranked else None
        if ticker in persisted:
            return PASS, f"{sector}:{ticker}:current-rank:{actual_rank}:included"
        return FAIL, f"{sector}:{ticker}:rank-context-mismatch:{actual_rank}"

    stage_by_family = {
        "sparse_history": "FACTS_AVAILABILITY",
        "cap_price_losses": "PRICE",
        "cached_annual_filings": "FILINGS",
        "extraction_gaps": "PARSING",
    }
    stage = stage_by_family.get(family)
    state = _repair_state(artifact, ticker)
    if stage is None or state is None:
        return NOT_EVALUABLE, f"{sector}:{ticker}:{stage or 'proof'}-not-found"
    persisted = _stage_result(state, stage)
    outcome = _upper(row.get("outcome") or row.get("status"))
    if family == "cached_annual_filings" and bool(row.get("recognized")):
        outcome = "RECOGNIZED"
    actual_outcome = _upper(persisted.get("outcome"))
    reason = _upper(row.get("reason_code"))
    actual_reason = _upper(persisted.get("reason_code"))
    if outcome == "NEEDS_DATA":
        if actual_outcome == "NEEDS_DATA" and reason and actual_reason == reason:
            return PASS, f"{sector}:{ticker}:{stage}:NEEDS_DATA:{reason}"
        return FAIL, f"{sector}:{ticker}:{stage}:needs-data-mismatch"
    accepted = {
        "sparse_history": {"AVAILABLE", "REPAIRED", "RESOLVED"},
        "cap_price_losses": {"AVAILABLE", "REPAIRED", "RESOLVED"},
        "cached_annual_filings": {"AVAILABLE", "RECOGNIZED", "RESOLVED"},
        "extraction_gaps": {"READABLE", "REPAIRED", "RESOLVED"},
    }[family]
    if outcome not in accepted or actual_outcome not in accepted:
        return FAIL, f"{sector}:{ticker}:{stage}:outcome-mismatch:{actual_outcome}"
    if family == "cap_price_losses":
        snapshot = _mapping(persisted.get("snapshot"))
        price = row.get("price", row.get("price_used"))
        if (
            not _positive_finite(snapshot.get("price"))
            or float(snapshot["price"]) != float(price)
            or str(snapshot.get("source") or "")
            != str(row.get("source") or row.get("price_source") or "")
            or str(snapshot.get("as_of_date") or "")
            != str(row.get("as_of_date") or row.get("price_as_of_date") or "")
        ):
            return FAIL, f"{sector}:{ticker}:PRICE:snapshot-mismatch"
    return PASS, f"{sector}:{ticker}:{stage}:{actual_outcome}"


def _artifact_proof_binding(
    bundle: _Bundle,
    *,
    family: str,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[str, dict[str, Any]]:
    results = [
        _row_artifact_proof(bundle, family=family, row=row) for row in rows
    ]
    statuses = [status for status, _ in results]
    status = (
        FAIL
        if FAIL in statuses
        else NOT_EVALUABLE
        if NOT_EVALUABLE in statuses
        else PASS
    )
    return status, {
        "status": status,
        "proved": statuses.count(PASS),
        "failed": statuses.count(FAIL),
        "not_evaluable": statuses.count(NOT_EVALUABLE),
        "examples": [detail for _, detail in results[:30]],
    }


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _resolve_reference(raw: Any, *, source_path: Path | None) -> Path | None:
    token = str(raw or "").strip()
    if not token:
        return None
    candidate = Path(token).expanduser()
    if candidate.is_absolute():
        return candidate
    if source_path is not None:
        relative = source_path.parent / candidate
        if relative.exists():
            return relative
    return Path.cwd() / candidate


def _load_reference(
    raw: Any,
    *,
    source_path: Path | None,
    label: str,
    errors: list[str],
) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        return dict(raw)
    path = _resolve_reference(raw, source_path=source_path)
    if path is None:
        return {}
    try:
        return _read_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"{label}:{path}:{type(exc).__name__}:{exc}")
        return {}


def _expected_artifact_digest(
    payload: Mapping[str, Any],
    *,
    reference: Any,
    artifact: Mapping[str, Any],
) -> str | None:
    """Return an explicitly declared artifact digest, never an inferred one."""

    reference_text = str(reference or "").strip() if not isinstance(reference, Mapping) else ""
    sector = _lower(artifact.get("sector"))
    run_id = str(artifact.get("run_id") or "").strip()
    candidates: list[str] = []
    for row in payload.get("sector_results") or []:
        if not isinstance(row, Mapping):
            continue
        row_reference = str(row.get("artifact_path") or "").strip()
        row_matches = bool(reference_text and row_reference == reference_text) or (
            sector
            and _lower(row.get("sector")) == sector
            and (not row.get("run_id") or str(row.get("run_id")) == run_id)
        )
        if not row_matches:
            continue
        for field_name in _ARTIFACT_DIGEST_FIELDS:
            token = str(row.get(field_name) or "").strip().lower()
            if token:
                candidates.append(token)
                break
    fixed = _mapping(payload.get("fixed_cohort"))
    for row in fixed.get("sector_artifacts") or []:
        if not isinstance(row, Mapping):
            continue
        row_reference = str(row.get("artifact_path") or "").strip()
        row_matches = bool(reference_text and row_reference == reference_text) or (
            sector
            and _lower(row.get("sector")) == sector
            and (not row.get("run_id") or str(row.get("run_id")) == run_id)
        )
        if not row_matches:
            continue
        token = str(row.get("artifact_sha256") or "").strip().lower()
        if token:
            candidates.append(token)
    return candidates[0] if candidates and len(set(candidates)) == 1 else None


def _verify_artifact_digest(
    payload: Mapping[str, Any],
    *,
    reference: Any,
    artifact: Mapping[str, Any],
    source_path: Path | None,
    label: str,
    errors: list[str],
) -> None:
    expected = _expected_artifact_digest(
        payload,
        reference=reference,
        artifact=artifact,
    )
    if isinstance(reference, Mapping):
        errors.append(f"{label}:INLINE_ARTIFACT_FORBIDDEN")
        return
    if expected is None:
        errors.append(f"{label}:MISSING_SHA256")
        return
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        errors.append(f"{label}:INVALID_SHA256:{expected}")
        return
    path = _resolve_reference(reference, source_path=source_path)
    if path is None:
        return
    try:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        errors.append(f"{label}:{path}:{type(exc).__name__}:{exc}")
        return
    if actual != expected:
        errors.append(f"{label}:{path}:SHA256_MISMATCH:{expected}:{actual}")


def _artifact_references(payload: Mapping[str, Any]) -> list[Any]:
    references: list[Any] = []
    explicit = payload.get("sector_artifacts")
    if isinstance(explicit, list):
        references.extend(item for item in explicit if not isinstance(item, Mapping))
    for row in payload.get("sector_results") or []:
        if not isinstance(row, Mapping):
            continue
        if row.get("artifact_path"):
            references.append(row["artifact_path"])
    return references


def load_acceptance_bundle(
    input_path: str | Path,
    *,
    evidence_path: str | Path | None = None,
) -> _Bundle:
    """Load one bounded local acceptance bundle without changing its inputs."""

    requested = Path(input_path)
    source_path = requested / "benchmark_summary.json" if requested.is_dir() else requested
    errors: list[str] = []
    try:
        payload = _read_json(source_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"input:{source_path}:{type(exc).__name__}:{exc}")
        payload = {}

    overlay: dict[str, Any] = {}
    if evidence_path is not None:
        evidence_source = Path(evidence_path)
        try:
            overlay = _read_json(evidence_source)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"evidence:{evidence_source}:{type(exc).__name__}:{exc}")

    evidence = _mapping(overlay.get("acceptance_evidence"))
    if not evidence:
        evidence = overlay if overlay else _mapping(payload.get("acceptance_evidence"))

    data_plane_raw = overlay.get("data_plane_report", payload.get("data_plane_report"))
    if not data_plane_raw and _lower(payload.get("artifact_type")) == (
        "all_sector_v2_data_plane_reconciliation_report"
    ):
        data_plane_report = payload
    else:
        data_plane_report = _load_reference(
            data_plane_raw,
            source_path=source_path,
            label="data_plane_report",
            errors=errors,
        )

    if isinstance(overlay.get("sector_artifacts"), list):
        errors.append("evidence:sector_artifacts:FORBIDDEN_ARTIFACT_REPLACEMENT")
    artifact_payload = dict(payload)
    explicit_artifacts = artifact_payload.get("sector_artifacts")
    if isinstance(explicit_artifacts, list):
        for index, reference in enumerate(explicit_artifacts):
            if isinstance(reference, Mapping):
                errors.append(
                    f"benchmark:sector_artifacts[{index}]:INLINE_ARTIFACT_FORBIDDEN"
                )
    for index, row in enumerate(artifact_payload.get("sector_results") or []):
        if isinstance(row, Mapping) and isinstance(row.get("artifact"), Mapping):
            errors.append(
                f"benchmark:sector_results[{index}].artifact:INLINE_ARTIFACT_FORBIDDEN"
            )
    if artifact_payload.get("sector") and "candidate_dispositions" in artifact_payload:
        errors.append("benchmark:root:INLINE_SECTOR_ARTIFACT_FORBIDDEN")
    artifacts: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for index, reference in enumerate(_artifact_references(artifact_payload)):
        artifact = _load_reference(
            reference,
            source_path=source_path,
            label=f"sector_artifact[{index}]",
            errors=errors,
        )
        if not artifact:
            continue
        _verify_artifact_digest(
            artifact_payload,
            reference=reference,
            artifact=artifact,
            source_path=source_path,
            label=f"sector_artifact[{index}]",
            errors=errors,
        )
        identity = (
            str(artifact.get("run_id") or ""),
            _lower(artifact.get("sector")),
        )
        if identity in seen:
            continue
        seen.add(identity)
        artifacts.append(artifact)

    return _Bundle(
        payload=payload,
        evidence=evidence,
        data_plane_report=data_plane_report,
        artifacts=tuple(artifacts),
        artifact_models=(),
        artifact_contract_errors=(),
        source_path=source_path,
        input_errors=tuple(errors),
    )


def _evidence_or_summary(bundle: _Bundle, key: str) -> tuple[Any, str | None]:
    if key in bundle.evidence:
        return bundle.evidence[key], f"acceptance_evidence.{key}"
    data_summary = _mapping(bundle.data_plane_report.get("summary"))
    if key in data_summary:
        return data_summary[key], f"data_plane_report.summary.{key}"
    root_summary = _mapping(bundle.payload.get("summary"))
    if key in root_summary:
        return root_summary[key], f"summary.{key}"
    return None, None


def _deserialize_artifacts(
    artifacts: Sequence[Mapping[str, Any]],
) -> tuple[tuple[AutonomousSectorFinancialRunArtifact, ...], tuple[str, ...]]:
    models: list[AutonomousSectorFinancialRunArtifact] = []
    errors: list[str] = []
    for index, raw in enumerate(artifacts):
        sector = _lower(raw.get("sector")) or f"index-{index}"
        try:
            model = AutonomousSectorFinancialRunArtifact.from_dict(dict(raw))
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{sector}:{type(exc).__name__}:{exc}")
            continue
        if model.pipeline_version != SECTOR_PIPELINE_VERSION_V2:
            errors.append(f"{sector}:pipeline_version:{model.pipeline_version}")
            continue
        if model.contract_version != SECTOR_CONTRACT_VERSION_V2:
            errors.append(f"{sector}:contract_version:{model.contract_version}")
            continue
        models.append(model)
    return tuple(models), tuple(errors)


def _sector_result_binding_errors(
    bundle: _Bundle,
    models: Sequence[AutonomousSectorFinancialRunArtifact],
) -> list[str]:
    rows = bundle.payload.get("sector_results")
    if not isinstance(rows, list):
        return ["benchmark:sector_results:missing"]
    normalized_rows = [dict(row) for row in rows if isinstance(row, Mapping)]
    row_sectors = [_lower(row.get("sector")) for row in normalized_rows]
    errors: list[str] = []
    if len(normalized_rows) != len(rows):
        errors.append("benchmark:sector_results:non-object-row")
    duplicate_sectors = sorted(
        sector for sector in set(row_sectors) if sector and row_sectors.count(sector) > 1
    )
    if duplicate_sectors:
        errors.append("benchmark:sector_results:duplicate:" + ",".join(duplicate_sectors))
    by_sector = {_lower(row.get("sector")): row for row in normalized_rows}
    model_sectors = {_lower(model.sector) for model in models}
    if set(by_sector) != model_sectors:
        errors.append(
            "benchmark:sector_results:sector-set-mismatch:"
            f"rows={sorted(by_sector)}:artifacts={sorted(model_sectors)}"
        )
    binding_fields = (
        "run_id",
        "pipeline_version",
        "status",
        "execution_status",
        "decision_status",
        "final_verdict",
        "selected_ticker",
    )
    for model in models:
        sector = _lower(model.sector)
        row = by_sector.get(sector)
        if row is None:
            continue
        expected = {
            "run_id": model.run_id,
            "pipeline_version": model.pipeline_version,
            "status": model.status,
            "execution_status": model.execution_status,
            "decision_status": model.decision_status,
            "final_verdict": model.final_verdict,
            "selected_ticker": model.selected_ticker,
        }
        for field_name in binding_fields:
            if field_name not in row:
                errors.append(f"{sector}:sector_results.{field_name}:missing")
            elif row.get(field_name) != expected[field_name]:
                errors.append(
                    f"{sector}:sector_results.{field_name}:"
                    f"{row.get(field_name)!r}!={expected[field_name]!r}"
                )
    return errors


def _fixed_cohort_artifact_index_errors(bundle: _Bundle) -> list[str]:
    """Validate the producer-owned path/digest index as a closed set."""

    fixed = _mapping(bundle.payload.get("fixed_cohort"))
    errors: list[str] = []
    if fixed.get("schema_version") != _FIXED_COHORT_SCHEMA:
        errors.append(
            "benchmark:fixed_cohort.schema_version:"
            f"{fixed.get('schema_version')!r}!={_FIXED_COHORT_SCHEMA!r}"
        )
    if fixed.get("trusted_manifest_policy") != _TRUSTED_MANIFEST_POLICY:
        errors.append(
            "benchmark:fixed_cohort.trusted_manifest_policy:"
            f"{fixed.get('trusted_manifest_policy')!r}!={_TRUSTED_MANIFEST_POLICY!r}"
        )

    raw_index = fixed.get("sector_artifacts")
    if not isinstance(raw_index, list):
        return [*errors, "benchmark:fixed_cohort.sector_artifacts:missing"]
    index_rows = [dict(row) for row in raw_index if isinstance(row, Mapping)]
    if len(index_rows) != len(raw_index):
        errors.append("benchmark:fixed_cohort.sector_artifacts:non-object-row")

    def index_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
        return (
            _lower(row.get("sector")),
            str(row.get("run_id") or "").strip(),
            str(row.get("artifact_path") or "").strip(),
        )

    index_keys = [index_key(row) for row in index_rows]
    for index, (row, key) in enumerate(zip(index_rows, index_keys, strict=True)):
        sector, run_id, artifact_path = key
        if not sector or not run_id or not artifact_path:
            errors.append(f"benchmark:fixed_cohort.sector_artifacts[{index}]:identity-missing")
        digest = str(row.get("artifact_sha256") or "").strip().lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            errors.append(
                f"benchmark:fixed_cohort.sector_artifacts[{index}].artifact_sha256:invalid"
            )
    duplicate_keys = sorted(key for key in set(index_keys) if index_keys.count(key) > 1)
    if duplicate_keys:
        errors.append(f"benchmark:fixed_cohort.sector_artifacts:duplicate:{duplicate_keys}")

    raw_results = bundle.payload.get("sector_results")
    if not isinstance(raw_results, list):
        return [*errors, "benchmark:sector_results:missing-for-fixed-cohort-index"]
    result_rows = [dict(row) for row in raw_results if isinstance(row, Mapping)]
    result_keys = [index_key(row) for row in result_rows]
    if len(result_rows) != len(raw_results):
        errors.append("benchmark:sector_results:non-object-row-for-fixed-cohort-index")
    for index, row in enumerate(result_rows):
        if isinstance(row.get("artifact"), Mapping):
            errors.append(f"benchmark:sector_results[{index}].artifact:inline-forbidden")
        if not str(row.get("artifact_path") or "").strip():
            errors.append(f"benchmark:sector_results[{index}].artifact_path:missing")
        digest = str(row.get("artifact_sha256") or "").strip().lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            errors.append(f"benchmark:sector_results[{index}].artifact_sha256:invalid")

    if sorted(index_keys) != sorted(result_keys):
        errors.append(
            "benchmark:fixed_cohort.sector_artifacts:index-set-mismatch:"
            f"fixed={sorted(index_keys)}:results={sorted(result_keys)}"
        )
    index_by_key = {key: row for key, row in zip(index_keys, index_rows, strict=True)}
    for key, result in zip(result_keys, result_rows, strict=True):
        indexed = index_by_key.get(key)
        if indexed is None:
            continue
        result_digest = str(result.get("artifact_sha256") or "").strip().lower()
        indexed_digest = str(indexed.get("artifact_sha256") or "").strip().lower()
        if result_digest != indexed_digest:
            errors.append(
                "benchmark:fixed_cohort.sector_artifacts:digest-mismatch:"
                f"{key}:{indexed_digest}!={result_digest}"
            )
    return errors


def _artifact_binding_errors(
    bundle: _Bundle,
    models: Sequence[AutonomousSectorFinancialRunArtifact],
) -> list[str]:
    payload = bundle.payload
    errors: list[str] = []
    required_root = {
        "artifact_type": "autonomous_sector_benchmark_v2",
        "pipeline_version": SECTOR_PIPELINE_VERSION_V2,
    }
    for field_name, expected in required_root.items():
        actual = payload.get(field_name)
        if actual != expected:
            errors.append(f"benchmark:{field_name}:{actual!r}!={expected!r}")
    for field_name in ("run_id", "as_of_date", "market_cap_focus", "objective"):
        if not str(payload.get(field_name) or "").strip():
            errors.append(f"benchmark:{field_name}:missing")

    model_sectors = [_lower(model.sector) for model in models]
    if len(model_sectors) != len(set(model_sectors)):
        errors.append("artifacts:duplicate-sector")
    model_run_ids = [str(model.run_id or "").strip() for model in models]
    if len(model_run_ids) != len(set(model_run_ids)):
        errors.append("artifacts:duplicate-run-id")
    declared_sectors = payload.get("sectors")
    if not isinstance(declared_sectors, list):
        errors.append("benchmark:sectors:missing")
    else:
        normalized_declared = [_lower(value) for value in declared_sectors]
        if (
            len(normalized_declared) != len(set(normalized_declared))
            or set(normalized_declared) != set(model_sectors)
        ):
            errors.append(
                "benchmark:sectors:mismatch:"
                f"declared={sorted(normalized_declared)}:artifacts={sorted(model_sectors)}"
            )

    for model in models:
        sector = _lower(model.sector)
        for field_name in ("as_of_date", "market_cap_focus", "objective"):
            declared = payload.get(field_name)
            if declared is not None and getattr(model, field_name) != declared:
                errors.append(
                    f"{sector}:{field_name}:{getattr(model, field_name)!r}!={declared!r}"
                )
    errors.extend(_sector_result_binding_errors(bundle, models))
    errors.extend(_fixed_cohort_artifact_index_errors(bundle))
    return errors


def _check_artifact_contract_and_binding(bundle: _Bundle, report: _Report) -> None:
    if not bundle.artifacts:
        report.unavailable(
            "artifact_contract_and_binding",
            expected="full source-bound autonomous_sector_financial_run_v2 artifacts",
            missing_fields=["sector_artifacts or sector_results[].artifact_path"],
        )
        return
    binding_errors = _artifact_binding_errors(bundle, bundle.artifact_models)
    errors = [*bundle.artifact_contract_errors, *binding_errors]
    passed = len(bundle.artifact_models) == len(bundle.artifacts) and not errors
    report.add(
        "artifact_contract_and_binding",
        PASS if passed else FAIL,
        expected=(
            "every sector artifact fully deserializes as v2 and is bound to the "
            "benchmark run, as-of date, cohort fields, and sector-result identity"
        ),
        actual={
            "raw_artifacts": len(bundle.artifacts),
            "validated_artifacts": len(bundle.artifact_models),
            "contract_or_binding_errors": len(errors),
        },
        evidence_source="benchmark root + sector_artifacts",
        details={"error_examples": errors[:40]},
    )


def _flatten_security_row(
    row: Mapping[str, Any],
    *,
    sector: str | None = None,
) -> dict[str, Any]:
    disposition = _mapping(row.get("disposition"))
    flattened = {**dict(row), **disposition}
    if sector and not flattened.get("sector"):
        flattened["sector"] = sector
    return flattened


def _security_slots(bundle: _Bundle) -> tuple[list[dict[str, Any]] | None, str | None]:
    if not bundle.artifact_models:
        return None, None
    slots: list[dict[str, Any]] = []
    for artifact in bundle.artifact_models:
        sector = _lower(artifact.sector)
        for disposition in artifact.candidate_dispositions:
            slots.append(_flatten_security_row(disposition.to_dict(), sector=sector))
    return slots, "validated sector_artifacts[].candidate_dispositions"


def _check_input_integrity(bundle: _Bundle, report: _Report) -> None:
    report.add(
        "input_integrity",
        PASS if not bundle.input_errors else FAIL,
        expected="all explicit local JSON inputs readable",
        actual={"errors": len(bundle.input_errors)},
        details={"error_examples": list(bundle.input_errors[:20])},
    )


def _check_sector_coverage(
    bundle: _Bundle, report: _Report, thresholds: AcceptanceThresholds
) -> None:
    sectors = {
        _lower(item.sector) for item in bundle.artifact_models if _lower(item.sector)
    }
    if not sectors:
        sectors = {
            _lower(item.get("sector"))
            for item in bundle.payload.get("sector_results") or []
            if isinstance(item, Mapping) and _lower(item.get("sector"))
        }
    if not sectors:
        expected = (
            {
                "sector_count": thresholds.sector_count,
                "sector_labels": list(thresholds.expected_sector_labels),
            }
            if thresholds.expected_sector_labels is not None
            else thresholds.sector_count
        )
        report.unavailable(
            "sector_coverage",
            expected=expected,
            missing_fields=["sector_artifacts or sector_results"],
        )
        return
    expected_labels = thresholds.expected_sector_labels
    if expected_labels is None:
        report.add(
            "sector_coverage",
            PASS if len(sectors) == thresholds.sector_count else FAIL,
            expected=thresholds.sector_count,
            actual=len(sectors),
            details={"sectors": sorted(sectors)},
        )
        return
    expected_set = set(expected_labels)
    missing = sorted(expected_set - sectors)
    unexpected = sorted(sectors - expected_set)
    report.add(
        "sector_coverage",
        (
            PASS
            if len(sectors) == thresholds.sector_count
            and len(expected_set) == thresholds.sector_count
            and not missing
            and not unexpected
            else FAIL
        ),
        expected={
            "sector_count": thresholds.sector_count,
            "sector_labels": list(expected_labels),
        },
        actual={"sector_count": len(sectors), "sector_labels": sorted(sectors)},
        details={
            "missing_sector_labels": missing,
            "unexpected_sector_labels": unexpected,
        },
    )


def _check_security_reconciliation(
    slots: list[dict[str, Any]] | None,
    source: str | None,
    report: _Report,
    thresholds: AcceptanceThresholds,
) -> None:
    if slots is None:
        report.unavailable(
            "security_slot_reconciliation",
            expected={"security_slots": thresholds.security_slots, "unreconciled": 0},
            missing_fields=["security_slots or candidate_dispositions"],
        )
        return
    keys: list[tuple[str, str] | None] = [_slot_key(row) for row in slots]
    duplicates = len([key for key in keys if key is not None]) - len(
        {key for key in keys if key is not None}
    )
    invalid: list[str] = []
    for index, (row, key) in enumerate(zip(slots, keys, strict=True)):
        scope = _upper(row.get("scope_status"))
        terminal = _upper(row.get("terminal_state"))
        reasons = [
            str(value).strip() for value in row.get("reason_codes") or [] if str(value).strip()
        ]
        valid = (
            key is not None
            and scope in VALID_SCOPE_STATUSES
            and terminal in VALID_TERMINAL_STATES
            and not (scope == "IN_SCOPE" and terminal == "OUT_OF_SCOPE")
            and not (scope == "OUT_OF_SCOPE" and (terminal != "OUT_OF_SCOPE" or not reasons))
        )
        if not valid:
            invalid.append(f"{key or ('row', str(index))}:{scope}:{terminal}")
    unique_count = len({key for key in keys if key is not None})
    passed = unique_count == thresholds.security_slots and duplicates == 0 and not invalid
    report.add(
        "security_slot_reconciliation",
        PASS if passed else FAIL,
        expected={"security_slots": thresholds.security_slots, "unreconciled": 0},
        actual={
            "rows": len(slots),
            "unique_security_slots": unique_count,
            "duplicates": duplicates,
            "invalid_or_unreconciled": len(invalid),
        },
        evidence_source=source,
        details={"invalid_examples": invalid[:30]},
    )


def _check_security_issuer_counts(
    bundle: _Bundle,
    slots: list[dict[str, Any]] | None,
    report: _Report,
    thresholds: AcceptanceThresholds,
) -> None:
    fixed = _mapping(bundle.payload.get("fixed_cohort"))
    declared_security = fixed.get("security_count")
    declared_issuer = fixed.get("issuer_count")
    derived_security: int | None = None
    derived_issuers: int | None = None
    unresolved = 0
    if slots is None:
        report.unavailable(
            "security_and_issuer_counts",
            expected="source-derived security and deduplicated issuer counts",
            missing_fields=["validated sector candidate_dispositions"],
        )
        return
    if slots is not None:
        keys = {_slot_key(row) for row in slots if _slot_key(row) is not None}
        issuers = {_issuer_key(row) for row in slots if _issuer_key(row) is not None}
        derived_security = len(keys)
        derived_issuers = len(issuers)
        unresolved = sum(1 for row in slots if _issuer_key(row) is None)
    if declared_security is None:
        report.unavailable(
            "security_and_issuer_counts",
            expected="separate security and deduplicated issuer counts",
            missing_fields=["benchmark.fixed_cohort.security_count"],
        )
        return
    if declared_issuer is None:
        report.unavailable(
            "security_and_issuer_counts",
            expected="separate security and deduplicated issuer counts",
            missing_fields=["benchmark.fixed_cohort.issuer_count"],
        )
        return
    if (
        not isinstance(declared_security, int)
        or isinstance(declared_security, bool)
        or not isinstance(declared_issuer, int)
        or isinstance(declared_issuer, bool)
    ):
        report.add(
            "security_and_issuer_counts",
            FAIL,
            expected="integer benchmark fixed-cohort counts",
            actual={
                "security_count": declared_security,
                "issuer_count": declared_issuer,
            },
            evidence_source="benchmark.fixed_cohort",
        )
        return
    security_count = int(
        declared_security if declared_security is not None else derived_security or 0
    )
    issuer_count = int(declared_issuer if declared_issuer is not None else derived_issuers or 0)
    mismatch = (
        derived_security is not None
        and declared_security is not None
        and security_count != derived_security
    ) or (
        derived_issuers is not None
        and declared_issuer is not None
        and issuer_count != derived_issuers
    )
    passed = (
        security_count == thresholds.security_slots
        and 0 < issuer_count <= security_count
        and unresolved == 0
        and not mismatch
    )
    report.add(
        "security_and_issuer_counts",
        PASS if passed else FAIL,
        expected={
            "security_count": thresholds.security_slots,
            "issuer_count": "separate and deduplicated",
        },
        actual={
            "security_count": security_count,
            "issuer_count": issuer_count,
            "derived_security_count": derived_security,
            "derived_issuer_count": derived_issuers,
            "unresolved_issuer_identity": unresolved,
            "declared_derived_mismatch": mismatch,
        },
    )


def _check_sparse_history(
    bundle: _Bundle, report: _Report, thresholds: AcceptanceThresholds
) -> None:
    value, source = _evidence_or_summary(bundle, "sparse_history")
    rows = _rows(value)
    if rows is not None:
        repaired = 0
        precise = 0
        unaccounted: list[str] = []
        for row in rows:
            outcome = _upper(row.get("outcome") or row.get("status") or row.get("terminal_state"))
            if outcome in {"AVAILABLE", "REPAIRED", "RESOLVED"}:
                repaired += 1
            elif outcome == "NEEDS_DATA" and _precise_reason(row.get("reason_code")):
                precise += 1
            else:
                unaccounted.append(_upper(row.get("ticker")) or "UNKNOWN")
        passed = len(rows) == thresholds.sparse_history and not unaccounted
        actual = {
            "total": len(rows),
            "repaired": repaired,
            "precise_needs_data": precise,
            "unaccounted": len(unaccounted),
        }
        details = {"unaccounted_examples": unaccounted[:30]}
        binding_status, binding = _cohort_manifest_binding(
            bundle,
            thresholds=thresholds,
            family="sparse_history",
            rows=rows,
        )
        actual["cohort_manifest"] = binding
        proof_status, proof = _artifact_proof_binding(
            bundle, family="sparse_history", rows=rows
        )
        actual["artifact_proof"] = proof
    elif isinstance(value, Mapping):
        report.add(
            "sparse_history_reconciliation",
            NOT_EVALUABLE,
            expected={"total": thresholds.sparse_history, "unaccounted": 0},
            actual=dict(value),
            evidence_source=source,
            details={"reason": "summary counters do not identify the underlying security rows"},
        )
        return
    else:
        report.unavailable(
            "sparse_history_reconciliation",
            expected={"total": thresholds.sparse_history, "unaccounted": 0},
            missing_fields=["acceptance_evidence.sparse_history or data-plane summary"],
        )
        return
    report.add(
        "sparse_history_reconciliation",
        _manifest_bound_status(
            content_passed=passed,
            binding_status=binding_status,
            proof_status=proof_status,
        ),
        expected={
            "total": thresholds.sparse_history,
            "repaired_or_precise_needs_data": thresholds.sparse_history,
        },
        actual=actual,
        evidence_source=source,
        details=details,
    )


def _check_cap_price_losses(
    bundle: _Bundle, report: _Report, thresholds: AcceptanceThresholds
) -> None:
    value, source = _evidence_or_summary(bundle, "cap_price_losses")
    rows = _rows(value)
    if rows is not None:
        recovered = 0
        precise = 0
        unexplained: list[str] = []
        for row in rows:
            outcome = _upper(row.get("outcome") or row.get("status"))
            price = row.get("price", row.get("price_used"))
            resolved = (
                outcome in {"AVAILABLE", "RESOLVED", "REPAIRED"}
                and _positive_finite(price)
                and bool(row.get("source") or row.get("price_source"))
                and bool(row.get("as_of_date") or row.get("price_as_of_date"))
            )
            needs_data = (
                outcome == "NEEDS_DATA"
                and _precise_reason(row.get("reason_code"))
                and isinstance(row.get("source_attempts"), list)
                and bool(row.get("source_attempts"))
            )
            if resolved:
                recovered += 1
            elif needs_data:
                precise += 1
            else:
                unexplained.append(_upper(row.get("ticker")) or "UNKNOWN")
        passed = len(rows) == thresholds.cap_price_losses and not unexplained
        actual = {
            "total": len(rows),
            "recovered": recovered,
            "source_exhausted_precise_needs_data": precise,
            "unexplained": len(unexplained),
        }
        details = {"unexplained_examples": unexplained[:30]}
        binding_status, binding = _cohort_manifest_binding(
            bundle,
            thresholds=thresholds,
            family="cap_price_losses",
            rows=rows,
        )
        actual["cohort_manifest"] = binding
        proof_status, proof = _artifact_proof_binding(
            bundle, family="cap_price_losses", rows=rows
        )
        actual["artifact_proof"] = proof
    elif isinstance(value, Mapping):
        report.add(
            "cap_price_loss_reconciliation",
            NOT_EVALUABLE,
            expected={"historical_losses": thresholds.cap_price_losses, "unexplained": 0},
            actual=dict(value),
            evidence_source=source,
            details={"reason": "summary counters do not expose price provenance per security"},
        )
        return
    else:
        report.unavailable(
            "cap_price_loss_reconciliation",
            expected={"historical_losses": thresholds.cap_price_losses, "unexplained": 0},
            missing_fields=["acceptance_evidence.cap_price_losses or data-plane summary"],
        )
        return
    report.add(
        "cap_price_loss_reconciliation",
        _manifest_bound_status(
            content_passed=passed,
            binding_status=binding_status,
            proof_status=proof_status,
        ),
        expected={"historical_losses": thresholds.cap_price_losses, "unexplained": 0},
        actual=actual,
        evidence_source=source,
        details=details,
    )


def _check_cached_filings(
    bundle: _Bundle, report: _Report, thresholds: AcceptanceThresholds
) -> None:
    value, source = _evidence_or_summary(bundle, "cached_annual_filings")
    rows = _rows(value)
    if rows is not None:
        unrecognized = [
            _upper(row.get("ticker")) or "UNKNOWN"
            for row in rows
            if not bool(row.get("recognized"))
            and _upper(row.get("outcome") or row.get("status")) not in {"AVAILABLE", "RECOGNIZED"}
        ]
        passed = len(rows) == thresholds.cached_annual_filings and not unrecognized
        actual = {
            "total": len(rows),
            "recognized": len(rows) - len(unrecognized),
            "unrecognized": len(unrecognized),
        }
        details = {"unrecognized_examples": unrecognized[:30]}
        binding_status, binding = _cohort_manifest_binding(
            bundle,
            thresholds=thresholds,
            family="cached_annual_filings",
            rows=rows,
        )
        actual["cohort_manifest"] = binding
        proof_status, proof = _artifact_proof_binding(
            bundle, family="cached_annual_filings", rows=rows
        )
        actual["artifact_proof"] = proof
    elif isinstance(value, Mapping):
        report.add(
            "cached_annual_filing_recognition",
            NOT_EVALUABLE,
            expected={"cached_annual_filings": thresholds.cached_annual_filings, "unrecognized": 0},
            actual=dict(value),
            evidence_source=source,
            details={"reason": "summary counters do not identify filing/accession evidence"},
        )
        return
    else:
        report.unavailable(
            "cached_annual_filing_recognition",
            expected={"cached_annual_filings": thresholds.cached_annual_filings, "unrecognized": 0},
            missing_fields=["acceptance_evidence.cached_annual_filings or data-plane summary"],
        )
        return
    report.add(
        "cached_annual_filing_recognition",
        _manifest_bound_status(
            content_passed=passed,
            binding_status=binding_status,
            proof_status=proof_status,
        ),
        expected={"cached_annual_filings": thresholds.cached_annual_filings, "unrecognized": 0},
        actual=actual,
        evidence_source=source,
        details=details,
    )


def _check_extraction_gaps(
    bundle: _Bundle, report: _Report, thresholds: AcceptanceThresholds
) -> None:
    value, source = _evidence_or_summary(bundle, "extraction_gaps")
    if value is None:
        value, source = _evidence_or_summary(bundle, "remaining_extraction_gaps")
    rows = _rows(value)
    if rows is not None:
        repaired = 0
        precise = 0
        unaccounted: list[str] = []
        for row in rows:
            outcome = _upper(row.get("outcome") or row.get("status"))
            if outcome in {"READABLE", "REPAIRED", "RESOLVED"}:
                repaired += 1
            elif outcome == "NEEDS_DATA" and _precise_reason(row.get("reason_code")):
                precise += 1
            else:
                unaccounted.append(_upper(row.get("ticker")) or "UNKNOWN")
        passed = len(rows) == thresholds.extraction_gaps and not unaccounted
        actual = {
            "total": len(rows),
            "repaired": repaired,
            "precise_extraction_gap": precise,
            "unaccounted": len(unaccounted),
        }
        details = {"unaccounted_examples": unaccounted[:30]}
        binding_status, binding = _cohort_manifest_binding(
            bundle,
            thresholds=thresholds,
            family="extraction_gaps",
            rows=rows,
        )
        actual["cohort_manifest"] = binding
        proof_status, proof = _artifact_proof_binding(
            bundle, family="extraction_gaps", rows=rows
        )
        actual["artifact_proof"] = proof
    elif isinstance(value, Mapping):
        report.add(
            "extraction_gap_classification",
            NOT_EVALUABLE,
            expected={"historical_gaps": thresholds.extraction_gaps, "unaccounted": 0},
            actual=dict(value),
            evidence_source=source,
            details={"reason": "summary counters do not identify extraction outcomes per filing"},
        )
        return
    else:
        report.unavailable(
            "extraction_gap_classification",
            expected={"historical_gaps": thresholds.extraction_gaps, "unaccounted": 0},
            missing_fields=["acceptance_evidence.extraction_gaps or data-plane summary"],
        )
        return
    report.add(
        "extraction_gap_classification",
        _manifest_bound_status(
            content_passed=passed,
            binding_status=binding_status,
            proof_status=proof_status,
        ),
        expected={
            "historical_gaps": thresholds.extraction_gaps,
            "repaired_or_precise": thresholds.extraction_gaps,
        },
        actual=actual,
        evidence_source=source,
        details=details,
    )


def _check_visibility(bundle: _Bundle, report: _Report, thresholds: AcceptanceThresholds) -> None:
    value = bundle.evidence.get("previously_unreviewed")
    source = "acceptance_evidence.previously_unreviewed"
    rows = _rows(value)
    if rows is not None:
        hidden = [
            _upper(row.get("ticker")) or "UNKNOWN"
            for row in rows
            if not bool(row.get("visible"))
            or _upper(row.get("terminal_state")) not in VALID_TERMINAL_STATES
        ]
        passed = len(rows) == thresholds.previously_unreviewed and not hidden
        actual = {"total": len(rows), "visible": len(rows) - len(hidden), "hidden": len(hidden)}
        details = {"hidden_examples": hidden[:30]}
        binding_status, binding = _cohort_manifest_binding(
            bundle,
            thresholds=thresholds,
            family="previously_unreviewed",
            rows=rows,
        )
        actual["cohort_manifest"] = binding
        proof_status, proof = _artifact_proof_binding(
            bundle, family="previously_unreviewed", rows=rows
        )
        actual["artifact_proof"] = proof
    elif isinstance(value, Mapping):
        report.add(
            "unreviewed_disposition_visibility",
            NOT_EVALUABLE,
            expected={
                "previously_unreviewed": thresholds.previously_unreviewed,
                "visible": thresholds.previously_unreviewed,
            },
            actual=dict(value),
            evidence_source=source,
            details={"reason": "summary counters do not identify the visible disposition rows"},
        )
        return
    else:
        report.unavailable(
            "unreviewed_disposition_visibility",
            expected={
                "previously_unreviewed": thresholds.previously_unreviewed,
                "visible": thresholds.previously_unreviewed,
            },
            missing_fields=["acceptance_evidence.previously_unreviewed"],
        )
        return
    report.add(
        "unreviewed_disposition_visibility",
        _manifest_bound_status(
            content_passed=passed,
            binding_status=binding_status,
            proof_status=proof_status,
        ),
        expected={
            "previously_unreviewed": thresholds.previously_unreviewed,
            "visible": thresholds.previously_unreviewed,
        },
        actual=actual,
        evidence_source=source,
        details=details,
    )


def _check_top_ten_context(
    bundle: _Bundle, report: _Report, thresholds: AcceptanceThresholds
) -> None:
    value = bundle.evidence.get("omitted_top_ten")
    rows = _rows(value)
    if rows is not None:
        passed = len(rows) == thresholds.omitted_top_ten
        actual = {"total": len(rows)}
        details: dict[str, Any] = {}
        binding_status, binding = _cohort_manifest_binding(
            bundle,
            thresholds=thresholds,
            family="omitted_top_ten",
            rows=rows,
        )
        actual["cohort_manifest"] = binding
        proof_status, proof = _artifact_proof_binding(
            bundle, family="omitted_top_ten", rows=rows
        )
        actual["artifact_proof"] = proof
        actual["included"] = proof["proved"]
        actual["missing"] = proof["failed"] + proof["not_evaluable"]
    elif isinstance(value, Mapping):
        report.add(
            "omitted_top_ten_prompt_context",
            NOT_EVALUABLE,
            expected={
                "historically_omitted": thresholds.omitted_top_ten,
                "included": thresholds.omitted_top_ten,
            },
            actual=dict(value),
            evidence_source="acceptance_evidence.omitted_top_ten",
            details={"reason": "summary counters do not identify ranked prompt-context rows"},
        )
        return
    else:
        report.unavailable(
            "omitted_top_ten_prompt_context",
            expected={
                "historically_omitted": thresholds.omitted_top_ten,
                "included": thresholds.omitted_top_ten,
            },
            missing_fields=["acceptance_evidence.omitted_top_ten"],
        )
        return
    report.add(
        "omitted_top_ten_prompt_context",
        _manifest_bound_status(
            content_passed=passed,
            binding_status=binding_status,
            proof_status=proof_status,
        ),
        expected={
            "historically_omitted": thresholds.omitted_top_ten,
            "included": thresholds.omitted_top_ten,
        },
        actual=actual,
        evidence_source="acceptance_evidence.omitted_top_ten",
        details=details,
    )


def _going_concern_evaluation(
    payload: Mapping[str, Any], ticker: str
) -> dict[str, Any] | None:
    evaluations = payload.get("gate_evaluations")
    if not isinstance(evaluations, list):
        evaluations = _mapping(payload.get("screen_result")).get("gate_evaluations")
    matches = [
        dict(row)
        for row in evaluations or []
        if isinstance(row, Mapping) and _upper(row.get("rule_id")) == "GOING_CONCERN"
    ]
    return matches[0] if len(matches) == 1 else None


def _going_concern_false_provenance_projection(
    row: Mapping[str, Any],
) -> dict[str, str] | None:
    canonical_id = _canonical_identifier("going_concern", row)
    projection = {
        "canonical_id": str(canonical_id or ""),
        "historical_context_revision": str(
            row.get("matched_context_revision")
            or row.get("fixture_content_revision")
            or ""
        ).strip().lower(),
        "filing_content_revision": str(
            row.get("filing_content_revision") or ""
        ).strip().lower(),
        "accession": str(row.get("accession") or "").strip(),
        "form_type": _upper(row.get("form_type")),
        "filing_date": str(row.get("filing_date") or "").strip(),
        "issuer_cik": _normalized_cik(row.get("issuer_cik")),
        "source_url": str(
            row.get("source_url") or row.get("primary_doc_url") or ""
        ).strip(),
        "evidence_ref_id": str(row.get("evidence_ref_id") or "").strip(),
    }
    sha_fields = {
        projection["historical_context_revision"],
        projection["filing_content_revision"],
    }
    if (
        any(not value for value in projection.values())
        or any(
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in sha_fields
        )
    ):
        return None
    return projection


def going_concern_false_provenance_manifest_sha256(
    rows: Sequence[Mapping[str, Any]],
) -> str | None:
    """Digest normalized, complete filing provenance for trusted pinning."""

    projections = [_going_concern_false_provenance_projection(row) for row in rows]
    if any(projection is None for projection in projections):
        return None
    serialized = [
        json.dumps(projection, sort_keys=True, separators=(",", ":"))
        for projection in projections
        if projection is not None
    ]
    if len(serialized) != len(set(serialized)):
        return None
    return _manifest_sha256(serialized)


def _going_concern_semantic_provenance_binding(
    rows: Sequence[Mapping[str, Any]],
    *,
    thresholds: AcceptanceThresholds,
) -> tuple[str, dict[str, Any]]:
    family = "going_concern_false_provenance"
    trusted = dict(thresholds.trusted_semantic_manifest_sha256)
    expected = str(trusted.get(family) or "").strip().lower()
    actual = going_concern_false_provenance_manifest_sha256(rows)
    details = {
        "family": family,
        "trusted_semantic_manifest_sha256": expected or None,
        "actual_semantic_manifest_sha256": actual,
        "provenance_row_count": len(rows),
    }
    if not expected:
        details["reason"] = "no independently trusted semantic provenance manifest is configured"
        return NOT_EVALUABLE, details
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        details["invalid_expected_manifest"] = True
        return FAIL, details
    if actual is None:
        details["reason"] = "normalized complete filing provenance is unavailable"
        return FAIL, details
    return (PASS if actual == expected else FAIL), details


def _canonical_going_concern_assertion_projection(
    assertion: Mapping[str, Any],
    *,
    content_revision_override: str | None = None,
) -> dict[str, Any]:
    """Normalize one detector assertion for runtime-versus-artifact equality."""

    subject_detail = assertion.get("subject_detail")
    distress = assertion.get("corroborating_distress")
    return {
        "subject": _upper(assertion.get("subject")),
        "subject_detail": (
            " ".join(subject_detail.split())
            if isinstance(subject_detail, str)
            else None
        ),
        "assertion_mode": _upper(assertion.get("assertion_mode")),
        "blockable": assertion.get("blockable"),
        "accession": str(assertion.get("accession") or "").strip(),
        "form_type": _upper(assertion.get("form_type")),
        "filing_date": str(assertion.get("filing_date") or "").strip(),
        "section": str(assertion.get("section") or "").strip(),
        "excerpt": " ".join(str(assertion.get("excerpt") or "").split()),
        "corroborating_distress": (
            [str(item).strip() for item in distress]
            if isinstance(distress, list)
            else None
        ),
        "issuer_cik": _normalized_cik(assertion.get("issuer_cik")),
        "source_url": str(assertion.get("source_url") or "").strip(),
        "content_revision": (
            str(content_revision_override).strip().lower()
            if content_revision_override is not None
            else str(assertion.get("content_revision") or "").strip().lower()
        ),
    }


def _canonical_going_concern_gate_projection(
    evaluation: Mapping[str, Any],
    *,
    content_revision_override: str | None = None,
) -> dict[str, Any]:
    """Normalize the complete detector-owned gate output for exact replay."""

    observed = _mapping(evaluation.get("observed_value"))
    raw_assertions = observed.get("assertions")
    assertions = (
        [
            _canonical_going_concern_assertion_projection(
                assertion,
                content_revision_override=content_revision_override,
            )
            for assertion in raw_assertions
            if isinstance(assertion, Mapping)
        ]
        if isinstance(raw_assertions, list)
        else None
    )
    return {
        "status": _upper(evaluation.get("status")),
        "reason_code": _upper(evaluation.get("reason_code")) or None,
        "evidence_ref_id": str(evaluation.get("evidence_ref_id") or "").strip(),
        "evidence_url": str(evaluation.get("evidence_url") or "").strip(),
        "observed_value": {
            "assertion": _upper(observed.get("assertion")),
            "assertions": assertions,
            "accession": str(observed.get("accession") or "").strip(),
            "form_type": _upper(observed.get("form_type")),
            "filing_date": str(observed.get("filing_date") or "").strip(),
            "issuer_cik": _normalized_cik(observed.get("issuer_cik")),
            "source_url": str(observed.get("source_url") or "").strip(),
            "content_revision": (
                str(content_revision_override).strip().lower()
                if content_revision_override is not None
                else str(observed.get("content_revision") or "").strip().lower()
            ),
        },
    }


def _going_concern_false_proof(
    bundle: _Bundle,
    row: Mapping[str, Any],
    *,
    frozen_fixture: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    sector = _lower(row.get("sector"))
    ticker = _upper(row.get("ticker"))
    artifact = _artifact_by_sector(bundle, sector)
    if artifact is None:
        return NOT_EVALUABLE, f"{sector}:{ticker}:sector-artifact-not-unique"
    dispositions = [
        item
        for item in artifact.get("candidate_dispositions") or []
        if isinstance(item, Mapping) and _upper(item.get("ticker")) == ticker
    ]
    if len(dispositions) != 1:
        return NOT_EVALUABLE, f"{sector}:{ticker}:disposition-not-found"
    disposition_eval = _going_concern_evaluation(
        _mapping(dispositions[0].get("screen_result")), ticker
    )
    structural = _mapping(
        _mapping(_mapping(artifact.get("candidate_selection")).get("structural_gate_results")).get(
            ticker
        )
    )
    structural_eval = _going_concern_evaluation(structural, ticker)
    if disposition_eval is None or structural_eval is None:
        return NOT_EVALUABLE, f"{sector}:{ticker}:going-concern-evaluation-missing"
    expected_fixture_revision = str(
        row.get("fixture_content_revision")
        or row.get("matched_context_revision")
        or row.get("content_revision")
        or ""
    ).strip().lower()
    expected_filing_revision = str(
        row.get("filing_content_revision") or row.get("content_revision") or ""
    ).strip().lower()
    expected_accession = str(row.get("accession") or "").strip()
    expected_form = _upper(row.get("form_type"))
    expected_filing_date = str(row.get("filing_date") or "").strip()
    expected_issuer_cik = _normalized_cik(row.get("issuer_cik"))
    expected_source_url = str(
        row.get("source_url") or row.get("primary_doc_url") or ""
    ).strip()
    expected_ref = str(row.get("evidence_ref_id") or "").strip()
    if (
        expected_fixture_revision != _sha256_text(row.get("matched_context"))
        or len(expected_filing_revision) != 64
        or any(char not in "0123456789abcdef" for char in expected_filing_revision)
        or not expected_accession
        or not expected_form
        or not expected_filing_date
        or not expected_issuer_cik
        or not expected_source_url
        or expected_ref != f"sec-filing:{ticker}:{expected_accession}"
    ):
        return FAIL, f"{sector}:{ticker}:fixture-content-binding-invalid"
    if frozen_fixture is not None and (
        sector != _lower(frozen_fixture.get("sector"))
        or str(row.get("gate_scorecard_as_of") or "")
        != str(frozen_fixture.get("gate_scorecard_as_of") or "")
        or expected_fixture_revision
        != str(frozen_fixture.get("content_revision") or "")
    ):
        return FAIL, f"{sector}:{ticker}:frozen-fixture-revision-mismatch"
    replayed_gate = evaluate_going_concern_filing_text(
        ticker,
        str(row.get("matched_context") or ""),
        accession=expected_accession,
        form_type=expected_form,
        filing_date=expected_filing_date,
        issuer_cik=expected_issuer_cik,
        source_url=expected_source_url,
    )
    replayed_projection = _canonical_going_concern_gate_projection(
        replayed_gate,
        content_revision_override=expected_filing_revision,
    )
    for label, evaluation in (
        ("disposition", disposition_eval),
        ("structural", structural_eval),
    ):
        observed = _mapping(evaluation.get("observed_value"))
        raw_assertions = observed.get("assertions")
        if not isinstance(raw_assertions, list):
            return FAIL, f"{sector}:{ticker}:{label}-assertions-list-missing"
        if any(not isinstance(item, Mapping) for item in raw_assertions):
            return FAIL, f"{sector}:{ticker}:{label}-assertion-row-not-object"
        assertions = [dict(item) for item in raw_assertions]
        required_assertion_fields = {
            "subject",
            "subject_detail",
            "assertion_mode",
            "blockable",
            "accession",
            "form_type",
            "filing_date",
            "section",
            "excerpt",
            "corroborating_distress",
            "issuer_cik",
            "source_url",
            "content_revision",
        }
        for assertion in assertions:
            if not required_assertion_fields.issubset(assertion):
                return FAIL, f"{sector}:{ticker}:{label}-assertion-schema-incomplete"
            assertion_subject = _upper(assertion.get("subject"))
            assertion_mode = _upper(assertion.get("assertion_mode"))
            if (
                assertion_subject not in GOING_CONCERN_ASSERTION_SUBJECTS
                or assertion_mode not in GOING_CONCERN_ASSERTION_MODES
                or not isinstance(assertion.get("blockable"), bool)
                or str(assertion.get("accession") or "") != expected_accession
                or _upper(assertion.get("form_type")) != expected_form
                or str(assertion.get("filing_date") or "") != expected_filing_date
                or _normalized_cik(assertion.get("issuer_cik")) != expected_issuer_cik
                or str(assertion.get("source_url") or "") != expected_source_url
                or str(assertion.get("content_revision") or "").strip().lower()
                != expected_filing_revision
                or not str(assertion.get("section") or "").strip()
                or not str(assertion.get("excerpt") or "").strip()
                or not isinstance(assertion.get("corroborating_distress"), list)
                or (
                    assertion.get("subject_detail") is not None
                    and not isinstance(assertion.get("subject_detail"), str)
                )
                or bool(assertion.get("blockable"))
                != (
                    assertion_mode == "AFFIRMATIVE_CURRENT"
                    and assertion_subject in GOING_CONCERN_BLOCKABLE_SUBJECTS
                )
            ):
                return FAIL, f"{sector}:{ticker}:{label}-assertion-attribution-invalid"
        observed_revision = str(observed.get("content_revision") or "").strip().lower()
        if not observed_revision and len(assertions) == 1:
            observed_revision = str(assertions[0].get("content_revision") or "").strip().lower()
        if (
            _upper(evaluation.get("status")) != "PASS"
            or str(evaluation.get("evidence_ref_id") or "") != expected_ref
            or str(evaluation.get("evidence_url") or "") != expected_source_url
            or _upper(observed.get("assertion"))
            != "NO_BLOCKABLE_ATTRIBUTED_ASSERTION"
            or any(bool(item.get("blockable")) for item in assertions)
            or observed_revision != expected_filing_revision
            or str(observed.get("accession") or "") != expected_accession
            or _upper(observed.get("form_type")) != expected_form
            or str(observed.get("filing_date") or "") != expected_filing_date
            or _normalized_cik(observed.get("issuer_cik")) != expected_issuer_cik
            or str(observed.get("source_url") or "") != expected_source_url
        ):
            return FAIL, f"{sector}:{ticker}:{label}-gate-proof-mismatch"
        if (
            _canonical_going_concern_gate_projection(evaluation)
            != replayed_projection
        ):
            return FAIL, f"{sector}:{ticker}:{label}-detector-replay-mismatch"
    return PASS, f"{sector}:{ticker}:exact-fixture-pass"


def _going_concern_true_proof(row: Mapping[str, Any]) -> tuple[str, str, str | None]:
    ticker = _upper(row.get("ticker"))
    expected = next(
        (item for item in _GOING_CONCERN_TRUE_CONTROLS if item["ticker"] == ticker),
        None,
    )
    if expected is None:
        return FAIL, f"{ticker or 'UNKNOWN'}:unknown-true-control", None
    required = {key: expected[key] for key in (
        "ticker",
        "subject",
        "content_revision",
        "accession",
        "form_type",
        "filing_date",
        "evidence_ref_id",
    )}
    required["fixture_kind"] = "true_positive"
    mismatches = [
        key
        for key, value in required.items()
        if str(row.get(key) or "").strip().upper() != str(value).strip().upper()
    ]
    if mismatches:
        return (
            FAIL,
            f"{ticker}:trusted-control-identity-mismatch:{','.join(mismatches)}",
            str(expected["subject"]),
        )
    executed = evaluate_going_concern_filing_text(
        ticker,
        str(expected["filing_text"]),
        accession=str(expected["accession"]),
        form_type=str(expected["form_type"]),
        filing_date=str(expected["filing_date"]),
    )
    observed = _mapping(executed.get("observed_value"))
    output_matches = bool(
        _upper(executed.get("status")) == "FAIL"
        and _upper(executed.get("reason_code"))
        == "QUARANTINE_STRUCTURAL:GOING_CONCERN"
        and str(executed.get("evidence_ref_id") or "") == expected["evidence_ref_id"]
        and _upper(observed.get("subject")) == expected["subject"]
        and _upper(observed.get("assertion_mode")) == "AFFIRMATIVE_CURRENT"
        and bool(observed.get("blockable"))
        and str(observed.get("accession") or "") == expected["accession"]
        and _upper(observed.get("form_type")) == expected["form_type"]
        and str(observed.get("filing_date") or "") == expected["filing_date"]
        and str(observed.get("content_revision") or "")
        == expected["content_revision"]
    )
    if not output_matches:
        return FAIL, f"{ticker}:current-detector-gate-did-not-block", str(expected["subject"])
    return PASS, f"{ticker}:current-detector-gate-blocked", str(expected["subject"])


def _check_going_concern(
    bundle: _Bundle, report: _Report, thresholds: AcceptanceThresholds
) -> None:
    value = _mapping(bundle.evidence.get("going_concern"))
    false_rows = _rows(value.get("false_positive_fixtures"))
    true_rows = _rows(value.get("true_positive_controls"))
    if false_rows is None or true_rows is None:
        report.unavailable(
            "going_concern_regressions",
            expected={
                "false_positives_not_blocked": list(thresholds.going_concern_false_positives),
                "true_positive_subjects_blocked": ["REGISTRANT", "CONSOLIDATED_SUBSIDIARY"],
            },
            missing_fields=[
                "acceptance_evidence.going_concern.false_positive_fixtures",
                "acceptance_evidence.going_concern.true_positive_controls",
            ],
        )
        return
    false_by_ticker = {_upper(row.get("ticker")): row for row in false_rows}
    required_false = {_upper(value) for value in thresholds.going_concern_false_positives}
    missing_false = sorted(required_false - set(false_by_ticker))
    frozen_by_ticker = {
        _upper(row.get("ticker")): row for row in _GOING_CONCERN_FALSE_FIXTURES
    }
    use_frozen_false = required_false == set(frozen_by_ticker)
    false_proofs = [
        _going_concern_false_proof(
            bundle,
            row,
            frozen_fixture=(
                frozen_by_ticker.get(_upper(row.get("ticker")))
                if use_frozen_false
                else None
            ),
        )
        for row in false_rows
    ]
    true_proofs = [_going_concern_true_proof(row) for row in true_rows]
    blocked_false = sorted(
        _upper(row.get("ticker"))
        for row, (status, _detail) in zip(false_rows, false_proofs, strict=True)
        if status == FAIL
    )
    blocked_subjects = {
        _upper(subject)
        for status, _detail, subject in true_proofs
        if status == PASS and subject
    }
    required_subjects = {"REGISTRANT", "CONSOLIDATED_SUBSIDIARY"}
    missing_subjects = sorted(required_subjects - blocked_subjects)
    proof_statuses = [status for status, _ in false_proofs] + [
        status for status, _detail, _subject in true_proofs
    ]
    proof_status = (
        FAIL
        if FAIL in proof_statuses
        else NOT_EVALUABLE
        if NOT_EVALUABLE in proof_statuses
        else PASS
    )
    passed = not missing_false and not blocked_false and not missing_subjects
    binding_rows = [
        {**row, "canonical_id": str(row.get("canonical_id") or "").strip()}
        for row in [*false_rows, *true_rows]
    ]
    binding_status, binding = _cohort_manifest_binding(
        bundle,
        thresholds=thresholds,
        family="going_concern",
        rows=binding_rows,
    )
    semantic_status, semantic_binding = _going_concern_semantic_provenance_binding(
        false_rows,
        thresholds=thresholds,
    )
    combined_binding_status = (
        FAIL
        if FAIL in {binding_status, semantic_status}
        else NOT_EVALUABLE
        if NOT_EVALUABLE in {binding_status, semantic_status}
        else PASS
    )
    report.add(
        "going_concern_regressions",
        _manifest_bound_status(
            content_passed=passed,
            binding_status=combined_binding_status,
            proof_status=proof_status,
        ),
        expected={
            "false_positives_not_blocked": sorted(required_false),
            "true_positive_subjects_blocked": sorted(required_subjects),
        },
        actual={
            "false_positive_fixtures": len(false_rows),
            "missing_false_positive_fixtures": missing_false,
            "incorrectly_blocked_false_positives": blocked_false,
            "blocked_true_positive_subjects": sorted(blocked_subjects),
            "missing_true_positive_subjects": missing_subjects,
            "current_artifact_gate_proof": {
                "status": (
                    FAIL
                    if any(status == FAIL for status, _detail in false_proofs)
                    else NOT_EVALUABLE
                    if any(status == NOT_EVALUABLE for status, _detail in false_proofs)
                    else PASS
                ),
                "examples": [detail for _, detail in false_proofs],
            },
            "frozen_fixture_execution": {
                "status": (
                    FAIL
                    if any(status == FAIL for status, _detail, _subject in true_proofs)
                    else PASS
                ),
                "examples": [detail for _, detail, _subject in true_proofs],
            },
            "cohort_manifest": binding,
            "semantic_provenance_manifest": semantic_binding,
        },
        evidence_source="acceptance_evidence.going_concern",
    )


def _check_one_disposition_per_issuer(
    slots: list[dict[str, Any]] | None,
    source: str | None,
    report: _Report,
) -> None:
    if slots is None:
        report.unavailable(
            "one_terminal_disposition_per_admitted_issuer",
            expected="exactly one terminal disposition per admitted issuer",
            missing_fields=["security_slots or candidate_dispositions"],
        )
        return
    admitted = [row for row in slots if _upper(row.get("scope_status")) == "IN_SCOPE"]
    by_issuer: dict[str, list[str]] = {}
    unresolved: list[str] = []
    invalid_terminal: list[str] = []
    for row in admitted:
        ticker = _upper(row.get("ticker")) or "UNKNOWN"
        issuer = _issuer_key(row)
        if issuer is None:
            unresolved.append(ticker)
            continue
        by_issuer.setdefault(issuer, []).append(ticker)
        if _upper(row.get("terminal_state")) not in VALID_TERMINAL_STATES - {"OUT_OF_SCOPE"}:
            invalid_terminal.append(ticker)
    duplicates = {issuer: tickers for issuer, tickers in by_issuer.items() if len(tickers) != 1}
    passed = not unresolved and not invalid_terminal and not duplicates
    report.add(
        "one_terminal_disposition_per_admitted_issuer",
        PASS if passed else FAIL,
        expected="exactly one terminal disposition per admitted issuer",
        actual={
            "admitted_securities": len(admitted),
            "admitted_issuers": len(by_issuer),
            "unresolved_identity": len(unresolved),
            "invalid_terminal_state": len(invalid_terminal),
            "duplicate_admitted_issuers": len(duplicates),
        },
        evidence_source=source,
        details={
            "unresolved_examples": unresolved[:30],
            "invalid_terminal_examples": invalid_terminal[:30],
            "duplicate_examples": dict(list(sorted(duplicates.items()))[:20]),
        },
    )


def _underwriting_proof_errors(
    artifact: AutonomousSectorFinancialRunArtifact,
    *,
    ticker: str,
) -> list[str]:
    selected = _upper(ticker)
    disposition = next(
        (item for item in artifact.candidate_dispositions if item.ticker == selected),
        None,
    )
    if disposition is None:
        return ["selected disposition missing"]
    underwriting = disposition.underwriting_result
    if underwriting is None:
        return ["underwritten disposition lacks structured underwriting_result"]
    errors: list[str] = []
    if underwriting.status != "COMPLETED" or underwriting.verdict not in {
        "ACTIONABLE",
        "WATCHLIST_ONLY",
        "AVOID",
    }:
        errors.append("underwriting_result is not a completed terminal verdict")
    matching_runs = [
        run
        for run in artifact.company_autonomy_runs
        if isinstance(run, Mapping)
        and _upper(run.get("ticker")) == selected
        and str(run.get("run_id") or "").strip() == underwriting.child_run_id
    ]
    if len(matching_runs) != 1:
        errors.append(
            "structured underwriting does not bind to exactly one selected-company child run"
        )
        return errors
    child = matching_runs[0]
    source_bindings = artifact.competitive_frontier.get("source_bindings")
    expected_binding = (
        source_bindings.get(selected) if isinstance(source_bindings, dict) else None
    )
    if not isinstance(expected_binding, dict) or child.get("source_binding") != expected_binding:
        errors.append(
            "selected-company child source_binding does not match the competitive frontier"
        )
    nested = _mapping(child.get("artifact"))
    if not nested:
        errors.append("selected-company child run lacks its raw artifact")
        return errors
    attempts = child.get("attempts")
    if (
        not isinstance(attempts, list)
        or not attempts
        or not isinstance(attempts[-1], Mapping)
        or dict(attempts[-1]) != nested
    ):
        errors.append("terminal company-child attempt does not match artifact ledger")
        return errors
    errors.extend(
        _provider_execution_proof_errors(
            nested.get("provider_usage"),
            scope="selected-company underwriting",
            expected_lane="company_underwriting",
        )
    )
    request = _mapping(nested.get("request"))
    candidate_scope = _mapping(request.get("candidate_scope"))
    if (
        str(request.get("run_id") or "").strip() != underwriting.child_run_id
        or str(request.get("as_of_date") or "") != artifact.as_of_date
        or str(candidate_scope.get("mode") or "") != "single_candidate"
        or [_upper(item) for item in candidate_scope.get("tickers") or []]
        != [selected]
    ):
        errors.append("selected-company nested request identity mismatch")
    if candidate_scope.get("source_binding") != expected_binding:
        errors.append(
            "selected-company nested child source_binding does not match the competitive frontier"
        )
    tool_rows = _rows(nested.get("tool_calls"))
    evidence_rows = _rows(nested.get("evidence"))
    decision_rows = _rows(nested.get("candidate_decisions"))
    if tool_rows is None or evidence_rows is None or decision_rows is None:
        errors.append("selected-company child artifact lacks raw tool/evidence/decision ledgers")
        return errors
    successful_tools = {
        str(row.get("call_id") or "").strip(): row
        for row in tool_rows
        if str(row.get("call_id") or "").strip() and _upper(row.get("status")) == "OK"
    }
    missing_tool_ids = sorted(set(underwriting.tool_call_ids) - set(successful_tools))
    if missing_tool_ids:
        errors.append("underwriting tool IDs are not successful raw calls: " + ",".join(missing_tool_ids))
    decision = next(
        (row for row in decision_rows if _upper(row.get("ticker")) == selected),
        None,
    )
    if decision is None:
        errors.append("selected-company child artifact lacks its candidate decision")
        return errors
    decision_refs = {
        str(value).strip() for value in decision.get("evidence_ref_ids") or [] if str(value).strip()
    }
    evidence_by_id = {
        str(row.get("evidence_id") or "").strip(): row
        for row in evidence_rows
        if str(row.get("evidence_id") or "").strip()
    }
    prefix = f"{underwriting.child_run_id}:"
    for full_ref in underwriting.evidence_ref_ids:
        local_ref = full_ref[len(prefix) :] if full_ref.startswith(prefix) else ""
        row = evidence_by_id.get(local_ref)
        if (
            not local_ref
            or local_ref not in decision_refs
            or row is None
            or _upper(row.get("confidence")) not in {"MODERATE", "HIGH"}
            or str(row.get("tool_call_id") or "").strip() not in successful_tools
        ):
            errors.append(f"underwriting evidence is not decision/tool linked: {full_ref}")
    return errors


def _provider_execution_proof_errors(
    raw_rows: Any,
    *,
    scope: str,
    expected_lane: str,
) -> list[str]:
    """Require physical, fully accounted provider calls for completed LLM work."""

    rows = _rows(raw_rows)
    if rows is None:
        return [f"{scope} provider_usage is not a list"]
    if not rows:
        return [f"{scope} has no provider execution records"]
    errors: list[str] = []
    seen_ids: set[str] = set()
    successful = 0
    for index, row in enumerate(rows):
        prefix = f"{scope} provider_usage[{index}]"
        call_id = str(row.get("provider_call_id") or "").strip()
        if not call_id:
            errors.append(f"{prefix} provider_call_id missing")
        elif call_id in seen_ids:
            errors.append(f"{prefix} provider_call_id duplicate")
        else:
            seen_ids.add(call_id)
        if not str(row.get("provider") or "").strip():
            errors.append(f"{prefix} provider missing")
        if not str(row.get("model") or "").strip():
            errors.append(f"{prefix} model missing")
        if _canonical_lane_name(str(row.get("lane") or "")) != expected_lane:
            errors.append(f"{prefix} lane mismatch")
        status = _upper(row.get("status"))
        if status not in {"OK", "ERROR", "INCOMPLETE"}:
            errors.append(f"{prefix} status invalid-or-missing")
        elif status == "OK":
            successful += 1
            provider = _lower(row.get("provider"))
            model = _lower(row.get("model"))
            if provider != "openai":
                errors.append(f"{prefix} successful provider must be openai")
            if model != "gpt-5.5" and not model.startswith("gpt-5.5-"):
                errors.append(f"{prefix} successful model must match pinned GPT-5.5")
            if not str(row.get("schema_name") or "").strip():
                errors.append(f"{prefix} schema_name missing")
            if not isinstance(row.get("estimated_tokens"), bool):
                errors.append(f"{prefix} estimated_tokens metadata missing")
            input_tokens = row.get("input_tokens")
            output_tokens = row.get("output_tokens")
            cached_tokens = row.get("cached_input_tokens")
            if (
                not isinstance(input_tokens, int)
                or isinstance(input_tokens, bool)
                or input_tokens <= 0
            ):
                errors.append(f"{prefix} successful input_tokens must be positive")
            if (
                not isinstance(output_tokens, int)
                or isinstance(output_tokens, bool)
                or output_tokens <= 0
            ):
                errors.append(f"{prefix} successful output_tokens must be positive")
            if (
                provider == "openai"
                and (model == "gpt-5.5" or model.startswith("gpt-5.5-"))
                and isinstance(input_tokens, int)
                and not isinstance(input_tokens, bool)
                and input_tokens > 0
                and isinstance(output_tokens, int)
                and not isinstance(output_tokens, bool)
                and output_tokens > 0
                and isinstance(cached_tokens, int)
                and not isinstance(cached_tokens, bool)
            ):
                expected_cost = _estimate_cost_usd(
                    model,
                    input_tokens,
                    output_tokens,
                    provider_name=provider,
                    cached_input_tokens=cached_tokens,
                )
                expected_microdollars = int(
                    (Decimal(str(expected_cost)) * Decimal(1_000_000)).quantize(
                        Decimal("1"), rounding=ROUND_HALF_UP
                    )
                )
                actual_microdollars = _record_cost_microdollars(row)
                if expected_microdollars <= 0 or actual_microdollars <= 0:
                    errors.append(f"{prefix} successful priced cost must be positive")
                elif actual_microdollars != expected_microdollars:
                    errors.append(
                        f"{prefix} GPT-5.5 cost mismatch:"
                        f"expected={expected_microdollars}:actual={actual_microdollars}"
                    )
        errors.extend(
            f"{prefix} {item}"
            for item in _raw_numeric_errors(row, include_tokens=True)
        )
    if successful == 0:
        errors.append(f"{scope} has no successful provider execution record")
    return errors


def _check_actionable_requirements(
    slots: list[dict[str, Any]] | None,
    source: str | None,
    bundle: _Bundle,
    report: _Report,
) -> None:
    if bundle.artifact_contract_errors:
        report.add(
            "actionable_underwriting_and_validation",
            FAIL,
            expected="100% of actionable selections UNDERWRITTEN and VALIDATED",
            actual={
                "completed_underwriting": 0,
                "completed_validation": 0,
                "completed_proofs": 0,
                "compliant": 0,
                "violations": len(bundle.artifact_contract_errors),
            },
            evidence_source=source,
            details={
                "violation_examples": list(bundle.artifact_contract_errors[:30]),
                "reason": "invalid sector artifacts cannot prove completed underwriting or validation",
            },
        )
        return
    if slots is None or not bundle.artifact_models:
        report.unavailable(
            "actionable_underwriting_and_validation",
            expected="100% of actionable selections UNDERWRITTEN and VALIDATED",
            missing_fields=["valid v2 sector artifacts"],
        )
        return
    completed_underwriting = [
        (artifact, disposition)
        for artifact in bundle.artifact_models
        for disposition in artifact.candidate_dispositions
        if disposition.terminal_state == "UNDERWRITTEN"
    ]
    completed_validation = [
        artifact
        for artifact in bundle.artifact_models
        if artifact.selection_validation is not None
        and artifact.selection_validation.status in {"VALIDATED", "CONTRADICTED"}
    ]
    violations: list[dict[str, Any]] = []
    for artifact, disposition in completed_underwriting:
        proof_errors = _underwriting_proof_errors(
            artifact,
            ticker=disposition.ticker,
        )
        if proof_errors:
            violations.append(
                {
                    "sector": _lower(artifact.sector),
                    "ticker": disposition.ticker,
                    "proof_type": "UNDERWRITING",
                    "underwriting_verdict": disposition.underwriting_verdict,
                    "underwriting_proof_errors": proof_errors,
                }
            )
    for artifact in completed_validation:
        validation = artifact.selection_validation
        assert validation is not None
        validation_provider_errors = _provider_execution_proof_errors(
            validation.provider_usage,
            scope="selected-company validation",
            expected_lane="selected_company_validation",
        )
        evidence_by_id = {item.evidence_id: item for item in validation.evidence}
        tool_by_id = {item.call_id: item for item in validation.tool_calls}
        validation_errors: list[str] = []
        if not validation.validator_run_id or not validation.validator_verdict:
            validation_errors.append("validation lacks identified validator/verdict")
        validator_run_id = str(validation.validator_run_id or "").strip()
        child_run_ids = {
            str(item.underwriting_result.child_run_id or "").strip()
            for item in artifact.candidate_dispositions
            if item.underwriting_result is not None
            and str(item.underwriting_result.child_run_id or "").strip()
        }
        child_run_ids.update(
            str(run.get("run_id") or "").strip()
            for run in artifact.company_autonomy_runs
            if isinstance(run, Mapping)
            and str(run.get("run_id") or "").strip()
        )
        if validator_run_id in child_run_ids:
            validation_errors.append("validator_run_id aliases an underwriting child_run_id")
        namespace = f"{validator_run_id}:"
        if any(not item.call_id.startswith(namespace) for item in validation.tool_calls):
            validation_errors.append("validation tool IDs are not validator namespaced")
        if any(
            not item.evidence_id.startswith(namespace)
            or (
                item.tool_call_id is not None
                and not item.tool_call_id.startswith(namespace)
            )
            for item in validation.evidence
        ):
            validation_errors.append("validation evidence IDs are not validator namespaced")
        if any(
            str(row.get("validator_run_id") or "") != validator_run_id
            or not str(row.get("provider_call_id") or "").startswith(namespace)
            for row in validation.provider_usage
        ):
            validation_errors.append("validation provider rows lack validator run identity")
        if (
            validation.terminal_ledger_fingerprint
            != selection_validation_terminal_ledger_fingerprint(validation.to_dict())
        ):
            validation_errors.append("validation terminal ledger fingerprint mismatch")
        if not validation.evidence_ref_ids:
            validation_errors.append("validation lacks challenge evidence references")
        for evidence_ref in validation.evidence_ref_ids:
            evidence = evidence_by_id.get(evidence_ref)
            tool = tool_by_id.get(str(evidence.tool_call_id or "")) if evidence else None
            if (
                evidence is None
                or tool is None
                or tool.status != "OK"
                or tool.lane != "selected_company_validation"
                or evidence_ref not in tool.evidence_ref_ids
                or _upper(evidence.ticker) != _upper(validation.selected_ticker)
                or _upper(evidence.confidence) not in {"MODERATE", "HIGH"}
            ):
                validation_errors.append(
                    f"validation evidence is not challenge/tool linked: {evidence_ref}"
                )
        if validation_errors or validation_provider_errors:
            violations.append(
                {
                    "sector": _lower(artifact.sector),
                    "ticker": _upper(validation.selected_ticker),
                    "proof_type": "SELECTION_VALIDATION",
                    "validation_proof_errors": validation_errors,
                    "validation_provider_proof_errors": validation_provider_errors,
                    "selection_validation_status": validation.status,
                    "validation_evidence_count": len(validation.evidence_ref_ids),
                    "validation_tool_call_count": len(validation.tool_calls),
                }
            )
    total_completed = len(completed_underwriting) + len(completed_validation)
    report.add(
        "actionable_underwriting_and_validation",
        PASS if not violations else FAIL,
        expected=(
            "100% of completed underwriting and VALIDATED/CONTRADICTED challenges "
            "have source-bound physical provider proof"
        ),
        actual={
            "completed_underwriting": len(completed_underwriting),
            "completed_validation": len(completed_validation),
            "completed_proofs": total_completed,
            "compliant": total_completed - len(violations),
            "violations": len(violations),
        },
        evidence_source=source,
        details={"violation_examples": violations[:30]},
    )


def _artifact_failure_evidence(artifact: Mapping[str, Any]) -> bool:
    execution = _upper(artifact.get("execution_status"))
    decision = _upper(artifact.get("decision_status"))
    benchmark = _upper(artifact.get("benchmark_execution_status"))
    degraded = " ".join(_upper(value) for value in artifact.get("degraded_states") or [])
    return (
        execution not in {"", "COMPLETED"}
        or decision == "INCOMPLETE"
        or benchmark in {"FAILED", "SKIPPED_PROVIDER", "SKIPPED_BUDGET"}
        or any(
            token in degraded for token in ("PROVIDER", "BUDGET", "RUNTIME", "EXCEPTION", "FAILED")
        )
    )


def _check_false_no_selection(bundle: _Bundle, report: _Report) -> None:
    raw_violations = [
        {
            "sector": _lower(artifact.get("sector")),
            "execution_status": _upper(artifact.get("execution_status")),
            "decision_status": _upper(artifact.get("decision_status")),
            "final_verdict": _upper(artifact.get("final_verdict")),
        }
        for artifact in bundle.artifacts
        if _upper(artifact.get("final_verdict")) == "NO_SELECTION"
        and _artifact_failure_evidence(artifact)
    ]
    if raw_violations:
        report.add(
            "no_false_no_selection_on_execution_failure",
            FAIL,
            expected=0,
            actual=len(raw_violations),
            evidence_source="raw sector_artifacts",
            details={"violation_examples": raw_violations[:30]},
        )
        return
    artifacts: list[Mapping[str, Any]] = [
        artifact.to_dict() for artifact in bundle.artifact_models
    ]
    if not artifacts:
        report.unavailable(
            "no_false_no_selection_on_execution_failure",
            expected=0,
            missing_fields=["valid v2 sector artifacts"],
        )
        return
    violations = [
        {
            "sector": _lower(artifact.get("sector")),
            "execution_status": _upper(artifact.get("execution_status")),
            "decision_status": _upper(artifact.get("decision_status")),
            "final_verdict": _upper(artifact.get("final_verdict")),
        }
        for artifact in artifacts
        if _upper(artifact.get("final_verdict")) == "NO_SELECTION"
        and _artifact_failure_evidence(artifact)
    ]
    report.add(
        "no_false_no_selection_on_execution_failure",
        PASS if not violations else FAIL,
        expected=0,
        actual=len(violations),
        evidence_source="validated sector_artifacts",
        details={"violation_examples": violations[:30]},
    )


def _accounting_payload(payload: Mapping[str, Any]) -> tuple[dict[str, Any], str | None]:
    rollups = _mapping(payload.get("rollups"))
    if isinstance(rollups.get("lane_usage_totals"), Mapping):
        return dict(rollups["lane_usage_totals"]), "rollups.lane_usage_totals"
    for key in ("lane_usage_totals", "lane_usage"):
        if isinstance(payload.get(key), Mapping):
            return dict(payload[key]), key
    return {}, None


def _canonical_lane_name(value: str) -> str:
    return "selected_company_validation" if value == "selected_validation" else value


def _empty_lane_totals() -> dict[str, int]:
    return {field_name: 0 for field_name in LANE_INTEGER_FIELDS}


def _record_cost_microdollars(record: Mapping[str, Any]) -> int:
    declared = record.get("cost_microdollars")
    if isinstance(declared, int) and not isinstance(declared, bool) and declared >= 0:
        return declared
    raw = record.get("cost_estimate_usd", record.get("cost_usd", 0))
    try:
        amount = Decimal(str(raw or 0))
    except (InvalidOperation, TypeError, ValueError):
        return 0
    if not amount.is_finite() or amount < 0:
        return 0
    return int(
        (amount * Decimal(1_000_000)).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    )


def _record_non_negative_int(record: Mapping[str, Any], field_name: str) -> int:
    value = record.get(field_name)
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


def _raw_numeric_errors(
    record: Mapping[str, Any],
    *,
    include_tokens: bool,
) -> list[str]:
    errors: list[str] = []
    if include_tokens:
        for field_name in (
            "input_tokens",
            "cached_input_tokens",
            "output_tokens",
            "reserved_output_tokens",
        ):
            if field_name not in record:
                errors.append(f"{field_name}:missing")
                continue
            value = record.get(field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                errors.append(f"{field_name}:must-be-non-negative-integer")
        input_tokens = record.get("input_tokens", 0)
        cached_tokens = record.get("cached_input_tokens", 0)
        if (
            isinstance(input_tokens, int)
            and not isinstance(input_tokens, bool)
            and isinstance(cached_tokens, int)
            and not isinstance(cached_tokens, bool)
            and cached_tokens > input_tokens
        ):
            errors.append("cached_input_tokens:exceeds-input_tokens")

    cost_fields = {
        field_name
        for field_name in ("cost_microdollars", "cost_estimate_usd", "cost_usd")
        if field_name in record
    }
    if include_tokens and not cost_fields:
        errors.append("cost:missing")
    declared_micro = record.get("cost_microdollars")
    if "cost_microdollars" in record and (
        not isinstance(declared_micro, int)
        or isinstance(declared_micro, bool)
        or declared_micro < 0
    ):
        errors.append("cost_microdollars:must-be-non-negative-integer")
    usd_micros: list[tuple[str, int]] = []
    for field_name in ("cost_estimate_usd", "cost_usd"):
        if field_name not in record:
            continue
        raw = record.get(field_name)
        if isinstance(raw, bool):
            errors.append(f"{field_name}:must-be-finite-non-negative")
            continue
        try:
            amount = Decimal(str(raw))
        except (InvalidOperation, TypeError, ValueError):
            errors.append(f"{field_name}:must-be-finite-non-negative")
            continue
        if not amount.is_finite() or amount < 0:
            errors.append(f"{field_name}:must-be-finite-non-negative")
            continue
        usd_micros.append(
            (
                field_name,
                int(
                    (amount * Decimal(1_000_000)).quantize(
                        Decimal("1"), rounding=ROUND_HALF_UP
                    )
                ),
            )
        )
    if (
        isinstance(declared_micro, int)
        and not isinstance(declared_micro, bool)
        and declared_micro >= 0
    ):
        for field_name, converted in usd_micros:
            if converted != declared_micro:
                errors.append(f"{field_name}:disagrees-with-cost_microdollars")
    if len({value for _, value in usd_micros}) > 1:
        errors.append("cost_usd_fields:disagree")
    return errors


def _raw_accounting_from_artifacts(
    bundle: _Bundle,
) -> tuple[dict[str, Any], list[str], int, int]:
    lanes = {lane: _empty_lane_totals() for lane in CANONICAL_LANES}
    errors: list[str] = []
    company_run_count = 0
    company_tool_attempts = 0
    seen_provider_ids: set[tuple[str, str]] = set()

    def add_tool(record: Mapping[str, Any], *, default_lane: str) -> None:
        nonlocal company_tool_attempts
        lane = _canonical_lane_name(str(record.get("lane") or default_lane))
        if lane not in lanes:
            lane = default_lane
        errors.extend(
            f"{lane}:{item}"
            for item in _raw_numeric_errors(record, include_tokens=False)
        )
        status = _upper(record.get("status"))
        if status not in {"OK", "ERROR"}:
            return
        values = lanes[lane]
        values["tool_call_attempts"] += 1
        values["tool_calls_ok" if status == "OK" else "tool_calls_failed"] += 1
        values["cost_microdollars"] += _record_cost_microdollars(record)
        if lane == "company_underwriting":
            company_tool_attempts += 1

    def add_provider(
        record: Mapping[str, Any], *, default_lane: str, scope: str
    ) -> None:
        lane = _canonical_lane_name(str(record.get("lane") or default_lane))
        if lane not in lanes:
            lane = default_lane
        call_id = str(record.get("provider_call_id") or "").strip()
        provider = str(record.get("provider") or "").strip()
        model = str(record.get("model") or "").strip()
        status = _upper(record.get("status"))
        if not call_id:
            errors.append(f"{lane}:provider_call_id:missing")
        elif (scope, call_id) in seen_provider_ids:
            errors.append(f"{lane}:provider_call_id:duplicate:{scope}:{call_id}")
        else:
            seen_provider_ids.add((scope, call_id))
        if not provider:
            errors.append(f"{lane}:provider:missing")
        if not model:
            errors.append(f"{lane}:model:missing")
        if status not in {"OK", "ERROR", "INCOMPLETE"}:
            errors.append(f"{lane}:status:invalid-or-missing")
        errors.extend(
            f"{lane}:{item}"
            for item in _raw_numeric_errors(record, include_tokens=True)
        )
        values = lanes[lane]
        values["provider_call_attempts"] += 1
        values["provider_calls_ok" if status == "OK" else "provider_calls_failed"] += 1
        input_tokens = _record_non_negative_int(record, "input_tokens")
        cached_input_tokens = min(
            _record_non_negative_int(record, "cached_input_tokens"),
            input_tokens,
        )
        values["input_tokens"] += input_tokens
        values["cached_input_tokens"] += cached_input_tokens
        for field_name in ("output_tokens", "reserved_output_tokens"):
            values[field_name] += _record_non_negative_int(record, field_name)
        values["cost_microdollars"] += _record_cost_microdollars(record)

    result_rows = {
        _lower(row.get("sector")): row
        for row in bundle.payload.get("sector_results") or []
        if isinstance(row, Mapping)
    }
    for artifact in bundle.artifacts:
        sector = _lower(artifact.get("sector")) or "unknown"
        result_row = result_rows.get(sector, {})
        if _upper(result_row.get("benchmark_execution_status")) == "REUSED_FROM_RESUME":
            continue
        parent_calls = artifact.get("tool_calls")
        if not isinstance(parent_calls, list):
            errors.append(f"{sector}:tool_calls:not-a-list")
            parent_calls = []
        for call in parent_calls:
            if not isinstance(call, Mapping):
                errors.append(f"{sector}:tool_calls:non-object")
                continue
            question_id = _upper(call.get("question_id"))
            add_tool(
                call,
                default_lane=(
                    "repair_fallback"
                    if question_id.startswith(("AGR", "WR"))
                    else "parent_research"
                ),
            )
        provider_rows = artifact.get("provider_usage")
        if not isinstance(provider_rows, list):
            errors.append(f"{sector}:provider_usage:not-a-list")
            provider_rows = []
        for row in provider_rows:
            if isinstance(row, Mapping):
                add_provider(
                    row,
                    default_lane="parent_research",
                    scope=f"{sector}:parent",
                )
            else:
                errors.append(f"{sector}:provider_usage:non-object")

        company_runs = artifact.get("company_autonomy_runs")
        if not isinstance(company_runs, list):
            errors.append(f"{sector}:company_autonomy_runs:not-a-list")
            company_runs = []
        for run_index, run in enumerate(company_runs):
            if not isinstance(run, Mapping):
                errors.append(f"{sector}:company_run[{run_index}]:non-object")
                continue
            company_run_count += 1
            attempts_raw = run.get("attempts")
            if isinstance(attempts_raw, list) and attempts_raw:
                attempts = attempts_raw
            elif isinstance(run.get("artifact"), Mapping):
                attempts = [run["artifact"]]
            else:
                errors.append(f"{sector}:company_run[{run_index}]:missing-raw-artifact")
                attempts = []
            for attempt_index, attempt in enumerate(attempts):
                if not isinstance(attempt, Mapping):
                    errors.append(
                        f"{sector}:company_run[{run_index}].attempt[{attempt_index}]:non-object"
                    )
                    continue
                calls = attempt.get("tool_calls")
                if not isinstance(calls, list):
                    errors.append(
                        f"{sector}:company_run[{run_index}].attempt[{attempt_index}].tool_calls"
                    )
                    calls = []
                for call in calls:
                    if isinstance(call, Mapping):
                        add_tool(call, default_lane="company_underwriting")
                    else:
                        errors.append(f"{sector}:company_run[{run_index}]:non-object-tool-call")
                attempt_usage = attempt.get("provider_usage")
                if not isinstance(attempt_usage, list):
                    errors.append(
                        f"{sector}:company_run[{run_index}].attempt[{attempt_index}].provider_usage"
                    )
                    attempt_usage = []
                for row in attempt_usage:
                    if isinstance(row, Mapping):
                        add_provider(
                            row,
                            default_lane="company_underwriting",
                            scope=(
                                f"{sector}:company:{run_index}:{attempt_index}"
                            ),
                        )
                    else:
                        errors.append(f"{sector}:company_run[{run_index}]:non-object-provider")

        validation = _mapping(artifact.get("selection_validation"))
        validation_calls = validation.get("tool_calls", [])
        if not isinstance(validation_calls, list):
            errors.append(f"{sector}:selection_validation.tool_calls:not-a-list")
            validation_calls = []
        for call in validation_calls:
            if isinstance(call, Mapping):
                add_tool(call, default_lane="selected_company_validation")
            else:
                errors.append(f"{sector}:selection_validation.tool_calls:non-object")
        validation_usage = validation.get("provider_usage", [])
        if not isinstance(validation_usage, list):
            errors.append(f"{sector}:selection_validation.provider_usage:not-a-list")
            validation_usage = []
        for row in validation_usage:
            if isinstance(row, Mapping):
                add_provider(
                    row,
                    default_lane="selected_company_validation",
                    scope=f"{sector}:validation",
                )
            else:
                errors.append(f"{sector}:selection_validation.provider_usage:non-object")

        selection = _mapping(artifact.get("candidate_selection"))
        repair = _mapping(selection.get("data_gap_repair"))
        terminal_records = repair.get("terminal_cap_search_usage_records", [])
        if not isinstance(terminal_records, list):
            errors.append(f"{sector}:terminal_cap_search_usage_records:not-a-list")
            terminal_records = []
        reserve_costs: dict[tuple[str, int, str], int] = {}
        reserve_call_ids: set[tuple[tuple[str, int, str], str]] = set()

        def terminal_group_key(
            row: Mapping[str, Any], *, row_index: int, group_sector: str = sector
        ) -> tuple[str, int, str] | None:
            authorization = str(row.get("authorization_run_id") or "").strip()
            attempt = row.get("attempt_number")
            ticker = _upper(row.get("ticker"))
            if (
                not authorization
                or not isinstance(attempt, int)
                or isinstance(attempt, bool)
                or attempt <= 0
                or not ticker
            ):
                errors.append(
                    f"{group_sector}:terminal_cap_search[{row_index}]:incomplete-group-key"
                )
                return None
            return authorization, attempt, ticker

        for index, row in enumerate(terminal_records):
            if not isinstance(row, Mapping) or str(row.get("call_type") or "") != (
                "web_search_call_reserve"
            ):
                continue
            key = terminal_group_key(row, row_index=index)
            errors.extend(
                f"terminal_cap_search:{item}"
                for item in _raw_numeric_errors(row, include_tokens=False)
            )
            reserve_cost = _record_cost_microdollars(row)
            call_id = str(row.get("call_id") or "").strip()
            if reserve_cost <= 0:
                errors.append(
                    f"{sector}:terminal_cap_search[{index}]:reserve-cost-not-positive"
                )
            if not call_id:
                errors.append(
                    f"{sector}:terminal_cap_search[{index}]:reserve-call-id-missing"
                )
            if key is None:
                continue
            reserve_identity = (key, call_id)
            if call_id and reserve_identity in reserve_call_ids:
                errors.append(
                    f"{sector}:terminal_cap_search[{index}]:duplicate-reserve-call"
                )
            reserve_call_ids.add(reserve_identity)
            reserve_costs[key] = reserve_costs.get(key, 0) + reserve_cost
        consumed_reserve_groups: set[tuple[str, int, str]] = set()
        for index, row in enumerate(terminal_records):
            if not isinstance(row, Mapping):
                errors.append(f"{sector}:terminal_cap_search[{index}]:non-object")
                continue
            call_type = str(row.get("call_type") or "")
            failed = (
                _upper(row.get("billing_status")) == "WORST_CASE_RESERVED"
                or _upper(row.get("attempt_status")) in {"FAILED", "PENDING"}
            )
            normalized = {**row, "status": "ERROR" if failed else "OK"}
            if call_type == "web_search_call":
                add_tool(normalized, default_lane="terminal_cap_search")
            elif call_type == "responses_model":
                key = terminal_group_key(row, row_index=index)
                reserve_cost = 0
                if key is not None:
                    if key in consumed_reserve_groups:
                        errors.append(
                            f"{sector}:terminal_cap_search[{index}]:duplicate-response-group"
                        )
                    elif key not in reserve_costs and _upper(
                        row.get("billing_status")
                    ) != "AUTHORITATIVE_USAGE":
                        errors.append(
                            f"{sector}:terminal_cap_search[{index}]:response-without-reserve"
                        )
                    elif key in reserve_costs:
                        reserve_cost = reserve_costs[key]
                        consumed_reserve_groups.add(key)
                normalized.pop("cost_estimate_usd", None)
                normalized.pop("cost_usd", None)
                normalized["cost_microdollars"] = (
                    _record_cost_microdollars(row) + reserve_cost
                )
                add_provider(
                    normalized,
                    default_lane="terminal_cap_search",
                    scope=f"{sector}:terminal",
                )
        for orphan in sorted(set(reserve_costs) - consumed_reserve_groups):
            errors.append(
                f"{sector}:terminal_cap_search:orphan-reserve-group:"
                f"{orphan[0]}:{orphan[1]}:{orphan[2]}"
            )

    preflight = _mapping(bundle.payload.get("provider_preflight"))
    preflight_calls = preflight.get("provider_calls", [])
    if not isinstance(preflight_calls, list):
        errors.append("benchmark:provider_preflight.provider_calls:not-a-list")
        preflight_calls = []
    for row in preflight_calls:
        if isinstance(row, Mapping):
            add_provider(
                row,
                default_lane="provider_preflight",
                scope="benchmark:provider_preflight",
            )
        else:
            errors.append("benchmark:provider_preflight.provider_calls:non-object")

    aggregate = _empty_lane_totals()
    for values in lanes.values():
        for field_name in LANE_INTEGER_FIELDS:
            aggregate[field_name] += values[field_name]
    return {"lanes": lanes, "aggregate": aggregate}, errors, company_run_count, company_tool_attempts


def _declared_accounting_mismatches(
    accounting: Mapping[str, Any],
    recomputed: Mapping[str, Any],
) -> tuple[list[str], dict[str, dict[str, int]]]:
    raw_lanes = _mapping(accounting.get("lanes") or accounting.get("lane_totals"))
    declared_lanes = {
        _canonical_lane_name(str(key)): _mapping(value) for key, value in raw_lanes.items()
    }
    missing_cells: list[str] = []
    mismatches: dict[str, dict[str, int]] = {}
    for lane in CANONICAL_LANES:
        declared = declared_lanes.get(lane)
        if declared is None:
            missing_cells.append(lane)
            continue
        expected = recomputed["lanes"][lane]
        for field_name in LANE_INTEGER_FIELDS:
            value = declared.get(field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                missing_cells.append(f"{lane}.{field_name}")
            elif value != expected[field_name]:
                mismatches[f"{lane}.{field_name}"] = {
                    "recomputed": expected[field_name],
                    "declared": value,
                }
    aggregate = _mapping(accounting.get("aggregate"))
    for field_name in LANE_INTEGER_FIELDS:
        value = aggregate.get(field_name)
        expected = recomputed["aggregate"][field_name]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            missing_cells.append(f"aggregate.{field_name}")
        elif value != expected:
            mismatches[f"aggregate.{field_name}"] = {
                "recomputed": expected,
                "declared": value,
            }
    return missing_cells, mismatches


def _check_lane_cost_reconciliation(bundle: _Bundle, report: _Report) -> None:
    accounting, source = _accounting_payload(bundle.payload)
    if not accounting:
        report.unavailable(
            "lane_and_cost_reconciliation",
            expected="aggregate exactly equals raw provider/tool records",
            missing_fields=["rollups.lane_usage_totals"],
        )
        return
    if not bundle.artifact_models or bundle.artifact_contract_errors:
        report.unavailable(
            "lane_and_cost_reconciliation",
            expected="aggregate exactly equals raw provider/tool records",
            missing_fields=["valid v2 sector artifacts"],
        )
        return
    recomputed, raw_errors, _, _ = _raw_accounting_from_artifacts(bundle)
    missing_cells, mismatches = _declared_accounting_mismatches(accounting, recomputed)
    producer_flag = accounting.get("aggregate_reconciles")
    passed = (
        not raw_errors
        and not missing_cells
        and not mismatches
        and producer_flag is True
    )
    report.add(
        "lane_and_cost_reconciliation",
        PASS if passed else FAIL,
        expected=(
            "declared per-lane and aggregate calls, tokens, and costs exactly equal "
            "raw parent, repair, child, validation, terminal, and preflight records"
        ),
        actual={
            "raw_record_errors": len(raw_errors),
            "missing_metric_cells": missing_cells,
            "mismatches": mismatches,
            "producer_aggregate_reconciles": producer_flag,
            "recomputed_aggregate": recomputed["aggregate"],
        },
        evidence_source=f"raw sector/provider/tool ledgers + {source}",
        details={"raw_record_error_examples": raw_errors[:30]},
    )


def _check_company_child_call_reconciliation(bundle: _Bundle, report: _Report) -> None:
    accounting, source = _accounting_payload(bundle.payload)
    raw_lanes = _mapping(accounting.get("lanes") or accounting.get("lane_totals"))
    company_lane = _mapping(raw_lanes.get("company_underwriting"))
    declared = company_lane.get("tool_call_attempts")
    if not isinstance(declared, int) or isinstance(declared, bool):
        report.unavailable(
            "company_child_call_reconciliation",
            expected="every physical company-child tool call included in benchmark totals",
            missing_fields=[
                "rollups.lane_usage_totals.lanes.company_underwriting.tool_call_attempts"
            ],
        )
        return
    if not bundle.artifact_models or bundle.artifact_contract_errors:
        report.unavailable(
            "company_child_call_reconciliation",
            expected="every physical company-child tool call included in benchmark totals",
            missing_fields=["valid v2 sector artifacts"],
        )
        return
    _, raw_errors, run_count, raw_total = _raw_accounting_from_artifacts(bundle)
    child_errors = [value for value in raw_errors if ":company_run[" in value]
    passed = not child_errors and raw_total == declared
    report.add(
        "company_child_call_reconciliation",
        PASS if passed else FAIL,
        expected="every physical company-child tool call included in benchmark totals",
        actual={
            "company_child_runs": run_count,
            "raw_tool_call_attempts": raw_total,
            "benchmark_company_underwriting_tool_call_attempts": declared,
            "unresolved_child_runs": len(child_errors),
        },
        evidence_source=f"raw sector_artifacts[].company_autonomy_runs + {source}",
        details={"unresolved_examples": child_errors[:30]},
    )


def _evaluate_all_sector_v2_acceptance(
    payload: Mapping[str, Any],
    *,
    evidence: Mapping[str, Any] | None = None,
    artifacts: Sequence[Mapping[str, Any]] = (),
    data_plane_report: Mapping[str, Any] | None = None,
    thresholds: AcceptanceThresholds | None = None,
    input_errors: Sequence[str] = (),
    source_path: Path | None = None,
    artifacts_verified: bool = False,
) -> dict[str, Any]:
    """Evaluate persisted evidence and return a JSON-safe acceptance report."""

    thresholds = thresholds or AcceptanceThresholds()
    integrity_errors = [str(value) for value in input_errors]
    inline_explicit = payload.get("sector_artifacts")
    if isinstance(inline_explicit, list):
        integrity_errors.extend(
            f"benchmark:sector_artifacts[{index}]:INLINE_ARTIFACT_FORBIDDEN"
            for index, item in enumerate(inline_explicit)
            if isinstance(item, Mapping)
        )
    for index, row in enumerate(payload.get("sector_results") or []):
        if isinstance(row, Mapping) and isinstance(row.get("artifact"), Mapping):
            integrity_errors.append(
                f"benchmark:sector_results[{index}].artifact:INLINE_ARTIFACT_FORBIDDEN"
            )
    if artifacts and not artifacts_verified:
        integrity_errors.append("artifacts:DIRECT_UNBOUND_ARTIFACTS_FORBIDDEN")
    bundle = _Bundle(
        payload=dict(payload),
        evidence=dict(evidence or _mapping(payload.get("acceptance_evidence"))),
        data_plane_report=dict(data_plane_report or _mapping(payload.get("data_plane_report"))),
        artifacts=tuple(dict(item) for item in artifacts),
        artifact_models=(),
        artifact_contract_errors=(),
        source_path=source_path,
        input_errors=tuple(integrity_errors),
    )

    artifact_models, artifact_contract_errors = _deserialize_artifacts(bundle.artifacts)
    bundle = _Bundle(
        payload=bundle.payload,
        evidence=bundle.evidence,
        data_plane_report=bundle.data_plane_report,
        artifacts=bundle.artifacts,
        artifact_models=artifact_models,
        artifact_contract_errors=artifact_contract_errors,
        source_path=bundle.source_path,
        input_errors=bundle.input_errors,
    )

    report = _Report()
    _check_input_integrity(bundle, report)
    _check_artifact_contract_and_binding(bundle, report)
    _check_sector_coverage(bundle, report, thresholds)
    slots, slot_source = _security_slots(bundle)
    _check_security_reconciliation(slots, slot_source, report, thresholds)
    _check_security_issuer_counts(bundle, slots, report, thresholds)
    _check_sparse_history(bundle, report, thresholds)
    _check_cap_price_losses(bundle, report, thresholds)
    _check_cached_filings(bundle, report, thresholds)
    _check_extraction_gaps(bundle, report, thresholds)
    _check_visibility(bundle, report, thresholds)
    _check_top_ten_context(bundle, report, thresholds)
    _check_going_concern(bundle, report, thresholds)
    _check_one_disposition_per_issuer(slots, slot_source, report)
    _check_actionable_requirements(slots, slot_source, bundle, report)
    _check_false_no_selection(bundle, report)
    _check_lane_cost_reconciliation(bundle, report)
    _check_company_child_call_reconciliation(bundle, report)

    status_counts = {
        status: sum(1 for row in report.checks if row["status"] == status)
        for status in (PASS, FAIL, NOT_EVALUABLE)
    }
    return {
        "artifact_type": "all_sector_v2_release_acceptance_report",
        "status": report.status,
        "pipeline_version": "v2",
        "network_calls": 0,
        "llm_calls": 0,
        "source_path": str(source_path) if source_path is not None else None,
        "thresholds": {
            "sector_count": thresholds.sector_count,
            "expected_sector_labels": (
                list(thresholds.expected_sector_labels)
                if thresholds.expected_sector_labels is not None
                else None
            ),
            "security_slots": thresholds.security_slots,
            "sparse_history": thresholds.sparse_history,
            "cap_price_losses": thresholds.cap_price_losses,
            "cached_annual_filings": thresholds.cached_annual_filings,
            "extraction_gaps": thresholds.extraction_gaps,
            "previously_unreviewed": thresholds.previously_unreviewed,
            "omitted_top_ten": thresholds.omitted_top_ten,
            "going_concern_false_positives": list(thresholds.going_concern_false_positives),
            "trusted_evidence_manifest_sha256": dict(
                thresholds.trusted_evidence_manifest_sha256
            ),
            "trusted_semantic_manifest_sha256": dict(
                thresholds.trusted_semantic_manifest_sha256
            ),
        },
        "status_counts": status_counts,
        "checks": report.checks,
    }


def evaluate_all_sector_v2_acceptance(
    payload: Mapping[str, Any],
    *,
    evidence: Mapping[str, Any] | None = None,
    artifacts: Sequence[Mapping[str, Any]] = (),
    data_plane_report: Mapping[str, Any] | None = None,
    thresholds: AcceptanceThresholds | None = None,
    input_errors: Sequence[str] = (),
    source_path: Path | None = None,
) -> dict[str, Any]:
    """Evaluate caller data without granting direct artifacts verified status."""

    return _evaluate_all_sector_v2_acceptance(
        payload,
        evidence=evidence,
        artifacts=artifacts,
        data_plane_report=data_plane_report,
        thresholds=thresholds,
        input_errors=input_errors,
        source_path=source_path,
        artifacts_verified=False,
    )


def run_all_sector_v2_acceptance(
    input_path: str | Path,
    *,
    evidence_path: str | Path | None = None,
    thresholds: AcceptanceThresholds | None = None,
) -> dict[str, Any]:
    bundle = load_acceptance_bundle(input_path, evidence_path=evidence_path)
    return _evaluate_all_sector_v2_acceptance(
        bundle.payload,
        evidence=bundle.evidence,
        artifacts=bundle.artifacts,
        data_plane_report=bundle.data_plane_report,
        thresholds=thresholds,
        input_errors=bundle.input_errors,
        source_path=bundle.source_path,
        artifacts_verified=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Benchmark/replay JSON, or a directory containing benchmark_summary.json.",
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        help="Optional independent acceptance-evidence JSON ledger.",
    )
    parser.add_argument(
        "--trusted-manifest",
        type=Path,
        help=(
            "Independent trusted-manifest JSON containing caller-pinned cohort "
            "SHA-256 values; benchmark-declared hashes are not trust anchors."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional report path. Without it, JSON is written only to stdout.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    thresholds = None
    if args.trusted_manifest is not None:
        thresholds = replace(
            AcceptanceThresholds(),
            trusted_evidence_manifest_sha256=load_trusted_manifest_hashes(
                args.trusted_manifest
            ),
            trusted_semantic_manifest_sha256=load_trusted_semantic_manifest_hashes(
                args.trusted_manifest
            ),
        )
    payload = run_all_sector_v2_acceptance(
        args.input,
        evidence_path=args.evidence,
        thresholds=thresholds,
    )
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)
    if payload["status"] == PASS:
        return 0
    if payload["status"] == FAIL:
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
