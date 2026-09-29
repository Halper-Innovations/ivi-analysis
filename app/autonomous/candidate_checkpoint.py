"""Crash-safe, exact-key checkpoints for autonomous candidate memos.

Checkpoint files are provisional execution state, not publishable decision
artifacts.  A caller must recompute and pass the current P0 financial-integrity
result before hydration; this module then verifies the complete immutable key,
the stored envelope checksum, substantive memo content, and physical provider
usage before returning a reusable result.
"""

from __future__ import annotations

import fcntl
import json
import math
import re
from contextlib import contextmanager
from functools import wraps
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

from app.autonomous.candidate_review import candidate_memo_is_substantive
from app.util.json_io import JsonCorruptError, atomic_write_json, read_json_safe


CANDIDATE_MEMO_CHECKPOINT_CONTRACT_VERSION = "autonomous_sector_candidate_memo_checkpoint_v1"
CANDIDATE_MEMO_CONTEXT_CHECKPOINT_CONTRACT_VERSION = (
    "autonomous_sector_candidate_memo_context_checkpoint_v1"
)
CANDIDATE_MEMO_CONTEXT_PROMPT_VERSION = "autonomous_sector_candidate_memo_context_v1"
CANDIDATE_MEMO_PROMPT_VERSION = "autonomous_sector_candidate_memo_prompt_v1"
CANDIDATE_MEMO_FINANCIAL_INTEGRITY_CONTRACT_VERSION = "financial_integrity_scope_v1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_KEY_FIELDS = frozenset(
    {
        "campaign_id",
        "run_identity",
        "sector",
        "market_cap_focus",
        "pipeline_version",
        "effective_as_of_date",
        "ticker",
        "issuer_identity",
        "issuer_identity_sha256",
        "packet_sha256",
        "scenarios_sha256",
        "all_scenarios_sha256",
        "evidence_sha256",
        "memo_context",
        "memo_context_sha256",
        "source_binding",
        "source_binding_sha256",
        "shared_context_checkpoint_key_sha256",
        "shared_context_source_binding_sha256",
        "financial_integrity_contract_version",
        "financial_integrity_scope_fingerprint",
        "quote_snapshot_ids",
        "price_share_basis",
        "price_share_basis_sha256",
        "prompt_version",
        "prompt_sha256",
        "schema_name",
        "schema_sha256",
        "provider",
        "model",
        "reasoning_level",
        "response_config",
        "response_config_sha256",
    }
)
_KEY_HASH_FIELDS = frozenset(
    {
        "issuer_identity_sha256",
        "packet_sha256",
        "scenarios_sha256",
        "all_scenarios_sha256",
        "evidence_sha256",
        "memo_context_sha256",
        "source_binding_sha256",
        "shared_context_checkpoint_key_sha256",
        "shared_context_source_binding_sha256",
        "financial_integrity_scope_fingerprint",
        "price_share_basis_sha256",
        "prompt_sha256",
        "schema_sha256",
        "response_config_sha256",
    }
)
_REQUIRED_CONTEXT_KEY_FIELDS = frozenset(
    {
        "campaign_id",
        "run_identity",
        "sector",
        "market_cap_focus",
        "pipeline_version",
        "effective_as_of_date",
        "source_binding",
        "source_binding_sha256",
        "financial_integrity_contract_version",
        "financial_integrity_scope_fingerprints",
        "financial_integrity_scopes_sha256",
        "quote_snapshot_ids_by_scope",
        "prompt_version",
        "cohort_prompt_sha256",
        "cohort_schema_sha256",
        "triage_schema_sha256",
        "provider",
        "model",
        "reasoning_level",
        "cohort_response_config",
        "cohort_response_config_sha256",
        "triage_response_config",
        "triage_response_config_sha256",
        "schema_names",
    }
)
_CONTEXT_KEY_HASH_FIELDS = frozenset(
    {
        "source_binding_sha256",
        "financial_integrity_scopes_sha256",
        "cohort_prompt_sha256",
        "cohort_schema_sha256",
        "triage_schema_sha256",
        "cohort_response_config_sha256",
        "triage_response_config_sha256",
    }
)


class CandidateMemoCheckpointError(RuntimeError):
    """Base fail-closed checkpoint error."""


class CandidateMemoCheckpointCorruptError(CandidateMemoCheckpointError):
    """An exact-key checkpoint exists but cannot be trusted."""


def canonical_json_sha256(value: Any) -> str:
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


def candidate_checkpoint_key_sha256(key: Mapping[str, Any]) -> str:
    _validate_checkpoint_key(key)
    return canonical_json_sha256(dict(key))


def candidate_context_checkpoint_key_sha256(key: Mapping[str, Any]) -> str:
    _validate_context_checkpoint_key(key)
    return canonical_json_sha256(dict(key))


def candidate_checkpoint_path(
    root: str | Path,
    *,
    namespace: Mapping[str, Any],
    key: Mapping[str, Any],
) -> Path:
    """Return the deterministic path for one exact candidate key."""

    key_digest = candidate_checkpoint_key_sha256(key)
    namespace_digest = canonical_json_sha256(dict(namespace))
    ticker = str(key.get("ticker") or "").strip().upper()
    ticker_token = "".join(char if char.isalnum() else "_" for char in ticker)
    if not ticker_token:
        raise ValueError("candidate checkpoint ticker is empty")
    return Path(root) / namespace_digest / ticker_token / f"{key_digest}.json"


def candidate_context_checkpoint_path(
    root: str | Path,
    *,
    namespace: Mapping[str, Any],
    key: Mapping[str, Any],
) -> Path:
    """Return the deterministic path for the shared cohort/triage context."""

    key_digest = candidate_context_checkpoint_key_sha256(key)
    namespace_digest = canonical_json_sha256(dict(namespace))
    return Path(root) / namespace_digest / "_memo_context" / f"{key_digest}.json"


def _validate_checkpoint_key(key: Mapping[str, Any]) -> None:
    if not isinstance(key, Mapping):
        raise ValueError("candidate checkpoint key must be a mapping")
    missing = sorted(_REQUIRED_KEY_FIELDS - set(key))
    if missing:
        raise ValueError(f"candidate checkpoint key missing fields: {missing}")
    ticker = str(key.get("ticker") or "").strip()
    if not ticker or ticker != ticker.upper():
        raise ValueError("candidate checkpoint ticker must be canonical uppercase")
    for field in _KEY_HASH_FIELDS:
        value = str(key.get(field) or "").strip().lower()
        if not _SHA256_RE.fullmatch(value):
            raise ValueError(f"candidate checkpoint {field} must be SHA-256")
    quote_snapshot_ids = key.get("quote_snapshot_ids")
    if not isinstance(quote_snapshot_ids, Mapping) or not quote_snapshot_ids:
        raise ValueError("candidate checkpoint requires quote snapshot identities")
    for snapshot_id in quote_snapshot_ids.values():
        if not _SHA256_RE.fullmatch(str(snapshot_id or "").strip().lower()):
            raise ValueError("candidate checkpoint quote snapshot identity is invalid")
    if (
        key.get("financial_integrity_contract_version")
        != CANDIDATE_MEMO_FINANCIAL_INTEGRITY_CONTRACT_VERSION
    ):
        raise ValueError("candidate checkpoint financial-integrity version is invalid")
    if key.get("prompt_version") != CANDIDATE_MEMO_PROMPT_VERSION:
        raise ValueError("candidate checkpoint prompt version is invalid")
    if not str(key.get("run_identity") or "").strip():
        raise ValueError("candidate checkpoint run identity is empty")
    if not str(key.get("provider") or "").strip():
        raise ValueError("candidate checkpoint provider is empty")
    if not str(key.get("model") or "").strip():
        raise ValueError("candidate checkpoint model is empty")
    if not isinstance(key.get("issuer_identity"), Mapping):
        raise ValueError("candidate checkpoint issuer identity is invalid")
    if not isinstance(key.get("memo_context"), Mapping):
        raise ValueError("candidate checkpoint memo context is invalid")
    if not isinstance(key.get("response_config"), Mapping):
        raise ValueError("candidate checkpoint response config is invalid")
    bound_hashes = {
        "issuer_identity_sha256": key["issuer_identity"],
        "memo_context_sha256": key["memo_context"],
        "response_config_sha256": key["response_config"],
    }
    if isinstance(key.get("source_binding"), Mapping):
        bound_hashes["source_binding_sha256"] = key["source_binding"]
    if isinstance(key.get("price_share_basis"), Mapping):
        bound_hashes["price_share_basis_sha256"] = key["price_share_basis"]
    for hash_field, bound_value in bound_hashes.items():
        if canonical_json_sha256(bound_value) != key.get(hash_field):
            raise ValueError(f"candidate checkpoint {hash_field} does not match bound content")
    source_binding = key["source_binding"]
    expected_source_hashes = {
        "candidate_packet_sha256": key["packet_sha256"],
        "candidate_scenarios_sha256": key["scenarios_sha256"],
        "all_scenarios_sha256": key["all_scenarios_sha256"],
        "evidence_sha256": key["evidence_sha256"],
        "shared_context_checkpoint_key_sha256": key["shared_context_checkpoint_key_sha256"],
        "shared_context_source_binding_sha256": key["shared_context_source_binding_sha256"],
    }
    if any(
        source_binding.get(field) != expected for field, expected in expected_source_hashes.items()
    ):
        raise ValueError("candidate checkpoint source binding does not match key hashes")


def _validate_context_checkpoint_key(key: Mapping[str, Any]) -> None:
    if not isinstance(key, Mapping):
        raise ValueError("candidate context checkpoint key must be a mapping")
    missing = sorted(_REQUIRED_CONTEXT_KEY_FIELDS - set(key))
    if missing:
        raise ValueError(f"candidate context checkpoint key missing fields: {missing}")
    for field in _CONTEXT_KEY_HASH_FIELDS:
        value = str(key.get(field) or "").strip().lower()
        if not _SHA256_RE.fullmatch(value):
            raise ValueError(f"candidate context checkpoint {field} must be SHA-256")
    scope_fingerprints = key.get("financial_integrity_scope_fingerprints")
    quote_snapshot_ids_by_scope = key.get("quote_snapshot_ids_by_scope")
    if (
        not isinstance(scope_fingerprints, Mapping)
        or set(scope_fingerprints) != {"cohort", "triage"}
        or any(
            not _SHA256_RE.fullmatch(str(value or "").strip().lower())
            for value in scope_fingerprints.values()
        )
    ):
        raise ValueError("candidate context financial-integrity fingerprints are invalid")
    if not isinstance(quote_snapshot_ids_by_scope, Mapping) or set(quote_snapshot_ids_by_scope) != {
        "cohort",
        "triage",
    }:
        raise ValueError("candidate context checkpoint requires quote identities by scope")
    for quote_snapshot_ids in quote_snapshot_ids_by_scope.values():
        if (
            not isinstance(quote_snapshot_ids, Mapping)
            or not quote_snapshot_ids
            or any(
                not _SHA256_RE.fullmatch(str(value or "").strip().lower())
                for value in quote_snapshot_ids.values()
            )
        ):
            raise ValueError("candidate context checkpoint quote snapshot identity is invalid")
    if (
        key.get("financial_integrity_contract_version")
        != CANDIDATE_MEMO_FINANCIAL_INTEGRITY_CONTRACT_VERSION
    ):
        raise ValueError("candidate context financial-integrity version is invalid")
    if key.get("prompt_version") != CANDIDATE_MEMO_CONTEXT_PROMPT_VERSION:
        raise ValueError("candidate context prompt version is invalid")
    if not str(key.get("run_identity") or "").strip():
        raise ValueError("candidate context run identity is empty")
    if not str(key.get("provider") or "").strip():
        raise ValueError("candidate context provider is empty")
    if not str(key.get("model") or "").strip():
        raise ValueError("candidate context model is empty")
    schema_names = key.get("schema_names")
    if not isinstance(schema_names, list) or sorted(str(item) for item in schema_names) != [
        "autonomous_sector_cohort_comparison",
        "autonomous_sector_triage_surprises",
    ]:
        raise ValueError("candidate context schema binding is invalid")
    bound_hashes = {
        "source_binding_sha256": key.get("source_binding"),
        "cohort_response_config_sha256": key.get("cohort_response_config"),
        "triage_response_config_sha256": key.get("triage_response_config"),
    }
    for hash_field, bound_value in bound_hashes.items():
        if not isinstance(bound_value, Mapping):
            raise ValueError(f"candidate context {hash_field} bound content is invalid")
        if canonical_json_sha256(bound_value) != key.get(hash_field):
            raise ValueError(f"candidate context {hash_field} does not match bound content")


def _integrity_payload(value: Any) -> dict[str, Any]:
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
    if not isinstance(value, Mapping):
        raise CandidateMemoCheckpointCorruptError(
            "candidate checkpoint requires a concrete financial-integrity result"
        )
    payload = json.loads(json.dumps(dict(value), sort_keys=True, default=str))
    if (
        payload.get("status") != "PASS"
        or payload.get("passed") is not True
        or payload.get("violations")
    ):
        raise CandidateMemoCheckpointCorruptError(
            "candidate checkpoint financial-integrity result is not PASS"
        )
    if not _SHA256_RE.fullmatch(str(payload.get("scope_fingerprint") or "").strip().lower()):
        raise CandidateMemoCheckpointCorruptError(
            "candidate checkpoint integrity fingerprint is invalid"
        )
    return payload


def _integrity_scope_payloads(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != {"cohort", "triage"}:
        raise CandidateMemoCheckpointCorruptError(
            "candidate context checkpoint requires exact cohort and triage "
            "financial-integrity results"
        )
    return {
        scope_name: _integrity_payload(value[scope_name]) for scope_name in ("cohort", "triage")
    }


def _partial_checkpoint_paths(path: Path) -> list[Path]:
    """Return legacy and unique temporary siblings left by interrupted writes."""

    candidates = [path.with_suffix(path.suffix + ".tmp")]
    if path.parent.exists():
        candidates.extend(sorted(path.parent.glob(f".{path.name}.*.tmp")))
    return [candidate for candidate in candidates if candidate.exists()]


def _require_no_partial_checkpoint(path: Path, *, label: str) -> None:
    partials = _partial_checkpoint_paths(path)
    if partials:
        rendered = ", ".join(str(item) for item in partials)
        raise CandidateMemoCheckpointCorruptError(
            f"ambiguous partial {label} checkpoint exists: {rendered}"
        )


@contextmanager
def _checkpoint_write_lock(path: Path) -> Iterator[None]:
    """Serialize same-key immutable comparison and publication across processes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _serialize_checkpoint_write(
    function: Callable[..., dict[str, Any]],
) -> Callable[..., dict[str, Any]]:
    @wraps(function)
    def wrapped(path: str | Path, *args: Any, **kwargs: Any) -> dict[str, Any]:
        resolved_path = Path(path)
        with _checkpoint_write_lock(resolved_path):
            return function(resolved_path, *args, **kwargs)

    return wrapped


def _finite_nonnegative(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value)) and float(value) >= 0


def _validate_provider_usage(
    value: Any,
    *,
    key: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise CandidateMemoCheckpointCorruptError(
            "candidate checkpoint has no physical provider usage"
        )
    records: list[dict[str, Any]] = []
    schema_names = key.get("schema_names")
    allowed_schema_names = (
        {str(item) for item in schema_names}
        if isinstance(schema_names, list)
        else {str(key.get("schema_name") or "")}
    )
    successful_schema_names: set[str] = set()
    rows_by_schema: dict[str, list[dict[str, Any]]] = {
        schema_name: [] for schema_name in allowed_schema_names
    }
    for raw in value:
        if not isinstance(raw, Mapping):
            raise CandidateMemoCheckpointCorruptError(
                "candidate checkpoint provider usage row is invalid"
            )
        row = json.loads(json.dumps(dict(raw), sort_keys=True, default=str))
        status = str(row.get("status") or "").strip().upper()
        schema_name = str(row.get("schema_name") or "")
        if status == "OK":
            successful_schema_names.add(schema_name)
        if status not in {"OK", "ERROR", "INCOMPLETE"}:
            raise CandidateMemoCheckpointCorruptError(
                "candidate checkpoint provider status is invalid"
            )
        if str(row.get("provider") or "").strip().lower() != str(key["provider"]).lower():
            raise CandidateMemoCheckpointCorruptError(
                "candidate checkpoint provider binding drifted"
            )
        if str(row.get("model") or "").strip() != str(key["model"]):
            raise CandidateMemoCheckpointCorruptError("candidate checkpoint model binding drifted")
        if schema_name not in allowed_schema_names:
            raise CandidateMemoCheckpointCorruptError("candidate checkpoint schema binding drifted")
        response_config = (
            key.get("cohort_response_config")
            if schema_name == "autonomous_sector_cohort_comparison"
            else key.get("triage_response_config")
            if schema_name == "autonomous_sector_triage_surprises"
            else key.get("response_config")
        )
        if not isinstance(response_config, Mapping):
            raise CandidateMemoCheckpointCorruptError(
                "candidate checkpoint response configuration is invalid"
            )
        if row.get("max_output_tokens_applied") is not True or row.get(
            "requested_max_output_tokens"
        ) != response_config.get("max_output_tokens"):
            raise CandidateMemoCheckpointCorruptError(
                "candidate checkpoint effective output configuration drifted"
            )
        for field in (
            "input_tokens",
            "cached_input_tokens",
            "output_tokens",
            "reserved_output_tokens",
            "cost_estimate_usd",
        ):
            if not _finite_nonnegative(row.get(field, 0)):
                raise CandidateMemoCheckpointCorruptError(
                    f"candidate checkpoint provider usage {field} is invalid"
                )
        records.append(row)
        rows_by_schema[schema_name].append(row)
    if successful_schema_names != allowed_schema_names:
        raise CandidateMemoCheckpointCorruptError(
            "candidate checkpoint does not have one successful response per schema"
        )
    strict_paid_execution = any(
        bool(
            (
                key.get("cohort_response_config")
                if schema_name == "autonomous_sector_cohort_comparison"
                else key.get("triage_response_config")
                if schema_name == "autonomous_sector_triage_surprises"
                else key.get("response_config")
            ).get("strict_paid_execution")
        )
        for schema_name in allowed_schema_names
    )
    if strict_paid_execution and any(
        len(rows_by_schema[schema_name]) != 1
        or rows_by_schema[schema_name][0].get("status") != "OK"
        for schema_name in allowed_schema_names
    ):
        raise CandidateMemoCheckpointCorruptError(
            "strict candidate checkpoint must contain exactly one successful "
            "physical response per schema"
        )
    return records


def _validate_candidate_payload(value: Any, *, ticker: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CandidateMemoCheckpointCorruptError("candidate checkpoint memo payload is invalid")
    payload = json.loads(json.dumps(dict(value), sort_keys=True, default=str))
    if (
        str(payload.get("ticker") or "").strip().upper() != ticker
        or str(payload.get("source") or "").strip().lower() != "llm"
        or str(payload.get("status") or "").strip().upper() != "OK"
        or not candidate_memo_is_substantive(payload)
    ):
        raise CandidateMemoCheckpointCorruptError(
            "candidate checkpoint memo is not a substantive ticker-bound LLM result"
        )
    return payload


def _validate_envelope(
    envelope: Any,
    *,
    expected_key: Mapping[str, Any],
    current_financial_integrity: Any,
) -> dict[str, Any]:
    if not isinstance(envelope, Mapping):
        raise CandidateMemoCheckpointCorruptError("candidate checkpoint root must be an object")
    payload = json.loads(json.dumps(dict(envelope), sort_keys=True, default=str))
    checksum = str(payload.pop("envelope_sha256", "") or "").strip().lower()
    if not _SHA256_RE.fullmatch(checksum) or canonical_json_sha256(payload) != checksum:
        raise CandidateMemoCheckpointCorruptError("candidate checkpoint envelope checksum mismatch")
    if payload.get("checkpoint_contract_version") != CANDIDATE_MEMO_CHECKPOINT_CONTRACT_VERSION:
        raise CandidateMemoCheckpointCorruptError("candidate checkpoint contract version mismatch")
    try:
        expected_digest = candidate_checkpoint_key_sha256(expected_key)
    except ValueError as exc:
        raise CandidateMemoCheckpointCorruptError(str(exc)) from exc
    if payload.get("checkpoint_key") != dict(expected_key):
        raise CandidateMemoCheckpointCorruptError("candidate checkpoint immutable key mismatch")
    if payload.get("checkpoint_key_sha256") != expected_digest:
        raise CandidateMemoCheckpointCorruptError("candidate checkpoint key checksum mismatch")
    current_integrity = _integrity_payload(current_financial_integrity)
    stored_integrity = _integrity_payload(payload.get("financial_integrity"))
    if stored_integrity != current_integrity:
        raise CandidateMemoCheckpointCorruptError(
            "candidate checkpoint financial-integrity scope drifted"
        )
    if stored_integrity.get("scope_fingerprint") != expected_key.get(
        "financial_integrity_scope_fingerprint"
    ) or stored_integrity.get("ticker_snapshot_ids") != expected_key.get("quote_snapshot_ids"):
        raise CandidateMemoCheckpointCorruptError(
            "candidate checkpoint quote or integrity fingerprint mismatch"
        )
    ticker = str(expected_key["ticker"])
    payload["candidate"] = _validate_candidate_payload(
        payload.get("candidate"),
        ticker=ticker,
    )
    usage = payload.get("usage")
    if not isinstance(usage, Mapping):
        raise CandidateMemoCheckpointCorruptError("candidate checkpoint logical usage is invalid")
    payload["usage"] = json.loads(json.dumps(dict(usage), sort_keys=True, default=str))
    payload["provider_usage"] = _validate_provider_usage(
        payload.get("provider_usage"),
        key=expected_key,
    )
    payload["envelope_sha256"] = checksum
    return payload


def _validate_context_section(value: Any, *, section: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CandidateMemoCheckpointCorruptError(f"candidate context {section} payload is invalid")
    payload = json.loads(json.dumps(dict(value), sort_keys=True, default=str))
    content_field = "paragraphs" if section == "cohort" else "items"
    content = payload.get(content_field)
    if (
        payload.get("source") != "llm"
        or payload.get("status") != "OK"
        or not isinstance(content, list)
        or not any(str(item).strip() for item in content)
        or not isinstance(payload.get("usage"), Mapping)
    ):
        raise CandidateMemoCheckpointCorruptError(
            f"candidate context {section} is not a complete LLM result"
        )
    return payload


def _validate_context_envelope(
    envelope: Any,
    *,
    expected_key: Mapping[str, Any],
    current_financial_integrity_scopes: Mapping[str, Any],
    triage_prompt_builder: Callable[[dict[str, Any]], str] | None = None,
    expected_triage_prompt_sha256: str | None = None,
) -> dict[str, Any]:
    if not isinstance(envelope, Mapping):
        raise CandidateMemoCheckpointCorruptError(
            "candidate context checkpoint root must be an object"
        )
    payload = json.loads(json.dumps(dict(envelope), sort_keys=True, default=str))
    checksum = str(payload.pop("envelope_sha256", "") or "").strip().lower()
    if not _SHA256_RE.fullmatch(checksum) or canonical_json_sha256(payload) != checksum:
        raise CandidateMemoCheckpointCorruptError(
            "candidate context checkpoint envelope checksum mismatch"
        )
    if (
        payload.get("checkpoint_contract_version")
        != CANDIDATE_MEMO_CONTEXT_CHECKPOINT_CONTRACT_VERSION
    ):
        raise CandidateMemoCheckpointCorruptError(
            "candidate context checkpoint contract version mismatch"
        )
    try:
        expected_digest = candidate_context_checkpoint_key_sha256(expected_key)
    except ValueError as exc:
        raise CandidateMemoCheckpointCorruptError(str(exc)) from exc
    if payload.get("checkpoint_key") != dict(expected_key):
        raise CandidateMemoCheckpointCorruptError(
            "candidate context checkpoint immutable key mismatch"
        )
    if payload.get("checkpoint_key_sha256") != expected_digest:
        raise CandidateMemoCheckpointCorruptError(
            "candidate context checkpoint key checksum mismatch"
        )
    current_integrity_scopes = _integrity_scope_payloads(current_financial_integrity_scopes)
    stored_integrity_scopes = _integrity_scope_payloads(payload.get("financial_integrity_scopes"))
    if stored_integrity_scopes != current_integrity_scopes:
        raise CandidateMemoCheckpointCorruptError(
            "candidate context financial-integrity scopes drifted"
        )
    scope_fingerprints = {
        scope_name: scope_payload["scope_fingerprint"]
        for scope_name, scope_payload in current_integrity_scopes.items()
    }
    quote_snapshot_ids_by_scope = {
        scope_name: scope_payload.get("ticker_snapshot_ids")
        for scope_name, scope_payload in current_integrity_scopes.items()
    }
    if (
        scope_fingerprints != expected_key.get("financial_integrity_scope_fingerprints")
        or quote_snapshot_ids_by_scope != expected_key.get("quote_snapshot_ids_by_scope")
        or canonical_json_sha256(current_integrity_scopes)
        != expected_key.get("financial_integrity_scopes_sha256")
    ):
        raise CandidateMemoCheckpointCorruptError(
            "candidate context quote or integrity fingerprint mismatch"
        )
    cohort = _validate_context_section(payload.get("cohort"), section="cohort")
    triage = _validate_context_section(payload.get("triage"), section="triage")
    triage_prompt_sha256 = str(expected_triage_prompt_sha256 or "")
    if triage_prompt_builder is not None:
        try:
            triage_prompt_sha256 = canonical_json_sha256(triage_prompt_builder(dict(cohort)))
        except Exception as exc:
            raise CandidateMemoCheckpointCorruptError(
                "candidate context triage prompt could not be reconstructed"
            ) from exc
    if not _SHA256_RE.fullmatch(triage_prompt_sha256):
        raise CandidateMemoCheckpointCorruptError(
            "candidate context expected triage prompt hash is invalid"
        )
    if payload.get("triage_prompt_sha256") != triage_prompt_sha256:
        raise CandidateMemoCheckpointCorruptError("candidate context triage prompt binding drifted")
    usage_calls = payload.get("usage_calls")
    if not isinstance(usage_calls, list) or [
        str(item.get("section") or "") for item in usage_calls if isinstance(item, Mapping)
    ] != ["cohort_comparison", "triage_surprises"]:
        raise CandidateMemoCheckpointCorruptError("candidate context logical usage is invalid")
    payload["cohort"] = cohort
    payload["triage"] = triage
    payload["usage_calls"] = json.loads(json.dumps(usage_calls, sort_keys=True, default=str))
    payload["provider_usage"] = _validate_provider_usage(
        payload.get("provider_usage"),
        key=expected_key,
    )
    payload["envelope_sha256"] = checksum
    return payload


def bind_provider_usage_to_checkpoint(
    rows: list[dict[str, Any]],
    *,
    checkpoint_key_sha256: str,
    reused: bool,
) -> list[dict[str, Any]]:
    bound_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        physical_identity = canonical_json_sha256(
            {
                "checkpoint_key_sha256": checkpoint_key_sha256,
                "physical_sequence": index,
                "provider_usage": {
                    key: value
                    for key, value in row.items()
                    if key
                    not in {
                        "provider_call_id",
                        "reused_from_checkpoint",
                        "checkpoint_key_sha256",
                        "checkpoint_physical_id",
                    }
                },
            }
        )
        bound_rows.append(
            {
                **row,
                "reused_from_checkpoint": bool(reused),
                "checkpoint_key_sha256": checkpoint_key_sha256,
                "checkpoint_physical_id": physical_identity,
            }
        )
    return bound_rows


@_serialize_checkpoint_write
def persist_candidate_memo_context_checkpoint(
    path: str | Path,
    *,
    checkpoint_key: Mapping[str, Any],
    cohort: Mapping[str, Any],
    triage: Mapping[str, Any],
    triage_prompt_sha256: str,
    usage_calls: list[dict[str, Any]],
    provider_usage: list[dict[str, Any]],
    financial_integrity_scopes: Mapping[str, Any],
    producer_run_id: str,
    created_at: str,
) -> dict[str, Any]:
    """Persist exact shared prompt context before candidate work begins."""

    path = Path(path)
    _require_no_partial_checkpoint(path, label="candidate context")
    integrity_scopes = _integrity_scope_payloads(financial_integrity_scopes)
    key = json.loads(json.dumps(dict(checkpoint_key), sort_keys=True, default=str))
    key_digest = candidate_context_checkpoint_key_sha256(key)
    validated_cohort = _validate_context_section(cohort, section="cohort")
    validated_triage = _validate_context_section(triage, section="triage")
    validated_provider_usage = _validate_provider_usage(provider_usage, key=key)
    if not _SHA256_RE.fullmatch(str(triage_prompt_sha256 or "").strip().lower()):
        raise CandidateMemoCheckpointCorruptError("candidate context triage prompt hash is invalid")
    body = {
        "checkpoint_contract_version": (CANDIDATE_MEMO_CONTEXT_CHECKPOINT_CONTRACT_VERSION),
        "checkpoint_key": key,
        "checkpoint_key_sha256": key_digest,
        "cohort": validated_cohort,
        "triage": validated_triage,
        "triage_prompt_sha256": str(triage_prompt_sha256),
        "usage_calls": json.loads(json.dumps(usage_calls, sort_keys=True, default=str)),
        "provider_usage": validated_provider_usage,
        "financial_integrity_scopes": integrity_scopes,
        "producer_run_id": str(producer_run_id),
        "created_at": str(created_at),
    }
    envelope = {**body, "envelope_sha256": canonical_json_sha256(body)}
    envelope = _validate_context_envelope(
        envelope,
        expected_key=key,
        current_financial_integrity_scopes=integrity_scopes,
        expected_triage_prompt_sha256=str(triage_prompt_sha256),
    )
    if path.exists():
        try:
            existing = read_json_safe(path)
        except (JsonCorruptError, OSError) as exc:
            raise CandidateMemoCheckpointCorruptError(
                f"candidate context checkpoint is unreadable: {path}: {exc}"
            ) from exc
        validated = _validate_context_envelope(
            existing,
            expected_key=key,
            current_financial_integrity_scopes=integrity_scopes,
            expected_triage_prompt_sha256=str(triage_prompt_sha256),
        )
        if validated.get("triage_prompt_sha256") != str(triage_prompt_sha256):
            raise CandidateMemoCheckpointCorruptError(
                "immutable candidate context has a different triage prompt"
            )
        for field in (
            "cohort",
            "triage",
            "usage_calls",
            "provider_usage",
            "financial_integrity_scopes",
        ):
            if validated.get(field) != envelope.get(field):
                raise CandidateMemoCheckpointCorruptError(
                    "immutable candidate context checkpoint has different content"
                )
        return validated
    atomic_write_json(path, envelope)
    return envelope


def load_candidate_memo_context_checkpoint(
    path: str | Path,
    *,
    expected_key: Mapping[str, Any],
    current_financial_integrity_scopes: Mapping[str, Any],
    triage_prompt_builder: Callable[[dict[str, Any]], str],
) -> dict[str, Any] | None:
    """Hydrate the exact shared context before any shared provider call."""

    path = Path(path)
    _require_no_partial_checkpoint(path, label="candidate context")
    if not path.exists():
        return None
    try:
        raw = read_json_safe(path)
    except (JsonCorruptError, OSError) as exc:
        raise CandidateMemoCheckpointCorruptError(
            f"candidate context checkpoint is unreadable: {path}: {exc}"
        ) from exc
    validated = _validate_context_envelope(
        raw,
        expected_key=expected_key,
        current_financial_integrity_scopes=current_financial_integrity_scopes,
        triage_prompt_builder=triage_prompt_builder,
    )
    key_digest = str(validated["checkpoint_key_sha256"])
    return {
        "cohort": dict(validated["cohort"]),
        "triage": dict(validated["triage"]),
        "usage_calls": [
            {
                **dict(item),
                "reused_from_checkpoint": True,
                "checkpoint_key_sha256": key_digest,
            }
            for item in validated["usage_calls"]
        ],
        "provider_usage": bind_provider_usage_to_checkpoint(
            validated["provider_usage"],
            checkpoint_key_sha256=key_digest,
            reused=True,
        ),
        "checkpoint_key_sha256": key_digest,
        "producer_run_id": validated.get("producer_run_id"),
        "created_at": validated.get("created_at"),
    }


@_serialize_checkpoint_write
def persist_candidate_memo_checkpoint(
    path: str | Path,
    *,
    checkpoint_key: Mapping[str, Any],
    candidate: Mapping[str, Any],
    usage: Mapping[str, Any],
    provider_usage: list[dict[str, Any]],
    financial_integrity: Any,
    producer_run_id: str,
    created_at: str,
) -> dict[str, Any]:
    """Persist one immutable successful candidate result atomically."""

    path = Path(path)
    _require_no_partial_checkpoint(path, label="candidate")
    integrity = _integrity_payload(financial_integrity)
    key = json.loads(json.dumps(dict(checkpoint_key), sort_keys=True, default=str))
    key_digest = candidate_checkpoint_key_sha256(key)
    validated_candidate = _validate_candidate_payload(
        candidate,
        ticker=str(key["ticker"]),
    )
    validated_usage = json.loads(json.dumps(dict(usage), sort_keys=True, default=str))
    validated_provider_usage = _validate_provider_usage(
        provider_usage,
        key=key,
    )
    body = {
        "checkpoint_contract_version": CANDIDATE_MEMO_CHECKPOINT_CONTRACT_VERSION,
        "checkpoint_key": key,
        "checkpoint_key_sha256": key_digest,
        "candidate": validated_candidate,
        "usage": validated_usage,
        "provider_usage": validated_provider_usage,
        "financial_integrity": integrity,
        "producer_run_id": str(producer_run_id),
        "created_at": str(created_at),
    }
    envelope = {
        **body,
        "envelope_sha256": canonical_json_sha256(body),
    }
    if path.exists():
        try:
            existing = read_json_safe(path)
        except (JsonCorruptError, OSError) as exc:
            raise CandidateMemoCheckpointCorruptError(
                f"candidate checkpoint is unreadable: {path}: {exc}"
            ) from exc
        validated = _validate_envelope(
            existing,
            expected_key=key,
            current_financial_integrity=integrity,
        )
        comparable_fields = (
            "candidate",
            "usage",
            "provider_usage",
            "financial_integrity",
        )
        if any(validated.get(field) != envelope.get(field) for field in comparable_fields):
            raise CandidateMemoCheckpointCorruptError(
                "immutable candidate checkpoint already exists with different content"
            )
        return validated
    atomic_write_json(path, envelope)
    return envelope


def load_candidate_memo_checkpoint(
    path: str | Path,
    *,
    expected_key: Mapping[str, Any],
    current_financial_integrity: Any,
) -> dict[str, Any] | None:
    """Hydrate one exact checkpoint, or return ``None`` when it is absent."""

    path = Path(path)
    _require_no_partial_checkpoint(path, label="candidate")
    if not path.exists():
        return None
    try:
        raw = read_json_safe(path)
    except (JsonCorruptError, OSError) as exc:
        raise CandidateMemoCheckpointCorruptError(
            f"candidate checkpoint is unreadable: {path}: {exc}"
        ) from exc
    validated = _validate_envelope(
        raw,
        expected_key=expected_key,
        current_financial_integrity=current_financial_integrity,
    )
    key_digest = str(validated["checkpoint_key_sha256"])
    return {
        "candidate": dict(validated["candidate"]),
        "usage": {
            **dict(validated["usage"]),
            "reused_from_checkpoint": True,
            "checkpoint_key_sha256": key_digest,
        },
        "provider_usage": bind_provider_usage_to_checkpoint(
            validated["provider_usage"],
            checkpoint_key_sha256=key_digest,
            reused=True,
        ),
        "checkpoint_key_sha256": key_digest,
        "producer_run_id": validated.get("producer_run_id"),
        "created_at": validated.get("created_at"),
    }


__all__ = [
    "CANDIDATE_MEMO_CHECKPOINT_CONTRACT_VERSION",
    "CANDIDATE_MEMO_CONTEXT_CHECKPOINT_CONTRACT_VERSION",
    "CANDIDATE_MEMO_CONTEXT_PROMPT_VERSION",
    "CANDIDATE_MEMO_FINANCIAL_INTEGRITY_CONTRACT_VERSION",
    "CANDIDATE_MEMO_PROMPT_VERSION",
    "CandidateMemoCheckpointCorruptError",
    "CandidateMemoCheckpointError",
    "candidate_checkpoint_key_sha256",
    "candidate_checkpoint_path",
    "candidate_context_checkpoint_key_sha256",
    "candidate_context_checkpoint_path",
    "canonical_json_sha256",
    "bind_provider_usage_to_checkpoint",
    "load_candidate_memo_checkpoint",
    "load_candidate_memo_context_checkpoint",
    "persist_candidate_memo_checkpoint",
    "persist_candidate_memo_context_checkpoint",
]
