"""Shared identity and minimum-content contract for v1 candidate reviews."""

from __future__ import annotations

import json
import math
from collections import Counter
from hashlib import sha256
from typing import Any


def candidate_memo_schema_name(ticker: str) -> str:
    return f"autonomous_sector_candidate_memo_{str(ticker).strip().lower()}"


def candidate_memo_is_substantive(payload: Any) -> bool:
    """Require the minimum fields that distinguish a review from an empty call."""
    if not isinstance(payload, dict) or not str(payload.get("thesis") or "").strip():
        return False
    for field in ("key_risks", "falsifiers", "open_questions"):
        values = payload.get(field)
        if not isinstance(values, list) or not any(str(item).strip() for item in values):
            return False
    return True


def provider_usage_attestation(
    payload: Any,
    *,
    include_reused: bool = True,
) -> dict[str, Any]:
    """Return one canonical rollup of every physical provider-usage ledger.

    Sector artifacts can contain provider ledgers at the parent, nested-company,
    validation, and repair levels.  Campaign accounting must cover all of them,
    while avoiding double counting when the same physical row is embedded at
    more than one level.
    """

    raw_records: list[dict[str, Any]] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "provider_usage" and isinstance(child, list):
                    raw_records.extend(
                        row
                        for row in child
                        if isinstance(row, dict)
                        and (include_reused or row.get("reused_from_checkpoint") is not True)
                    )
                else:
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(payload)
    by_fingerprint: dict[str, dict[str, Any]] = {}
    for record in raw_records:
        checkpoint_physical_id = str(record.get("checkpoint_physical_id") or "").strip().lower()
        if len(checkpoint_physical_id) == 64 and all(
            char in "0123456789abcdef" for char in checkpoint_physical_id
        ):
            fingerprint = f"checkpoint:{checkpoint_physical_id}"
        else:
            canonical = json.dumps(
                record,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            fingerprint = sha256(canonical.encode("utf-8")).hexdigest()
        by_fingerprint.setdefault(fingerprint, record)
    records = [by_fingerprint[key] for key in sorted(by_fingerprint)]

    valid = True
    total_cost = 0.0
    bindings: set[tuple[str, str]] = set()
    for record in records:
        provider = str(record.get("provider") or "").strip().lower()
        model = str(record.get("model") or "").strip()
        try:
            cost = float(record.get("cost_estimate_usd") or 0.0)
        except (TypeError, ValueError):
            valid = False
            continue
        if not math.isfinite(cost) or cost < 0:
            valid = False
            continue
        if not provider or not model:
            valid = False
        else:
            bindings.add((provider, model))
        total_cost += cost

    canonical_records = json.dumps(records, sort_keys=True, separators=(",", ":"), default=str)
    return {
        "valid": valid,
        "physical_attempt_count": len(records),
        "cost_estimate_usd": round(total_cost, 6),
        "provider_models": [
            {"provider": provider, "model": model} for provider, model in sorted(bindings)
        ],
        "usage_records_sha256": sha256(canonical_records.encode("utf-8")).hexdigest(),
    }


def provider_usage_tail_not_in_snapshot(
    snapshot: Any,
    attached_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return attached physical rows not already represented in a snapshot."""

    snapshot_records: list[dict[str, Any]] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "provider_usage" and isinstance(child, list):
                    snapshot_records.extend(row for row in child if isinstance(row, dict))
                else:
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    def comparable(record: dict[str, Any]) -> str:
        normalized = {
            key: value
            for key, value in record.items()
            if key
            not in {
                "provider_call_id",
                "reused_from_checkpoint",
                "checkpoint_key_sha256",
                "checkpoint_physical_id",
            }
        }
        return json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )

    visit(snapshot)
    remaining = Counter(comparable(record) for record in snapshot_records)
    tail: list[dict[str, Any]] = []
    for record in attached_records:
        if not isinstance(record, dict):
            continue
        fingerprint = comparable(record)
        if remaining[fingerprint] > 0:
            remaining[fingerprint] -= 1
        else:
            tail.append(record)
    return tail


__all__ = [
    "candidate_memo_is_substantive",
    "candidate_memo_schema_name",
    "provider_usage_attestation",
    "provider_usage_tail_not_in_snapshot",
]
