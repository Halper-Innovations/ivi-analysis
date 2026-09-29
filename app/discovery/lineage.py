"""Exact mutable-row lineage for discovery candidates used by calibration."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import sqlite3
import stat
from datetime import date
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4


DISCOVERY_CANDIDATE_BINDING_SCHEMA_VERSION = "discovery_candidate_report_binding_v3"
DISCOVERY_CANDIDATE_PUBLICATION_SCHEMA_VERSION = "discovery_candidate_publication_v3"
_PUBLICATION_PRODUCER = "app.discovery.runner._persist_candidate"
_MAX_SOURCE_CHAIN_DEPTH = 64
_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9_.-]+")
_CANDIDATE_STATE_FIELDS = {
    "id",
    "ticker",
    "run_id",
    "discovery_score",
    "payload",
    "created_at",
}
_CANDIDATE_RUN_STATE_FIELDS = {
    "run_id",
    "run_as_of_date",
    "seed_hash",
    "config_hash",
    "seed_path",
    "tickers_targeted_json",
    "created_at",
}
_BINDING_FIELDS = {
    "schema_version",
    "candidate_state",
    "candidate_run_state",
    "payload_sha256",
    "candidate_state_sha256",
    "candidate_run_state_sha256",
    "publication_receipt",
}
_PUBLICATION_BINDING_FIELDS = {
    "schema_version",
    "path",
    "sha256",
}
_PUBLICATION_RECEIPT_FIELDS = {
    "schema_version",
    "producer",
    "candidate_state",
    "candidate_run_state",
    "payload_sha256",
    "candidate_state_sha256",
    "candidate_run_state_sha256",
    "source_candidate_binding",
}


def _canonical_bytes(value: Any) -> bytes | None:
    try:
        return (
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ": "),
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError):
        return None


def _canonical_sha256(value: Any) -> str | None:
    encoded = _canonical_bytes(value)
    return hashlib.sha256(encoded).hexdigest() if encoded is not None else None


def _candidate_state(row: Mapping[str, Any] | Any) -> dict[str, Any] | None:
    try:
        values = dict(row)
        candidate_id = values["id"]
        raw_ticker = values["ticker"]
        raw_run_id = values["run_id"]
        raw_score = values["discovery_score"]
        raw_payload = values["payload_json"]
        raw_created_at = values["created_at"]
    except (KeyError, TypeError, ValueError):
        return None
    if isinstance(candidate_id, bool) or not isinstance(candidate_id, int) or candidate_id <= 0:
        return None
    if not isinstance(raw_ticker, str) or raw_ticker != raw_ticker.strip().upper():
        return None
    if not raw_ticker:
        return None
    if not isinstance(raw_run_id, str) or raw_run_id != raw_run_id.strip() or not raw_run_id:
        return None
    if (
        isinstance(raw_score, bool)
        or not isinstance(raw_score, (int, float))
        or not math.isfinite(float(raw_score))
    ):
        return None
    if not isinstance(raw_created_at, str) or raw_created_at != raw_created_at.strip():
        return None
    if not raw_created_at:
        return None
    try:
        payload = json.loads(raw_payload)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    try:
        from app.discovery.schemas import DiscoveryCandidate

        canonical_candidate = DiscoveryCandidate.model_validate(copy.deepcopy(payload)).model_dump(
            mode="json"
        )
    except (TypeError, ValueError):
        return None
    payload_score = payload.get("discovery_score") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload != canonical_candidate
        or payload.get("ticker") != raw_ticker
        or payload.get("run_id") != raw_run_id
        or isinstance(payload_score, bool)
        or not isinstance(payload_score, (int, float))
        or not math.isfinite(float(payload_score))
        or float(payload_score) != float(raw_score)
        or _canonical_sha256(payload) is None
    ):
        return None
    return {
        "id": candidate_id,
        "ticker": raw_ticker,
        "run_id": raw_run_id,
        "discovery_score": float(raw_score),
        "payload": payload,
        "created_at": raw_created_at,
    }


def _candidate_run_state(row: Mapping[str, Any] | Any) -> dict[str, Any] | None:
    try:
        values = dict(row)
        run_id = values["run_id"]
        run_as_of_date = values["run_as_of_date"]
        seed_hash = values["seed_hash"]
        config_hash = values["config_hash"]
        seed_path = values["seed_path"]
        tickers_targeted_json = values["tickers_targeted_json"]
        created_at = values["created_at"]
    except (KeyError, TypeError, ValueError):
        return None
    fields = (
        run_id,
        run_as_of_date,
        seed_hash,
        config_hash,
        seed_path,
        tickers_targeted_json,
        created_at,
    )
    if any(not isinstance(value, str) or value != value.strip() or not value for value in fields):
        return None
    try:
        date.fromisoformat(run_as_of_date)
        targeted = json.loads(tickers_targeted_json)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if (
        not isinstance(targeted, list)
        or any(
            not isinstance(ticker, str) or ticker != ticker.strip().upper() or not ticker
            for ticker in targeted
        )
        or len(targeted) != len(set(targeted))
    ):
        return None
    return {
        "run_id": run_id,
        "run_as_of_date": run_as_of_date,
        "seed_hash": seed_hash,
        "config_hash": config_hash,
        "seed_path": seed_path,
        "tickers_targeted_json": tickers_targeted_json,
        "created_at": created_at,
    }


def _candidate_dates_match_run(
    state: Mapping[str, Any],
    run_state: Mapping[str, Any],
) -> bool:
    payload = state.get("payload")
    if not isinstance(payload, Mapping):
        return False
    run_id = str(state.get("run_id") or "")
    ticker = str(state.get("ticker") or "")
    run_as_of_text = payload.get("run_as_of_date")
    effective_as_of_text = payload.get("effective_as_of_date")
    if (
        run_id != run_state.get("run_id")
        or payload.get("run_id") != run_id
        or run_as_of_text != run_state.get("run_as_of_date")
        or not isinstance(run_as_of_text, str)
        or not isinstance(effective_as_of_text, str)
    ):
        return False
    try:
        run_as_of = date.fromisoformat(run_as_of_text)
        effective_as_of = date.fromisoformat(effective_as_of_text)
        targeted = json.loads(str(run_state.get("tickers_targeted_json") or ""))
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    return effective_as_of <= run_as_of and ticker in targeted


def _cached_candidate_copy_matches_source(
    state: Mapping[str, Any],
    source_candidate_binding: Mapping[str, Any],
) -> bool:
    source_state = source_candidate_binding.get("candidate_state")
    source_run_state = source_candidate_binding.get("candidate_run_state")
    target_payload = state.get("payload")
    if (
        not isinstance(source_state, Mapping)
        or not isinstance(source_run_state, Mapping)
        or not isinstance(target_payload, Mapping)
        or source_state.get("ticker") != state.get("ticker")
        or source_run_state.get("run_as_of_date") != target_payload.get("run_as_of_date")
    ):
        return False
    source_payload = source_state.get("payload")
    if not isinstance(source_payload, Mapping):
        return False
    expected_payload = copy.deepcopy(dict(source_payload))
    expected_payload["run_id"] = target_payload.get("run_id")
    expected_payload["run_as_of_date"] = target_payload.get("run_as_of_date")
    expected_payload["newly_surfaced"] = target_payload.get("newly_surfaced")
    expected_payload["repeat_surfaced"] = target_payload.get("repeat_surfaced")
    return expected_payload == dict(target_payload)


def _publication_payload(
    state: Mapping[str, Any],
    *,
    candidate_run_state: Mapping[str, Any],
    source_candidate_binding: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    payload_sha256 = _canonical_sha256(state.get("payload"))
    state_sha256 = _canonical_sha256(state)
    run_state_sha256 = _canonical_sha256(candidate_run_state)
    if payload_sha256 is None or state_sha256 is None or run_state_sha256 is None:
        return None
    return {
        "schema_version": DISCOVERY_CANDIDATE_PUBLICATION_SCHEMA_VERSION,
        "producer": _PUBLICATION_PRODUCER,
        "candidate_state": dict(state),
        "candidate_run_state": dict(candidate_run_state),
        "payload_sha256": payload_sha256,
        "candidate_state_sha256": state_sha256,
        "candidate_run_state_sha256": run_state_sha256,
        "source_candidate_binding": (
            copy.deepcopy(dict(source_candidate_binding))
            if source_candidate_binding is not None
            else None
        ),
    }


def _publication_receipt_path(state: Mapping[str, Any]) -> Path | None:
    from app.config import get_config

    run_id = str(state.get("run_id") or "")
    ticker = str(state.get("ticker") or "")
    state_sha256 = _canonical_sha256(state)
    if (
        not run_id
        or not ticker
        or state_sha256 is None
        or run_id in {".", ".."}
        or ticker in {".", ".."}
        or _SAFE_COMPONENT.fullmatch(run_id) is None
        or _SAFE_COMPONENT.fullmatch(ticker) is None
    ):
        return None
    root = (Path(get_config().discovery_dir).resolve() / "candidate_publications").resolve()
    return root / run_id / f"{ticker}.{state_sha256}.json"


def _validated_publication_receipt(
    state: Mapping[str, Any],
    *,
    path_text: Any,
    expected_sha256: Any,
    _seen_receipts: frozenset[tuple[str, str]] = frozenset(),
    _depth: int = 0,
) -> tuple[dict[str, str], dict[str, Any] | None, dict[str, Any]] | None:
    """Bind one candidate state to exact immutable bytes at its canonical path."""

    if (
        not isinstance(path_text, str)
        or not isinstance(expected_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
    ):
        return None
    expected_path = _publication_receipt_path(state)
    if expected_path is None:
        return None
    path = Path(path_text).expanduser()
    if not path.is_absolute() or path_text != str(path) or path != expected_path:
        return None
    receipt_identity = (path_text, expected_sha256)
    if _depth >= _MAX_SOURCE_CHAIN_DEPTH or receipt_identity in _seen_receipts:
        return None
    try:
        if path.parent.is_symlink() or path.parent.resolve() != path.parent:
            return None
        path_stat = path.lstat()
        if (
            not stat.S_ISREG(path_stat.st_mode)
            or stat.S_ISLNK(path_stat.st_mode)
            or path_stat.st_nlink != 1
        ):
            return None
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            receipt_bytes = handle.read()
            after = os.fstat(handle.fileno())
    except OSError:
        return None
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_nlink)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_nlink)
    if (
        identity_before != identity_after
        or (before.st_dev, before.st_ino) != (path_stat.st_dev, path_stat.st_ino)
        or before.st_nlink != 1
        or not stat.S_ISREG(before.st_mode)
        or hashlib.sha256(receipt_bytes).hexdigest() != expected_sha256
    ):
        return None
    try:
        payload = json.loads(receipt_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    candidate_run_state = payload.get("candidate_run_state")
    if (
        not isinstance(candidate_run_state, Mapping)
        or set(candidate_run_state) != _CANDIDATE_RUN_STATE_FIELDS
        or _candidate_run_state(candidate_run_state) != dict(candidate_run_state)
        or not _candidate_dates_match_run(state, candidate_run_state)
        or payload.get("candidate_run_state_sha256") != _canonical_sha256(candidate_run_state)
    ):
        return None
    source_candidate_binding = payload.get("source_candidate_binding")
    if source_candidate_binding is not None and not isinstance(
        source_candidate_binding,
        Mapping,
    ):
        return None
    expected_payload = _publication_payload(
        state,
        candidate_run_state=candidate_run_state,
        source_candidate_binding=source_candidate_binding,
    )
    canonical = _canonical_bytes(payload)
    if (
        not isinstance(payload, dict)
        or set(payload) != _PUBLICATION_RECEIPT_FIELDS
        or expected_payload is None
        or payload != expected_payload
        or canonical != receipt_bytes
    ):
        return None
    normalized_source: dict[str, Any] | None = None
    if source_candidate_binding is not None:
        if not serialized_discovery_candidate_binding_is_structurally_valid(
            source_candidate_binding,
            _seen_receipts=_seen_receipts | {receipt_identity},
            _depth=_depth + 1,
        ):
            return None
        if not _cached_candidate_copy_matches_source(state, source_candidate_binding):
            return None
        normalized_source = copy.deepcopy(dict(source_candidate_binding))
    return (
        {
            "schema_version": DISCOVERY_CANDIDATE_PUBLICATION_SCHEMA_VERSION,
            "path": path_text,
            "sha256": expected_sha256,
        },
        normalized_source,
        copy.deepcopy(dict(candidate_run_state)),
    )


def publish_discovery_candidate_row(
    conn: sqlite3.Connection,
    candidate_id: int,
    *,
    source_candidate_binding: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Publish an immutable receipt for one exact row written by the discovery runner."""

    if isinstance(candidate_id, bool) or not isinstance(candidate_id, int) or candidate_id <= 0:
        raise ValueError("candidate_id must be a positive integer")
    normalized_source: dict[str, Any] | None = None
    if source_candidate_binding is not None:
        if not serialized_discovery_candidate_binding_is_current(
            conn,
            source_candidate_binding,
        ):
            raise RuntimeError("cached discovery candidate source binding is not current")
        normalized_source = copy.deepcopy(dict(source_candidate_binding))
    row = conn.execute(
        """
        SELECT id, ticker, run_id, discovery_score, payload_json, created_at
        FROM discovery_candidates
        WHERE id = ?
        LIMIT 1
        """,
        (candidate_id,),
    ).fetchone()
    state = _candidate_state(row) if row is not None else None
    run_row = (
        conn.execute(
            """
            SELECT run_id, run_as_of_date, seed_hash, config_hash, seed_path,
                   tickers_targeted_json, created_at
            FROM discovery_runs
            WHERE run_id = ?
            LIMIT 1
            """,
            (state["run_id"],),
        ).fetchone()
        if state is not None
        else None
    )
    run_state = _candidate_run_state(run_row) if run_row is not None else None
    if state is not None and (
        run_state is None or not _candidate_dates_match_run(state, run_state)
    ):
        raise RuntimeError("discovery candidate dates do not match its exact source run metadata")
    if state is not None and normalized_source is not None:
        source_state = normalized_source["candidate_state"]
        if int(source_state["id"]) == int(state["id"]) or (
            source_state["ticker"] == state["ticker"] and source_state["run_id"] == state["run_id"]
        ):
            raise RuntimeError("discovery candidate cannot cite itself as its cached source")
        if not _cached_candidate_copy_matches_source(state, normalized_source):
            raise RuntimeError(
                "cached discovery candidate does not exactly match its source payload"
            )
    receipt_payload = (
        _publication_payload(
            state,
            candidate_run_state=run_state,
            source_candidate_binding=normalized_source,
        )
        if state is not None and run_state is not None
        else None
    )
    destination = _publication_receipt_path(state) if state is not None else None
    receipt_bytes = _canonical_bytes(receipt_payload) if receipt_payload is not None else None
    if state is None or receipt_payload is None or destination is None or receipt_bytes is None:
        raise RuntimeError("cannot publish malformed discovery candidate state")

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.is_symlink() or destination.parent.resolve() != destination.parent:
        raise RuntimeError("candidate publication directory is not canonical")
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(receipt_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            try:
                current = destination.read_bytes()
            except OSError as exc:
                raise RuntimeError("existing discovery candidate receipt is unreadable") from exc
            if current != receipt_bytes:
                raise RuntimeError(
                    "refusing to replace a discovery candidate publication receipt"
                ) from None
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass

    receipt_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
    validated_publication = _validated_publication_receipt(
        state,
        path_text=str(destination),
        expected_sha256=receipt_sha256,
    )
    if validated_publication is None:
        raise RuntimeError("discovery candidate publication receipt failed exact validation")
    publication, validated_source, validated_run_state = validated_publication
    if validated_source != normalized_source or validated_run_state != run_state:
        raise RuntimeError("discovery candidate publication source binding changed")
    cursor = conn.execute(
        """
        UPDATE discovery_candidates
        SET publication_receipt_path = ?, publication_receipt_sha256 = ?
        WHERE id = ?
          AND ticker = ?
          AND run_id = ?
          AND discovery_score = ?
          AND payload_json = ?
          AND created_at = ?
        """,
        (
            publication["path"],
            publication["sha256"],
            state["id"],
            state["ticker"],
            state["run_id"],
            state["discovery_score"],
            row["payload_json"],
            state["created_at"],
        ),
    )
    if cursor.rowcount != 1:
        raise RuntimeError("discovery candidate changed during publication")
    published_row = conn.execute(
        """
        SELECT id, ticker, run_id, discovery_score, payload_json, created_at,
               publication_receipt_path, publication_receipt_sha256
        FROM discovery_candidates
        WHERE id = ?
        LIMIT 1
        """,
        (state["id"],),
    ).fetchone()
    published_binding = (
        serialize_discovery_candidate_binding(published_row) if published_row is not None else None
    )
    if published_binding is None or not serialized_discovery_candidate_binding_is_current(
        conn,
        published_binding,
    ):
        raise RuntimeError("published discovery candidate source chain is not current")
    return publication


def serialize_discovery_candidate_binding(
    row: Mapping[str, Any] | Any,
) -> dict[str, Any] | None:
    """Serialize an exact candidate only when its independent receipt is valid."""

    state = _candidate_state(row)
    if state is None:
        return None
    try:
        values = dict(row)
        validated_publication = _validated_publication_receipt(
            state,
            path_text=values["publication_receipt_path"],
            expected_sha256=values["publication_receipt_sha256"],
        )
    except (KeyError, TypeError, ValueError):
        return None
    if validated_publication is None:
        return None
    publication, _source_candidate_binding, candidate_run_state = validated_publication
    payload_sha256 = _canonical_sha256(state["payload"])
    state_sha256 = _canonical_sha256(state)
    run_state_sha256 = _canonical_sha256(candidate_run_state)
    if payload_sha256 is None or state_sha256 is None or run_state_sha256 is None:
        return None
    return {
        "schema_version": DISCOVERY_CANDIDATE_BINDING_SCHEMA_VERSION,
        "candidate_state": state,
        "candidate_run_state": candidate_run_state,
        "payload_sha256": payload_sha256,
        "candidate_state_sha256": state_sha256,
        "candidate_run_state_sha256": run_state_sha256,
        "publication_receipt": publication,
    }


def serialized_discovery_candidate_binding_is_structurally_valid(
    binding: Mapping[str, Any] | Any,
    *,
    _seen_receipts: frozenset[tuple[str, str]] = frozenset(),
    _depth: int = 0,
) -> bool:
    """Validate a serialized binding without trusting its supplied hashes."""

    if not isinstance(binding, Mapping) or set(binding) != _BINDING_FIELDS:
        return False
    if binding.get("schema_version") != DISCOVERY_CANDIDATE_BINDING_SCHEMA_VERSION:
        return False
    state = binding.get("candidate_state")
    if not isinstance(state, Mapping) or set(state) != _CANDIDATE_STATE_FIELDS:
        return False
    run_state = binding.get("candidate_run_state")
    if (
        not isinstance(run_state, Mapping)
        or set(run_state) != _CANDIDATE_RUN_STATE_FIELDS
        or _candidate_run_state(run_state) != dict(run_state)
    ):
        return False
    candidate_id = state.get("id")
    ticker = state.get("ticker")
    run_id = state.get("run_id")
    score = state.get("discovery_score")
    created_at = state.get("created_at")
    payload = state.get("payload")
    if (
        isinstance(candidate_id, bool)
        or not isinstance(candidate_id, int)
        or candidate_id <= 0
        or not isinstance(ticker, str)
        or ticker != ticker.strip().upper()
        or not ticker
        or not isinstance(run_id, str)
        or run_id != run_id.strip()
        or not run_id
        or isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
        or not isinstance(created_at, str)
        or created_at != created_at.strip()
        or not created_at
        or not isinstance(payload, Mapping)
        or payload.get("ticker") != ticker
        or payload.get("run_id") != run_id
    ):
        return False
    try:
        from app.discovery.schemas import DiscoveryCandidate

        canonical_candidate = DiscoveryCandidate.model_validate(
            copy.deepcopy(dict(payload))
        ).model_dump(mode="json")
    except (TypeError, ValueError):
        return False
    if canonical_candidate != dict(payload) or float(
        canonical_candidate["discovery_score"]
    ) != float(score):
        return False
    normalized_state = {
        "id": candidate_id,
        "ticker": ticker,
        "run_id": run_id,
        "discovery_score": float(score),
        "payload": dict(payload),
        "created_at": created_at,
    }
    publication = binding.get("publication_receipt")
    if (
        binding.get("payload_sha256") != _canonical_sha256(normalized_state["payload"])
        or binding.get("candidate_state_sha256") != _canonical_sha256(normalized_state)
        or binding.get("candidate_run_state_sha256") != _canonical_sha256(run_state)
        or not _candidate_dates_match_run(normalized_state, run_state)
        or not isinstance(publication, Mapping)
        or set(publication) != _PUBLICATION_BINDING_FIELDS
        or publication.get("schema_version") != DISCOVERY_CANDIDATE_PUBLICATION_SCHEMA_VERSION
    ):
        return False
    validated = _validated_publication_receipt(
        normalized_state,
        path_text=publication.get("path"),
        expected_sha256=publication.get("sha256"),
        _seen_receipts=_seen_receipts,
        _depth=_depth,
    )
    return (
        validated is not None
        and validated[0] == dict(publication)
        and validated[2] == dict(run_state)
    )


def serialized_discovery_candidate_binding_is_current(
    conn: sqlite3.Connection,
    binding: Mapping[str, Any] | Any,
    *,
    _seen_candidates: frozenset[tuple[int, str, str, str]] = frozenset(),
    _depth: int = 0,
) -> bool:
    """Require the binding to equal the one current exact DB row byte-for-byte."""

    if _depth >= _MAX_SOURCE_CHAIN_DEPTH:
        return False
    if not serialized_discovery_candidate_binding_is_structurally_valid(binding):
        return False
    state = binding["candidate_state"]
    candidate_identity = (
        int(state["id"]),
        str(state["run_id"]),
        str(state["ticker"]),
        str(binding["candidate_state_sha256"]),
    )
    if candidate_identity in _seen_candidates:
        return False
    try:
        rows = conn.execute(
            """
            SELECT id, ticker, run_id, discovery_score, payload_json, created_at,
                   publication_receipt_path, publication_receipt_sha256
            FROM discovery_candidates
            WHERE run_id = ? AND ticker = ?
            LIMIT 2
            """,
            (state["run_id"], state["ticker"]),
        ).fetchall()
    except sqlite3.Error:
        return False
    if len(rows) != 1 or int(rows[0]["id"]) != int(state["id"]):
        return False
    try:
        run_row = conn.execute(
            """
            SELECT run_id, run_as_of_date, seed_hash, config_hash, seed_path,
                   tickers_targeted_json, created_at
            FROM discovery_runs
            WHERE run_id = ?
            LIMIT 2
            """,
            (state["run_id"],),
        ).fetchall()
    except sqlite3.Error:
        return False
    if len(run_row) != 1 or _candidate_run_state(run_row[0]) != dict(
        binding["candidate_run_state"]
    ):
        return False
    current = serialize_discovery_candidate_binding(rows[0])
    if current is None or current != dict(binding):
        return False
    publication = binding["publication_receipt"]
    validated_publication = _validated_publication_receipt(
        state,
        path_text=publication.get("path"),
        expected_sha256=publication.get("sha256"),
    )
    if validated_publication is None:
        return False
    _publication, source_candidate_binding, candidate_run_state = validated_publication
    if candidate_run_state != dict(binding["candidate_run_state"]):
        return False
    if source_candidate_binding is None:
        return True
    return serialized_discovery_candidate_binding_is_current(
        conn,
        source_candidate_binding,
        _seen_candidates=_seen_candidates | {candidate_identity},
        _depth=_depth + 1,
    )


def discovery_candidate_bindings_are_current(
    conn: sqlite3.Connection,
    bindings: Any,
) -> bool:
    """Validate one sorted, duplicate-free set of exact current bindings."""

    if not isinstance(bindings, list):
        return False
    identities: list[tuple[str, str, int]] = []
    for binding in bindings:
        if not serialized_discovery_candidate_binding_is_structurally_valid(binding):
            return False
        state = binding["candidate_state"]
        identities.append(
            (
                str(state["run_id"]),
                str(state["ticker"]),
                int(state["id"]),
            )
        )
    if identities != sorted(identities) or len(identities) != len(set(identities)):
        return False
    return all(
        serialized_discovery_candidate_binding_is_current(conn, binding) for binding in bindings
    )


__all__ = [
    "DISCOVERY_CANDIDATE_BINDING_SCHEMA_VERSION",
    "DISCOVERY_CANDIDATE_PUBLICATION_SCHEMA_VERSION",
    "discovery_candidate_bindings_are_current",
    "publish_discovery_candidate_row",
    "serialize_discovery_candidate_binding",
    "serialized_discovery_candidate_binding_is_current",
    "serialized_discovery_candidate_binding_is_structurally_valid",
]
