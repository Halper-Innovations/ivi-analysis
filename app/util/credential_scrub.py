from __future__ import annotations

import json
import os
import sqlite3
import stat
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable

from app.util.credential_hygiene import (
    contains_credential_material,
    sanitize_json_text,
    sanitize_url_credentials,
)


@dataclass(frozen=True)
class DatabaseScrubReport:
    path: str
    rows_scanned: int
    rows_changed: int
    source_urls_changed: int
    raw_json_changed: int
    additional_json_values_changed: int
    tables_changed: tuple[tuple[str, int], ...]
    applied: bool


@dataclass(frozen=True)
class JsonRootScrubReport:
    path: str
    files_scanned: int
    files_changed: int
    applied: bool


@dataclass(frozen=True)
class EnvPermissionReport:
    path: str
    exists: bool
    mode_before: str | None
    mode_after: str | None
    change_required: bool
    applied: bool


@dataclass(frozen=True)
class CredentialScrubReport:
    apply: bool
    key_rotation_confirmed: bool
    databases: tuple[DatabaseScrubReport, ...]
    json_roots: tuple[JsonRootScrubReport, ...]
    env: EnvPermissionReport | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _readonly_sqlite_uri(path: Path) -> str:
    return f"{path.resolve().as_uri()}?mode=ro"


def _json_text_contains_credentials(value: str) -> bool:
    try:
        payload = json.loads(value)
    except (TypeError, ValueError):
        return contains_credential_material(value)
    return contains_credential_material(payload)


def scrub_price_quote_database(
    path: str | Path,
    *,
    apply: bool,
) -> DatabaseScrubReport:
    db_path = Path(path)
    if not db_path.is_file():
        raise FileNotFoundError(f"Price database not found: {db_path}")
    conn = (
        sqlite3.connect(str(db_path))
        if apply
        else sqlite3.connect(_readonly_sqlite_uri(db_path), uri=True)
    )
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA busy_timeout = 300000")
        columns = {
            str(row[1])
            for row in conn.execute("PRAGMA table_info(price_quotes)").fetchall()
        }
        required = {"id", "source_url", "raw_json", "quote_hash"}
        missing = required - columns
        if missing:
            raise RuntimeError(
                "price_quotes schema is missing required columns: "
                + ", ".join(sorted(missing))
            )
        price_rows = conn.execute(
            "SELECT id, source_url, raw_json, quote_hash FROM price_quotes ORDER BY id"
        ).fetchall()
        price_updates: list[tuple[str | None, str, str, int]] = []
        source_urls_changed = 0
        raw_json_changed = 0
        for row in price_rows:
            original_source = str(row["source_url"]) if row["source_url"] is not None else None
            sanitized_source = sanitize_url_credentials(original_source)
            if sanitized_source != original_source:
                source_urls_changed += 1

            original_raw = str(row["raw_json"] or "")
            sanitized_raw, raw_changed = sanitize_json_text(original_raw)
            if raw_changed:
                raw_json_changed += 1
                quote_hash = sha256(sanitized_raw.encode("utf-8")).hexdigest()
            else:
                quote_hash = str(row["quote_hash"] or "")

            if sanitized_source != original_source or raw_changed:
                price_updates.append(
                    (
                        sanitized_source,
                        sanitized_raw,
                        quote_hash,
                        int(row["id"]),
                    )
                )

        json_table_columns = {
            "valuations": ("inputs_json", "outputs_json", "warnings_json"),
            "valuations_measurement": (
                "inputs_json",
                "outputs_json",
                "warnings_json",
            ),
            "valuations_history": (
                "inputs_json",
                "outputs_json",
                "warnings_json",
            ),
        }
        json_updates: list[tuple[str, int, dict[str, str]]] = []
        additional_rows_scanned = 0
        additional_json_values_changed = 0
        table_change_counts: dict[str, int] = {
            "price_quotes": len(price_updates),
        }
        for table, candidate_columns in json_table_columns.items():
            table_columns = {
                str(row[1])
                for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
            }
            columns = tuple(column for column in candidate_columns if column in table_columns)
            if "id" not in table_columns or not columns:
                continue
            table_rows = conn.execute(
                f"SELECT id, {', '.join(columns)} FROM {table} ORDER BY id"
            ).fetchall()
            additional_rows_scanned += len(table_rows)
            table_changed = 0
            for row in table_rows:
                replacements: dict[str, str] = {}
                for column in columns:
                    original = str(row[column] or "")
                    sanitized, was_changed = sanitize_json_text(original)
                    if was_changed:
                        replacements[column] = sanitized
                        additional_json_values_changed += 1
                if replacements:
                    json_updates.append((table, int(row["id"]), replacements))
                    table_changed += 1
            table_change_counts[table] = table_changed

        if apply and (price_updates or json_updates):
            conn.execute("BEGIN IMMEDIATE")
            if price_updates:
                conn.executemany(
                    """
                    UPDATE price_quotes
                    SET source_url = ?, raw_json = ?, quote_hash = ?
                    WHERE id = ?
                    """,
                    price_updates,
                )
            for table, row_id, replacements in json_updates:
                assignments = ", ".join(f"{column} = ?" for column in replacements)
                conn.execute(
                    f"UPDATE {table} SET {assignments} WHERE id = ?",
                    (*replacements.values(), row_id),
                )
            conn.commit()

        if apply:
            remaining = conn.execute(
                "SELECT source_url, raw_json FROM price_quotes ORDER BY id"
            ).fetchall()
            if any(
                contains_credential_material(row["source_url"])
                or _json_text_contains_credentials(str(row["raw_json"] or ""))
                for row in remaining
            ):
                raise RuntimeError(
                    f"Credential scrub verification failed for price_quotes in {db_path}"
                )
            for table, candidate_columns in json_table_columns.items():
                table_columns = {
                    str(row[1])
                    for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
                }
                columns = tuple(
                    column for column in candidate_columns if column in table_columns
                )
                if not columns:
                    continue
                rows = conn.execute(
                    f"SELECT {', '.join(columns)} FROM {table}"
                ).fetchall()
                if any(
                    _json_text_contains_credentials(str(row[column] or ""))
                    for row in rows
                    for column in columns
                ):
                    raise RuntimeError(
                        f"Credential scrub verification failed for {table} in {db_path}"
                    )
        return DatabaseScrubReport(
            path=str(db_path),
            rows_scanned=len(price_rows) + additional_rows_scanned,
            rows_changed=len(price_updates) + len(json_updates),
            source_urls_changed=source_urls_changed,
            raw_json_changed=raw_json_changed,
            additional_json_values_changed=additional_json_values_changed,
            tables_changed=tuple(
                (table, count)
                for table, count in table_change_counts.items()
                if count > 0
            ),
            applied=apply,
        )
    except Exception:
        if apply:
            conn.rollback()
        raise
    finally:
        conn.close()


def _atomic_write_text(path: Path, text: str) -> None:
    original_mode = stat.S_IMODE(path.stat().st_mode)
    temporary = path.with_name(f".{path.name}.credential-scrub.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.chmod(temporary, original_mode)
    temporary.replace(path)


def scrub_json_root(
    path: str | Path,
    *,
    apply: bool,
) -> JsonRootScrubReport:
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"JSON root not found: {root}")
    files = sorted(candidate for candidate in root.rglob("*.json") if candidate.is_file())
    changed = 0
    for candidate in files:
        original = candidate.read_text(encoding="utf-8")
        sanitized, was_changed = sanitize_json_text(original)
        if not was_changed:
            continue
        changed += 1
        if apply:
            _atomic_write_text(candidate, sanitized)
            persisted = candidate.read_text(encoding="utf-8")
            try:
                payload = json.loads(persisted)
            except ValueError as exc:
                raise RuntimeError(
                    f"Credential scrub produced invalid JSON at {candidate}"
                ) from exc
            if contains_credential_material(payload):
                raise RuntimeError(
                    f"Credential scrub verification failed for JSON file {candidate}"
                )
    return JsonRootScrubReport(
        path=str(root),
        files_scanned=len(files),
        files_changed=changed,
        applied=apply,
    )


def secure_env_permissions(
    path: str | Path,
    *,
    apply: bool,
) -> EnvPermissionReport:
    env_path = Path(path)
    if not env_path.exists():
        return EnvPermissionReport(
            path=str(env_path),
            exists=False,
            mode_before=None,
            mode_after=None,
            change_required=False,
            applied=apply,
        )
    mode_before = stat.S_IMODE(env_path.stat().st_mode)
    change_required = mode_before != 0o600
    if apply and change_required:
        os.chmod(env_path, 0o600)
    mode_after = stat.S_IMODE(env_path.stat().st_mode)
    if apply and mode_after != 0o600:
        raise RuntimeError(
            f".env permission verification failed for {env_path}: {mode_after:04o}"
        )
    return EnvPermissionReport(
        path=str(env_path),
        exists=True,
        mode_before=f"{mode_before:04o}",
        mode_after=f"{mode_after:04o}",
        change_required=change_required,
        applied=apply,
    )


def run_credential_scrub(
    *,
    database_paths: Iterable[str | Path] = (),
    json_roots: Iterable[str | Path] = (),
    env_path: str | Path | None = None,
    apply: bool = False,
    key_rotation_confirmed: bool = False,
) -> CredentialScrubReport:
    databases = tuple(dict.fromkeys(Path(path) for path in database_paths))
    roots = tuple(dict.fromkeys(Path(path) for path in json_roots))
    if not databases and not roots and env_path is None:
        raise ValueError("At least one explicit database, JSON root, or .env path is required.")
    if apply and not key_rotation_confirmed:
        raise RuntimeError(
            "KEY_ROTATION_CONFIRMATION_REQUIRED: rotate the exposed provider key first, "
            "then rerun with --confirm-key-rotated."
        )
    database_reports = tuple(
        scrub_price_quote_database(path, apply=apply) for path in databases
    )
    json_reports = tuple(scrub_json_root(path, apply=apply) for path in roots)
    env_report = (
        secure_env_permissions(env_path, apply=apply)
        if env_path is not None
        else None
    )
    return CredentialScrubReport(
        apply=apply,
        key_rotation_confirmed=key_rotation_confirmed,
        databases=database_reports,
        json_roots=json_reports,
        env=env_report,
    )
