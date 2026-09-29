"""Reader read model: the library of markdown artifacts and their renderer.

Families are discovered from the artifact trees the platform actually
writes (digests, sector-run reports via the run index, 10-K dossiers, peer
reports, memos); empty families stay out of the library rather than
rendering hollow sections.

Serving is allowlist-first: an artifact renders only if its exact normalized
path is not a symlink/alias, lives under ``data/outputs/``, and is a real
``.md`` file within the size ceiling. Dossier renders attach the sibling
``dossier.json`` claims so every number can defend itself in the Reader.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.autonomous.artifact_financial_audit import (
    PASS as FINANCIAL_INTEGRITY_PASS,
    artifact_decision_eligibility,
    authorized_artifact_bytes,
)
from app.config import get_config
from app.web.readmodel.md import render_markdown_with_toc

_DIGEST_PATTERN = re.compile(r"^digest_(\d{4}-\d{2}-\d{2})\.md$")

MAX_ARTIFACT_BYTES = 5 * 1024 * 1024
MAX_CLAIMS = 500
_FAMILY_CAP = 400


class ArtifactRefused(Exception):
    """The path is not an allowlisted, readable markdown artifact."""


def outputs_dir() -> Path:
    return Path(get_config().outputs_dir)


def _exact_reader_path(path: str | Path) -> Path | None:
    try:
        expanded = Path(path).expanduser()
        lexical = Path(os.path.abspath(os.fspath(expanded)))
        resolved = expanded.resolve()
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
    return lexical if lexical == resolved else None


def _mtime_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def _current_product_eligible(path: Path) -> bool:
    return artifact_decision_eligibility(path) == FINANCIAL_INTEGRITY_PASS


def _digest_items(base: Path) -> list[dict[str, Any]]:
    directory = base / "digests"
    if not directory.is_dir():
        return []
    items = []
    for candidate in directory.iterdir():
        path = _exact_reader_path(candidate)
        if path is None:
            continue
        match = _DIGEST_PATTERN.match(path.name)
        if not match:
            continue
        if not _current_product_eligible(path):
            continue
        items.append(
            {
                "family": "digest",
                "title": f"Daily digest {match.group(1)}",
                "path": str(path),
                "mtime": _mtime_iso(path),
                "meta": {"date": match.group(1)},
            }
        )
    items.sort(key=lambda item: item["meta"]["date"], reverse=True)
    return items


def _run_report_items(ui_conn: sqlite3.Connection | None) -> list[dict[str, Any]]:
    if ui_conn is None:
        return []
    items = []
    for row in ui_conn.execute(
        """
        SELECT slug, sector, market_cap_focus, as_of_date, created_at, report_path
        FROM run_index
        WHERE report_path IS NOT NULL AND decision_eligible = 1
        ORDER BY created_at IS NULL, created_at DESC
        LIMIT ?
        """,
        (_FAMILY_CAP,),
    ):
        path = _exact_reader_path(str(row["report_path"]))
        if path is None or not path.is_file() or not _current_product_eligible(path):
            continue
        sector = row["sector"] or "unknown sector"
        band = row["market_cap_focus"] or "unknown band"
        items.append(
            {
                "family": "run_report",
                "title": f"{sector} · {band} · {row['as_of_date'] or '—'}",
                "path": str(path),
                "mtime": _mtime_iso(path),
                "meta": {
                    "slug": row["slug"],
                    "sector": row["sector"],
                    "band": row["market_cap_focus"],
                },
            }
        )
    return items


def _dossier_items(base: Path) -> list[dict[str, Any]]:
    directory = base / "dossiers"
    if not directory.is_dir():
        return []
    items = []
    for candidate in directory.glob("*/*/dossier.md"):
        path = _exact_reader_path(candidate)
        if path is None:
            continue
        ticker = path.parent.name
        run_label = path.parent.parent.name
        items.append(
            {
                "family": "dossier",
                "title": f"{ticker} 10-K dossier",
                "path": str(path),
                "mtime": _mtime_iso(path),
                "meta": {
                    "ticker": ticker,
                    "run_label": run_label,
                    "has_claims": (path.parent / "dossier.json").is_file(),
                },
            }
        )
    items.sort(key=lambda item: item["mtime"], reverse=True)
    return items[:_FAMILY_CAP]


def _peer_report_items(base: Path) -> list[dict[str, Any]]:
    directory = base / "dossiers"
    if not directory.is_dir():
        return []
    items = []
    for candidate in directory.glob("*/peer_report.md"):
        path = _exact_reader_path(candidate)
        if path is None:
            continue
        items.append(
            {
                "family": "peer_report",
                "title": f"{path.parent.name} peer report",
                "path": str(path),
                "mtime": _mtime_iso(path),
                "meta": {"run_label": path.parent.name},
            }
        )
    items.sort(key=lambda item: item["mtime"], reverse=True)
    return items[:_FAMILY_CAP]


def _memo_items(base: Path) -> list[dict[str, Any]]:
    directory = base / "memos"
    if not directory.is_dir():
        return []
    items = []
    for candidate in directory.glob("*/memo.md"):
        path = _exact_reader_path(candidate)
        if path is None:
            continue
        label = path.parent.name
        ticker, _, memo_date = label.partition("_")
        items.append(
            {
                "family": "memo",
                "title": f"{ticker} memo · {memo_date or '—'}",
                "path": str(path),
                "mtime": _mtime_iso(path),
                "meta": {"ticker": ticker, "date": memo_date or None},
            }
        )
    items.sort(key=lambda item: item["mtime"], reverse=True)
    return items[:_FAMILY_CAP]


_FAMILY_LABELS = {
    "digest": "Daily digests",
    "run_report": "Sector run reports",
    "dossier": "10-K dossiers",
    "peer_report": "Peer reports",
    "memo": "Memos",
}


def library(ui_conn: sqlite3.Connection | None = None) -> dict[str, Any]:
    base = outputs_dir()
    families = []
    for family, items in (
        ("digest", _digest_items(base)),
        ("run_report", _run_report_items(ui_conn)),
        ("dossier", _dossier_items(base)),
        ("peer_report", _peer_report_items(base)),
        ("memo", _memo_items(base)),
    ):
        if items:
            families.append(
                {
                    "family": family,
                    "label": _FAMILY_LABELS[family],
                    "total": len(items),
                    "items": items,
                }
            )
    return {"families": families}


def _load_claims(md_path: Path) -> list[dict[str, Any]] | None:
    sidecar = md_path.parent / "dossier.json"
    if md_path.name != "dossier.md" or not sidecar.is_file():
        return None
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    raw_claims = payload.get("claims") if isinstance(payload, dict) else None
    if not isinstance(raw_claims, list):
        return None
    claims: list[dict[str, Any]] = []
    for raw in raw_claims[:MAX_CLAIMS]:
        if not isinstance(raw, dict):
            continue
        citations = [
            {
                "source_url": str(c.get("source_url") or ""),
                "snippet": str(c.get("snippet") or ""),
                "section_label": str(c.get("section_label") or ""),
            }
            for c in raw.get("citations", [])
            if isinstance(c, dict)
        ]
        claims.append(
            {
                "claim_id": str(raw.get("claim_id") or ""),
                "label": str(raw.get("label") or ""),
                "value": raw.get("value"),
                "unit": raw.get("unit"),
                "citations": citations,
            }
        )
    return claims


def render_artifact(path_str: str) -> dict[str, Any]:
    """Render one allowlisted markdown artifact; raises ArtifactRefused."""
    if not path_str or not str(path_str).strip():
        raise ArtifactRefused("Empty artifact path")
    base = outputs_dir().resolve()
    target = _exact_reader_path(str(path_str))
    if target is None:
        raise ArtifactRefused(f"Artifact path is a symlink or alias: {path_str}")
    if base not in target.parents:
        raise ArtifactRefused(f"Artifact path escapes data/outputs: {path_str}")
    if target.suffix != ".md":
        raise ArtifactRefused(f"Not a markdown artifact: {path_str}")
    if not target.is_file():
        raise ArtifactRefused(f"No such artifact: {path_str}")
    size = target.stat().st_size
    if size > MAX_ARTIFACT_BYTES:
        raise ArtifactRefused(
            f"Artifact exceeds {MAX_ARTIFACT_BYTES // (1024 * 1024)}MB ceiling: {path_str}"
        )
    integrity_status, authorized_bytes = authorized_artifact_bytes(target)
    decision_eligible = integrity_status == FINANCIAL_INTEGRITY_PASS
    if decision_eligible:
        if authorized_bytes is None:
            raise ArtifactRefused("Authorized artifact bytes were unavailable")
        if len(authorized_bytes) > MAX_ARTIFACT_BYTES:
            raise ArtifactRefused(
                f"Artifact exceeds {MAX_ARTIFACT_BYTES // (1024 * 1024)}MB ceiling: {path_str}"
            )
        text = authorized_bytes.decode("utf-8", errors="replace")
    else:
        text = (
            "# Historical artifact — excluded from current decisions\n\n"
            f"Financial-integrity status: **{integrity_status}**. The original artifact "
            "body and claims are suppressed because this record is not decision-eligible."
        )
    html, toc = render_markdown_with_toc(text)
    title = next((entry["text"] for entry in toc if entry["level"] == 1), target.stem)
    return {
        "path": str(target),
        "title": title,
        "html": html,
        "toc": toc,
        "claims": _load_claims(target) if decision_eligible else None,
        "size": size,
        "mtime": _mtime_iso(target),
        "integrity_status": integrity_status,
        "decision_eligible": decision_eligible,
    }
