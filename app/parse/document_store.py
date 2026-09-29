from __future__ import annotations

from pathlib import Path

from app.config import AppConfig, get_config
from app.util.hashing import sha256_file


def filing_local_path(
    cik: str,
    accession: str,
    primary_doc: str,
    *,
    cfg: AppConfig | None = None,
    create_parent: bool = True,
) -> Path:
    cfg = cfg or get_config()
    safe_accession = accession.replace("-", "")
    folder = cfg.raw_filings_dir / cik / safe_accession
    if create_parent:
        folder.mkdir(parents=True, exist_ok=True)
    return folder / primary_doc


def file_hash(path: Path) -> str:
    return sha256_file(path)
