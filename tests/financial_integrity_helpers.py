"""Literal, materialized financial-integrity fixtures shared by tests."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from app.autonomous.financial_integrity import (
    SPLIT_PROOF_KIND_NO_INTERVENING_SPLIT,
    SPLIT_PROOF_KIND_SPLIT_EVENT,
    canonical_split_proof_cache_path,
    canonical_split_proof_cache_root,
    canonical_split_proof_raw_cache_path,
    split_proof_raw_materialization_envelope,
    split_proof_materialization_envelope,
    stable_quote_hash,
)

_DERIVE_RAW_PAYLOAD = object()


def authorize_valuation_rows(
    monkeypatch: Any,
    tmp_path: Path,
    *,
    cfg: Any,
    run_id: str,
    row_ids: list[int],
    issuer_ciks: dict[str, str],
) -> Path:
    """Bind synthetic valuation rows to one exact canonical source artifact.

    The artifact embeds the writer-form row records before its bytes are
    hashed. The mutable rows are then bound to that exact path/hash and receive
    their final integrity fingerprints, matching the production publication
    order without bypassing authorization.
    """

    from app.autonomous.artifact_financial_audit import (
        AUDIT_SCHEMA_VERSION,
        AUDIT_SCOPE_ID,
        CANONICAL_AUDIT_ROOT_IDS,
    )
    from app.config import get_config
    from app.valuation.lineage import (
        valuation_integrity_fingerprint,
        valuation_source_record,
    )

    if not row_ids:
        raise ValueError("at least one valuation row is required")
    roots = {
        "autonomous_sector": (Path(cfg.runs_dir) / "autonomous_sector").resolve(),
        "analyst_output": Path(cfg.analyst_outputs_dir).resolve(),
        "scan": (Path(cfg.outputs_dir) / "scans").resolve(),
        "research_output": Path(cfg.research_dir).resolve(),
        "watchlist_report": (Path(cfg.outputs_dir) / "digests").resolve(),
    }
    for root in roots.values():
        root.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(cfg.db_path))
    conn.row_factory = sqlite3.Row
    try:
        marks = ",".join("?" for _ in row_ids)
        conn.execute(
            f"UPDATE valuations SET source_run_id = ? WHERE id IN ({marks})",
            (run_id, *row_ids),
        )
        rows = conn.execute(
            f"SELECT * FROM valuations WHERE id IN ({marks}) ORDER BY id",
            tuple(row_ids),
        ).fetchall()
        if len(rows) != len(row_ids):
            raise ValueError("valuation fixture row missing")
        source_records = [valuation_source_record(row) for row in rows]
        if any(record is None for record in source_records):
            raise ValueError("valuation fixture row is not source-record complete")

        tickers = sorted({str(row["ticker"]).strip().upper() for row in rows})
        artifact_path = (
            roots["autonomous_sector"] / run_id / "autonomous_sector_run.json"
        ).resolve()
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "company_packets": [
                        {
                            "ticker": ticker,
                            "issuer_cik": issuer_ciks[ticker],
                            "financial_integrity_status": "PASS",
                        }
                        for ticker in tickers
                    ],
                    "valuation_source_records": source_records,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        source_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        conn.execute(
            f"""
            UPDATE valuations
            SET source_artifact_path = ?, source_artifact_sha256 = ?
            WHERE id IN ({marks})
            """,
            (str(artifact_path), source_sha256, *row_ids),
        )
        rebound_rows = conn.execute(
            f"SELECT * FROM valuations WHERE id IN ({marks}) ORDER BY id",
            tuple(row_ids),
        ).fetchall()
        for row in rebound_rows:
            conn.execute(
                """
                UPDATE valuations
                SET financial_integrity_fingerprint = ?
                WHERE id = ?
                """,
                (valuation_integrity_fingerprint(row), int(row["id"])),
            )
        conn.commit()
    finally:
        conn.close()

    manifest = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "audit_scope_id": AUDIT_SCOPE_ID,
        "generated_at": "2026-07-23T20:00:00Z",
        "complete": True,
        "source_roots": [
            {
                "family": family,
                "root_id": CANONICAL_AUDIT_ROOT_IDS[family],
                "path": str(root),
            }
            for family, root in roots.items()
        ],
        "summary": {
            "artifacts_scanned": 1,
            "tickers_scanned": len(tickers),
            "violations": 0,
            "violations_by_invariant": {},
            "affected_run_ids": 0,
            "affected_tickers": 0,
            "affected_run_id_values": [],
            "affected_ticker_values": [],
            "earliest_date": None,
            "latest_date": None,
            "llm_consumed_violation_count": 0,
            "source_artifacts_rewritten": 0,
        },
        "invalid_run_ids": [],
        "artifacts": [
            {
                "path": str(artifact_path),
                "family": "autonomous_sector",
                "sha256": source_sha256,
                "integrity_status": "PASS",
                "decision_eligible": True,
                "run_id": run_id,
            }
        ],
        "violations": [],
    }
    manifest_path = (tmp_path / f"{run_id}_financial_integrity_manifest.json").resolve()
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    monkeypatch.setenv("VOE_FINANCIAL_INTEGRITY_MANIFEST", str(manifest_path))
    get_config.cache_clear()
    return artifact_path


def materialized_split_proof(
    record: dict[str, object],
    *,
    raw_payload_override: Any = _DERIVE_RAW_PAYLOAD,
) -> dict[str, object]:
    """Persist provider material plus its exact derived proof cache envelope."""

    bound_record = dict(record)
    source_reference = str(bound_record.get("source_reference") or "")
    parsed = urlparse(source_reference)
    host = parsed.netloc.lower().split(":", 1)[0]
    ticker = str(bound_record.get("ticker") or "").strip().upper()
    if host == "sec.gov" or host.endswith(".sec.gov"):
        source_provider = "sec"
        match = re.fullmatch(
            r"/Archives/edgar/data/(\d+)/(\d{18})/[^/]+",
            parsed.path,
            flags=re.IGNORECASE,
        )
        if match is None:
            raise ValueError("SEC split fixture requires canonical archive URL")
        issuer_cik = match.group(1).zfill(10)
        compact_accession = match.group(2)
        accession = f"{compact_accession[:10]}-{compact_accession[10:12]}-{compact_accession[12:]}"
        proof_kind = SPLIT_PROOF_KIND_SPLIT_EVENT
        retrieved_at = str(bound_record.get("retrieved_at") or bound_record.get("filed_date") or "")
        factor = float(bound_record["factor"])
        effective_date = str(bound_record["effective_date"])
        bound_record.update(
            {
                "source_provider": source_provider,
                "issuer_cik": issuer_cik,
                "provider_symbol": ticker,
                "proof_kind": proof_kind,
                "retrieved_at": retrieved_at,
                "accession": accession,
            }
        )
        document_bytes = (
            f"Issuer declared a {factor:g}-for-1 stock split effective {effective_date}."
        ).encode("utf-8")
        cache_root = canonical_split_proof_cache_root().parents[1]
        document_path = cache_root / "filings" / issuer_cik / accession / "primary_document.html"
        document_path.parent.mkdir(parents=True, exist_ok=True)
        document_path.write_bytes(document_bytes)
        raw_payload: Any = {
            "issuer_cik": issuer_cik,
            "accession": accession,
            "source_reference": source_reference,
            "document_relative_path": document_path.relative_to(cache_root).as_posix(),
            "document_sha256": hashlib.sha256(document_bytes).hexdigest(),
        }
    elif host == "eodhd.com" or host.endswith(".eodhd.com"):
        source_provider = "eodhd"
        issuer_cik = str(bound_record.get("issuer_cik") or "0000000001").zfill(10)
        proof_kind = (
            SPLIT_PROOF_KIND_SPLIT_EVENT
            if bound_record.get("factor") is not None
            else SPLIT_PROOF_KIND_NO_INTERVENING_SPLIT
        )
        retrieved_at = str(
            bound_record.get("retrieved_at")
            or bound_record.get("verified_as_of")
            or bound_record.get("filed_date")
            or ""
        )
        provider_symbol = parsed.path.rstrip("/").rsplit("/", 1)[-1].upper()
        bound_record.update(
            {
                "source_provider": source_provider,
                "issuer_cik": issuer_cik,
                "provider_symbol": provider_symbol,
                "proof_kind": proof_kind,
                "retrieved_at": retrieved_at,
            }
        )
        raw_payload = (
            [
                {
                    "date": str(bound_record["effective_date"]),
                    "split": f"{float(bound_record['factor']):g}/1",
                }
            ]
            if proof_kind == SPLIT_PROOF_KIND_SPLIT_EVENT
            else []
        )
    else:
        raise ValueError("split fixture requires SEC or EODHD source material")

    if raw_payload_override is not _DERIVE_RAW_PAYLOAD:
        raw_payload = raw_payload_override

    raw_envelope = split_proof_raw_materialization_envelope(bound_record, raw_payload)
    raw_bytes = json.dumps(
        raw_envelope,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    raw_path = canonical_split_proof_raw_cache_path(bound_record, raw_payload)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(raw_bytes)
    bound_record.update(
        {
            "raw_relative_path": raw_path.relative_to(
                canonical_split_proof_cache_root()
            ).as_posix(),
            "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "raw_payload_sha256": raw_envelope["payload_sha256"],
        }
    )

    envelope = split_proof_materialization_envelope(bound_record)
    payload = json.dumps(
        envelope,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    path = canonical_split_proof_cache_path(bound_record)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        **bound_record,
        "materialized_path": str(path),
        "materialized_sha256": digest,
    }


def materialized_no_split_proof(
    *,
    ticker: str,
    period_start: str,
    period_end: str,
    verified_as_of: str | None = None,
    issuer_cik: str = "0000000001",
) -> dict[str, object]:
    return materialized_split_proof(
        {
            "ticker": ticker.upper(),
            "status": "PASS",
            "period_start": period_start,
            "period_end": period_end,
            "verified_as_of": verified_as_of or period_end,
            "issuer_cik": issuer_cik,
            "source": "fixture_corporate_actions",
            "source_reference": f"https://eodhd.com/api/splits/{ticker.upper()}",
        }
    )


def canonicalize_financial_packet(
    packet: Any,
    *,
    as_of_date: str | None = None,
    shares_mm: float | None = None,
) -> Any:
    """Mutate a synthetic packet into a complete literal canonical baseline."""

    def get(name: str, default: Any = None) -> Any:
        if isinstance(packet, MutableMapping):
            return packet.get(name, default)
        return getattr(packet, name, default)

    def set_value(name: str, value: Any) -> None:
        if isinstance(packet, MutableMapping):
            packet[name] = value
        else:
            setattr(packet, name, value)

    ticker = str(get("ticker") or "").strip().upper()
    if not ticker:
        raise ValueError("canonical test packet requires ticker")
    price = float(get("current_price") or 0.0)
    if price <= 0:
        raise ValueError("canonical test packet requires positive current_price")
    effective_as_of = str(
        as_of_date
        or get("current_price_as_of_date")
        or get("market_cap_effective_as_of_date")
        or ""
    )[:10]
    if not effective_as_of:
        raise ValueError("canonical test packet requires as_of_date")
    normalized_shares = float(shares_mm or get("shares_outstanding_mm") or 10.0)
    quote_source = str(get("current_price_source") or "fixture_quote")
    quote_source_url = str(
        get("current_price_source_url") or f"https://example.test/quotes/{ticker}"
    )
    shares_as_of = str(get("shares_as_of_date") or effective_as_of)[:10]
    shares_filed_date = str(get("shares_filed_date") or effective_as_of)[:10]
    companyfacts_url = (
        f"https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json#{ticker}-shares"
    )

    set_value("ticker", ticker)
    set_value("current_price", price)
    set_value("current_price_unit", "USD_per_share")
    set_value("current_price_as_of_date", effective_as_of)
    set_value("current_price_currency", "USD")
    set_value("current_price_source", quote_source)
    set_value("current_price_source_url", quote_source_url)
    set_value("price_basis", "UNADJUSTED")
    set_value("raw_price", price)
    set_value("split_adjustment_factor", 1.0)
    set_value("split_effective_date", None)
    set_value(
        "split_lineage_proof",
        materialized_no_split_proof(
            ticker=ticker,
            period_start=shares_as_of,
            period_end=effective_as_of,
        ),
    )

    set_value("shares_outstanding_mm", normalized_shares)
    set_value("raw_shares_outstanding_mm", normalized_shares)
    set_value("raw_shares_source_value", normalized_shares * 1_000_000.0)
    set_value("raw_shares_source_unit", "shares")
    set_value("shares_unit", "shares_millions")
    set_value("shares_basis", "UNADJUSTED")
    set_value("shares_as_of_date", shares_as_of)
    set_value("shares_filed_date", shares_filed_date)
    set_value("shares_source", "SEC_COMPANYFACTS")
    set_value("shares_source_url", companyfacts_url)

    issuer_quote_ratio = float(get("issuer_quote_ratio") or 1.0)
    set_value("issuer_cik", "0000000001")
    set_value("issuer_primary_ticker", ticker)
    set_value("issuer_listed_tickers", [ticker])
    set_value("security_role", "PRIMARY")
    set_value("is_secondary_class", False)
    set_value("is_adr", False)
    set_value("identity_source", "sec_submissions_exchange_binding")
    set_value(
        "identity_source_url",
        f"https://data.sec.gov/submissions/CIK0000000001.json#{ticker}",
    )
    set_value("identity_as_of_date", effective_as_of)
    set_value("identity_confidence", "HIGH")
    market_cap_mm = price * normalized_shares / issuer_quote_ratio
    set_value("issuer_quote_ratio", issuer_quote_ratio)
    set_value("market_cap_mm", market_cap_mm)
    set_value("market_cap_unit", "USD_millions")
    set_value("market_cap_source", "price_times_shares")
    set_value(
        "market_cap_method",
        "price_times_shares_divided_by_issuer_quote_ratio",
    )
    set_value("market_cap_effective_as_of_date", effective_as_of)

    snapshot_id = stable_quote_hash(
        ticker=ticker,
        price=price,
        as_of_date=effective_as_of,
        currency="USD",
        source=quote_source,
        source_url=quote_source_url,
        price_basis="UNADJUSTED",
        raw_price=price,
        split_adjustment_factor=1.0,
        split_effective_date=None,
    )
    set_value("quote_snapshot_id", snapshot_id)
    set_value("cap_stage_price", price)
    set_value("cap_stage_price_as_of_date", effective_as_of)
    set_value("cap_stage_price_currency", "USD")
    set_value("cap_stage_price_source", quote_source)
    set_value("cap_stage_price_source_url", quote_source_url)
    set_value("cap_stage_quote_snapshot_id", snapshot_id)
    return packet
