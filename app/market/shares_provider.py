from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from app.config import AppConfig, get_config
from app.db import get_db, utc_now_iso


SharesResolveVia = Literal["COVER_PAGE", "XBRL_CONCEPT", "CACHED", "UNKNOWN"]
SharesConfidence = Literal["HIGH", "MEDIUM", "LOW"]
SharesReasonCode = Literal[
    "NO_FILINGS",
    "COVER_PAGE_PARSE_MISS",
    "XBRL_MISS",
    "STALE_CACHE",
    "BUDGET_EXHAUSTED",
    "EXCEPTION",
]


@dataclass(eq=True, frozen=True)
class SharesSnapshot:
    ticker: str
    as_of_date: str
    shares_outstanding: float
    unit: str = "shares"
    source: str = "filings"
    retrieved_at: str = ""
    filing_accession: str | None = None
    filing_date: str | None = None
    url: str | None = None
    confidence: SharesConfidence = "MEDIUM"
    resolved_via: SharesResolveVia = "UNKNOWN"
    reason_code: SharesReasonCode | None = None
    evidence_id: str | None = None


class SharesProvider(Protocol):
    provider_name: str

    def get_shares_asof(self, ticker: str, as_of_date: str, run_id: str | None = None) -> SharesSnapshot | None:
        ...

    def get_last_diagnostic(self, ticker: str, as_of_date: str, run_id: str | None = None) -> dict[str, Any] | None:
        ...


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float))


def _safe_parse_date(value: str) -> bool:
    try:
        datetime.strptime(str(value).strip(), "%Y-%m-%d")
        return True
    except Exception:
        return False


def _empty_diag(*, ticker: str, as_of_date: str, cache_path: Path, run_id: str | None) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "requested_as_of": as_of_date,
        "run_id": run_id,
        "provider_attempts": [],
        "cache": {
            "hit": False,
            "path": str(cache_path),
            "snapshot_found": False,
            "cached_as_of_used": None,
        },
        "result": {
            "status": "UNKNOWN",
            "reason_code": "NO_FILINGS",
            "reason_detail": "No shares snapshot resolved.",
        },
        "output_fields": {
            "shares_outstanding": "UNKNOWN",
            "shares_asof_used": None,
            "shares_source": None,
            "confidence": None,
            "resolved_via": "UNKNOWN",
            "evidence_id": None,
        },
        "derived_from": [],
    }


def _as_snapshot(payload: dict[str, Any] | None) -> SharesSnapshot | None:
    if not isinstance(payload, dict):
        return None
    ticker = str(payload.get("ticker") or "").upper()
    as_of_date = str(payload.get("as_of_date") or "")
    shares = payload.get("shares_outstanding")
    if not ticker or not as_of_date or not _is_num(shares) or float(shares) <= 0:
        return None
    confidence = str(payload.get("confidence") or "MEDIUM").upper()
    if confidence not in {"HIGH", "MEDIUM", "LOW"}:
        confidence = "MEDIUM"
    resolved_via = str(payload.get("resolved_via") or "UNKNOWN").upper()
    if resolved_via not in {"COVER_PAGE", "XBRL_CONCEPT", "CACHED", "UNKNOWN"}:
        resolved_via = "UNKNOWN"
    reason_code = payload.get("reason_code")
    if reason_code is not None:
        reason_code = str(reason_code).upper()
    return SharesSnapshot(
        ticker=ticker,
        as_of_date=as_of_date,
        shares_outstanding=float(shares),
        unit=str(payload.get("unit") or "shares"),
        source=str(payload.get("source") or "filings"),
        retrieved_at=str(payload.get("retrieved_at") or utc_now_iso()),
        filing_accession=str(payload.get("filing_accession")) if payload.get("filing_accession") else None,
        filing_date=str(payload.get("filing_date")) if payload.get("filing_date") else None,
        url=str(payload.get("url")) if payload.get("url") else None,
        confidence=confidence,  # type: ignore[arg-type]
        resolved_via=resolved_via,  # type: ignore[arg-type]
        reason_code=reason_code,  # type: ignore[arg-type]
        evidence_id=str(payload.get("evidence_id")) if payload.get("evidence_id") else None,
    )


def _snapshot_payload(snapshot: SharesSnapshot) -> dict[str, Any]:
    return {
        "ticker": snapshot.ticker,
        "as_of_date": snapshot.as_of_date,
        "shares_outstanding": float(snapshot.shares_outstanding),
        "unit": snapshot.unit or "shares",
        "source": snapshot.source,
        "retrieved_at": snapshot.retrieved_at,
        "filing_accession": snapshot.filing_accession,
        "filing_date": snapshot.filing_date,
        "url": snapshot.url,
        "confidence": snapshot.confidence,
        "resolved_via": snapshot.resolved_via,
        "reason_code": snapshot.reason_code,
        "evidence_id": snapshot.evidence_id,
    }


def _parse_shares_value(value_json: Any) -> float | None:
    payload = value_json
    if isinstance(value_json, str):
        try:
            payload = json.loads(value_json)
        except Exception:
            payload = {}
    if not isinstance(payload, dict):
        return None
    value = payload.get("value")
    if not _is_num(value):
        return None
    shares = float(value)
    if shares <= 0:
        return None
    return shares


def _load_run_scoped_snapshot(
    *,
    cfg: AppConfig,
    run_id: str,
    ticker: str,
    as_of_date: str,
) -> tuple[SharesSnapshot | None, Path]:
    path = cfg.outputs_dir / "shares" / run_id / f"{ticker.upper()}.json"
    if not path.exists():
        return None, path
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None, path
    if not isinstance(payload, dict):
        return None, path
    if str(payload.get("status") or "").upper() != "OK":
        return None, path
    if str(payload.get("requested_as_of_date") or "") != str(as_of_date):
        return None, path
    snapshot = _as_snapshot(payload.get("snapshot") if isinstance(payload.get("snapshot"), dict) else None)
    return snapshot, path


class SharesCache:
    def __init__(self, cfg: AppConfig | None = None) -> None:
        self.cfg = cfg or get_config()
        self.base_dir = self.cfg.cache_dir / "shares"
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def path_for_ticker(self, ticker: str) -> Path:
        return self.base_dir / f"{ticker.upper()}.json"

    def load(self, ticker: str, *, requested_as_of_date: str) -> tuple[SharesSnapshot | None, bool]:
        path = self.path_for_ticker(ticker)
        if not path.exists():
            return None, False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None, True
        entries = payload.get("entries") if isinstance(payload, dict) else None
        if not isinstance(entries, list):
            return None, True
        for row in entries:
            if not isinstance(row, dict):
                continue
            if str(row.get("requested_as_of_date") or "") != str(requested_as_of_date):
                continue
            snapshot = _as_snapshot(row.get("snapshot") if isinstance(row.get("snapshot"), dict) else None)
            if snapshot is not None:
                return snapshot, True
        return None, True

    def store(self, ticker: str, *, requested_as_of_date: str, snapshot: SharesSnapshot) -> SharesSnapshot:
        path = self.path_for_ticker(ticker)
        entries: list[dict[str, Any]] = []
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                existing = payload.get("entries") if isinstance(payload, dict) else []
                if isinstance(existing, list):
                    entries = [row for row in existing if isinstance(row, dict)]
            except Exception:
                entries = []

        for row in entries:
            if str(row.get("requested_as_of_date") or "") != str(requested_as_of_date):
                continue
            existing_snapshot = _as_snapshot(row.get("snapshot") if isinstance(row.get("snapshot"), dict) else None)
            if existing_snapshot is not None:
                return existing_snapshot

        entries.append(
            {
                "requested_as_of_date": str(requested_as_of_date),
                "snapshot": _snapshot_payload(snapshot),
            }
        )
        entries.sort(key=lambda row: str(row.get("requested_as_of_date") or ""))
        path.write_text(
            json.dumps({"ticker": ticker.upper(), "entries": entries}, indent=2),
            encoding="utf-8",
        )
        return snapshot


class FilingsSharesProvider:
    provider_name = "shares_filings"

    def __init__(
        self,
        cfg: AppConfig | None = None,
        *,
        cache: SharesCache | None = None,
        cache_only: bool = False,
    ) -> None:
        self.cfg = cfg or get_config()
        self.cache = cache or SharesCache(self.cfg)
        self.cache_only = bool(cache_only)
        self._diag: dict[tuple[str, str, str | None], dict[str, Any]] = {}

    def _set_diag(self, ticker: str, as_of_date: str, run_id: str | None, payload: dict[str, Any]) -> None:
        self._diag[(ticker.upper(), str(as_of_date), run_id)] = payload

    def get_last_diagnostic(self, ticker: str, as_of_date: str, run_id: str | None = None) -> dict[str, Any] | None:
        payload = self._diag.get((ticker.upper(), str(as_of_date), run_id))
        return json.loads(json.dumps(payload)) if isinstance(payload, dict) else None

    def _resolve_from_filings(
        self,
        *,
        ticker: str,
        as_of_date: str,
        diag: dict[str, Any],
    ) -> SharesSnapshot | None:
        with get_db() as conn:
            rows = conn.execute(
                """
                SELECT
                    f.accession,
                    f.filing_date,
                    f.primary_doc_url,
                    ef.value_json,
                    ef.source_url,
                    ef.snippet,
                    ef.section_label,
                    ef.id
                FROM extracted_facts ef
                JOIN filings f ON f.id = ef.filing_id
                WHERE f.ticker = ?
                  AND ef.fact_type = 'shares_outstanding'
                  AND COALESCE(f.filing_date, '1900-01-01') <= ?
                ORDER BY COALESCE(f.filing_date, '1900-01-01') DESC, ef.id DESC
                """,
                (ticker.upper(), as_of_date),
            ).fetchall()
        if not rows:
            diag["provider_attempts"].append(
                {
                    "provider": self.provider_name,
                    "status": "NO_FILINGS",
                    "error_code": "NO_FILINGS",
                    "error_summary": "No filings with shares_outstanding facts available.",
                }
            )
            diag["result"] = {
                "status": "UNKNOWN",
                "reason_code": "NO_FILINGS",
                "reason_detail": "No filings with shares_outstanding facts found for ticker/as-of.",
            }
            return None

        cover_rows = [row for row in rows if str(row["section_label"] or "").strip().lower() == "cover_page"]
        xbrl_rows = [row for row in rows if str(row["section_label"] or "").strip().lower() != "cover_page"]

        for row in cover_rows:
            shares = _parse_shares_value(row["value_json"])
            if shares is None:
                continue
            snapshot = SharesSnapshot(
                ticker=ticker.upper(),
                as_of_date=str(row["filing_date"] or as_of_date),
                shares_outstanding=float(shares),
                unit="shares",
                source="filings_cover_page",
                retrieved_at=utc_now_iso(),
                filing_accession=str(row["accession"] or "") or None,
                filing_date=str(row["filing_date"] or "") or None,
                url=str(row["source_url"] or row["primary_doc_url"] or "") or None,
                confidence="HIGH",
                resolved_via="COVER_PAGE",
            )
            diag["provider_attempts"].append(
                {
                    "provider": self.provider_name,
                    "status": "PROVIDER_OK",
                    "source": "COVER_PAGE",
                    "accession": snapshot.filing_accession,
                    "filing_date": snapshot.filing_date,
                }
            )
            return snapshot

        if cover_rows:
            diag["provider_attempts"].append(
                {
                    "provider": self.provider_name,
                    "status": "COVER_PAGE_PARSE_MISS",
                    "error_code": "COVER_PAGE_PARSE_MISS",
                    "error_summary": "Cover-page shares facts exist but no positive numeric value parsed.",
                }
            )

        for row in xbrl_rows:
            shares = _parse_shares_value(row["value_json"])
            if shares is None:
                continue
            snapshot = SharesSnapshot(
                ticker=ticker.upper(),
                as_of_date=str(row["filing_date"] or as_of_date),
                shares_outstanding=float(shares),
                unit="shares",
                source="filings_xbrl",
                retrieved_at=utc_now_iso(),
                filing_accession=str(row["accession"] or "") or None,
                filing_date=str(row["filing_date"] or "") or None,
                url=str(row["source_url"] or row["primary_doc_url"] or "") or None,
                confidence="MEDIUM",
                resolved_via="XBRL_CONCEPT",
            )
            diag["provider_attempts"].append(
                {
                    "provider": self.provider_name,
                    "status": "PROVIDER_OK",
                    "source": "XBRL_CONCEPT",
                    "accession": snapshot.filing_accession,
                    "filing_date": snapshot.filing_date,
                }
            )
            return snapshot

        reason = "XBRL_MISS" if xbrl_rows else "COVER_PAGE_PARSE_MISS"
        diag["provider_attempts"].append(
            {
                "provider": self.provider_name,
                "status": reason,
                "error_code": reason,
                "error_summary": "No positive numeric shares value found in filing facts.",
            }
        )
        diag["result"] = {
            "status": "UNKNOWN",
            "reason_code": reason,
            "reason_detail": "No positive numeric shares value found in cover-page or XBRL-like facts.",
        }
        return None

    def get_shares_asof(self, ticker: str, as_of_date: str, run_id: str | None = None) -> SharesSnapshot | None:
        ticker_norm = str(ticker).upper().strip()
        as_of_norm = str(as_of_date).strip()
        cache_path = self.cache.path_for_ticker(ticker_norm)
        diag = _empty_diag(ticker=ticker_norm, as_of_date=as_of_norm, cache_path=cache_path, run_id=run_id)
        self._set_diag(ticker_norm, as_of_norm, run_id, diag)

        if not ticker_norm or not _safe_parse_date(as_of_norm):
            diag["result"] = {
                "status": "UNKNOWN",
                "reason_code": "EXCEPTION",
                "reason_detail": "Invalid ticker or as_of_date format for shares lookup.",
            }
            self._set_diag(ticker_norm, as_of_norm, run_id, diag)
            return None

        try:
            if run_id:
                scoped_snapshot, scoped_path = _load_run_scoped_snapshot(
                    cfg=self.cfg,
                    run_id=run_id,
                    ticker=ticker_norm,
                    as_of_date=as_of_norm,
                )
                if scoped_snapshot is not None:
                    snapshot = SharesSnapshot(
                        **{
                            **scoped_snapshot.__dict__,
                            "resolved_via": "CACHED",
                        }
                    )
                    diag["cache"] = {
                        "hit": True,
                        "path": str(scoped_path),
                        "snapshot_found": True,
                        "cached_as_of_used": snapshot.as_of_date,
                    }
                    diag["provider_attempts"].append(
                        {
                            "provider": "run_scoped_shares_fetch",
                            "status": "CACHE_HIT",
                            "path": str(scoped_path),
                        }
                    )
                    diag["result"] = {
                        "status": "OK",
                        "reason_code": "CACHE_HIT",
                        "reason_detail": "Resolved from outputs/shares/<run_id>.",
                    }
                    diag["output_fields"] = {
                        "shares_outstanding": float(snapshot.shares_outstanding),
                        "shares_asof_used": snapshot.as_of_date,
                        "shares_source": "run_scoped_output",
                        "confidence": snapshot.confidence,
                        "resolved_via": "CACHED",
                        "evidence_id": snapshot.evidence_id,
                    }
                    self._set_diag(ticker_norm, as_of_norm, run_id, diag)
                    return snapshot

            cached, cache_exists = self.cache.load(ticker_norm, requested_as_of_date=as_of_norm)
            if cached is not None:
                snapshot = SharesSnapshot(
                    **{
                        **cached.__dict__,
                        "resolved_via": "CACHED",
                    }
                )
                diag["cache"] = {
                    "hit": True,
                    "path": str(cache_path),
                    "snapshot_found": True,
                    "cached_as_of_used": snapshot.as_of_date,
                }
                diag["provider_attempts"].append(
                    {
                        "provider": self.provider_name,
                        "status": "CACHE_HIT",
                        "path": str(cache_path),
                    }
                )
                diag["result"] = {
                    "status": "OK",
                    "reason_code": "CACHE_HIT",
                    "reason_detail": "Resolved from data/cache/shares.",
                }
                diag["output_fields"] = {
                    "shares_outstanding": float(snapshot.shares_outstanding),
                    "shares_asof_used": snapshot.as_of_date,
                    "shares_source": "disk_cache",
                    "confidence": snapshot.confidence,
                    "resolved_via": "CACHED",
                    "evidence_id": snapshot.evidence_id,
                }
                self._set_diag(ticker_norm, as_of_norm, run_id, diag)
                return snapshot

            diag["provider_attempts"].append(
                {
                    "provider": self.provider_name,
                    "status": "CACHE_MISS",
                    "path": str(cache_path),
                }
            )
            if self.cache_only:
                reason_code = "STALE_CACHE" if cache_exists else "NO_FILINGS"
                diag["result"] = {
                    "status": "UNKNOWN",
                    "reason_code": reason_code,
                    "reason_detail": "Cache-only mode enabled and no exact shares snapshot was found.",
                }
                self._set_diag(ticker_norm, as_of_norm, run_id, diag)
                return None

            snapshot = self._resolve_from_filings(ticker=ticker_norm, as_of_date=as_of_norm, diag=diag)
            if snapshot is None:
                self._set_diag(ticker_norm, as_of_norm, run_id, diag)
                return None

            snapshot = self.cache.store(
                ticker_norm,
                requested_as_of_date=as_of_norm,
                snapshot=snapshot,
            )
            diag["cache"] = {
                "hit": False,
                "path": str(cache_path),
                "snapshot_found": False,
                "cached_as_of_used": snapshot.as_of_date,
            }
            diag["result"] = {
                "status": "OK",
                "reason_code": "PROVIDER_OK",
                "reason_detail": "Resolved shares from filings facts.",
            }
            diag["output_fields"] = {
                "shares_outstanding": float(snapshot.shares_outstanding),
                "shares_asof_used": snapshot.as_of_date,
                "shares_source": snapshot.source,
                "confidence": snapshot.confidence,
                "resolved_via": snapshot.resolved_via,
                "evidence_id": snapshot.evidence_id,
            }
            self._set_diag(ticker_norm, as_of_norm, run_id, diag)
            return snapshot
        except Exception as exc:  # noqa: BLE001
            reason_code = "BUDGET_EXHAUSTED" if "budget" in str(exc).lower() else "EXCEPTION"
            diag["provider_attempts"].append(
                {
                    "provider": self.provider_name,
                    "status": reason_code,
                    "error_code": reason_code,
                    "error_summary": str(exc),
                }
            )
            diag["result"] = {
                "status": "UNKNOWN",
                "reason_code": reason_code,
                "reason_detail": str(exc),
            }
            self._set_diag(ticker_norm, as_of_norm, run_id, diag)
            return None


def build_shares_provider(*, cfg: AppConfig | None = None, cache_only: bool = False) -> SharesProvider:
    return FilingsSharesProvider(cfg=cfg or get_config(), cache_only=cache_only)


def write_shares_for_run(
    *,
    tickers: list[str],
    as_of_date: str,
    run_id: str,
    cache_only: bool = False,
    cfg: AppConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or get_config()
    out_dir = cfg.outputs_dir / "shares" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    provider = build_shares_provider(cfg=cfg, cache_only=cache_only)

    rows: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {}
    ok_count = 0
    for ticker in sorted({str(symbol).strip().upper() for symbol in tickers if str(symbol).strip()}):
        snapshot = provider.get_shares_asof(ticker=ticker, as_of_date=as_of_date, run_id=run_id)
        diag = provider.get_last_diagnostic(ticker=ticker, as_of_date=as_of_date, run_id=run_id)
        result = diag.get("result") if isinstance(diag, dict) else {}
        reason_code = str((result or {}).get("reason_code") or ("PROVIDER_OK" if snapshot is not None else "EXCEPTION"))
        reason_counts[reason_code] = reason_counts.get(reason_code, 0) + 1
        status = "OK" if snapshot is not None else "UNKNOWN"
        if status == "OK":
            ok_count += 1

        payload = {
            "ticker": ticker,
            "requested_as_of_date": as_of_date,
            "status": status,
            "reason_code": reason_code,
            "diagnostic": diag,
            "snapshot": _snapshot_payload(snapshot) if snapshot is not None else None,
        }
        ticker_path = out_dir / f"{ticker}.json"
        ticker_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        rows.append(
            {
                "ticker": ticker,
                "status": status,
                "reason_code": reason_code,
                "shares_outstanding": float(snapshot.shares_outstanding) if snapshot is not None else None,
                "as_of_used": snapshot.as_of_date if snapshot is not None else None,
                "source": snapshot.source if snapshot is not None else None,
                "path": str(ticker_path),
            }
        )

    summary = {
        "run_id": run_id,
        "requested_as_of_date": as_of_date,
        "output_dir": str(out_dir),
        "provider_effective": str(getattr(provider, "provider_name", "unknown")),
        "cache_only": bool(cache_only),
        "ticker_count": len(rows),
        "ok_count": int(ok_count),
        "unknown_count": int(len(rows) - ok_count),
        "reason_counts": dict(sorted(reason_counts.items(), key=lambda kv: kv[0])),
        "rows": rows,
        "generated_at": utc_now_iso(),
    }
    summary_path = out_dir / "shares_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary

