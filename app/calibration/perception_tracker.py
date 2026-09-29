from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from app.config import get_config
from app.calibration.schemas import PerceptionTrackingRecord
from app.db import get_db, utc_now_iso
from app.synthesis.schemas import VariantPerception, VariantPerceptionReport
from app.valuation.facts import _dedupe_refs
from app.valuation.lineage import latest_decision_eligible_valuation_rows


_HORIZON_DAYS = {
    "SHORT": 365,
    "MEDIUM": 365 * 2,
    "LONG": 365 * 4,
}


def _safe_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _tracking_dir(cfg: Any) -> Path:
    return Path(cfg.outputs_dir) / "perception_tracking"


def _tracking_path(*, ticker: str, perception_id: str, cfg: Any) -> Path:
    return _tracking_dir(cfg) / f"{str(ticker).strip().upper()}_{str(perception_id).strip()}.json"


def _parse_iso_date(value: str) -> date:
    token = str(value or "").strip()
    if "T" in token:
        token = token.split("T", 1)[0]
    return date.fromisoformat(token)


def _expected_resolution_date(*, as_of_date: str, time_horizon: str) -> str:
    # Anchor the resolution window on the perception's point-in-time as_of_date,
    # NOT on the wall-clock registered_at timestamp. The baseline price is
    # captured at as_of_date, so the horizon must run from the same anchor;
    # registered_at is retained only as an audit timestamp.
    anchor_date = _parse_iso_date(as_of_date)
    days = _HORIZON_DAYS.get(str(time_horizon or "MEDIUM").strip().upper(), _HORIZON_DAYS["MEDIUM"])
    return (anchor_date + timedelta(days=days)).isoformat()


def _load_registered_market_price(*, ticker: str, as_of_date: str) -> float | None:
    try:
        with get_db() as conn:
            rows = latest_decision_eligible_valuation_rows(
                conn,
                ticker=ticker.upper(),
                as_of_date=as_of_date,
                exact_as_of_date=True,
            )
    except Exception:
        return None
    rows.sort(
        key=lambda row: (
            str(row["created_at"] or ""),
            int(row["id"]),
        ),
        reverse=True,
    )
    for row in rows:
        try:
            inputs = json.loads(row["inputs_json"] or "{}")
            outputs = json.loads(row["outputs_json"] or "{}")
        except Exception:
            continue
        for payload in [outputs, inputs]:
            if not isinstance(payload, dict):
                continue
            for key in ["market_price", "price"]:
                value = payload.get(key)
                if isinstance(value, (int, float)):
                    return float(value)
            nested = payload.get("outputs")
            if isinstance(nested, dict):
                value = nested.get("market_price") or nested.get("price")
                if isinstance(value, (int, float)):
                    return float(value)
    return None


def register_perception(perception: VariantPerception, cfg=None) -> PerceptionTrackingRecord:
    cfg = cfg or get_config()
    registered_at = utc_now_iso()
    path = _tracking_path(ticker=perception.ticker, perception_id=perception.perception_id, cfg=cfg)
    supporting_sources = sorted({signal.source for signal in perception.supporting_signals})
    pattern_ids = sorted(
        {
            signal.signal_type
            for signal in perception.supporting_signals
            if signal.source == "PATTERN"
        }
    )
    diff_signal_types = sorted(
        {
            signal.signal_type
            for signal in perception.supporting_signals
            if signal.source == "FILING_DIFF"
        }
    )
    record = PerceptionTrackingRecord(
        perception_id=perception.perception_id,
        ticker=perception.ticker.upper(),
        as_of_date=perception.as_of_date,
        thesis=perception.thesis,
        direction=perception.direction,
        confidence=perception.confidence,
        testable_prediction=perception.testable_prediction,
        falsification_trigger=perception.risk,
        time_horizon=perception.time_horizon,
        expected_resolution_date=_expected_resolution_date(
            as_of_date=perception.as_of_date,
            time_horizon=perception.time_horizon,
        ),
        supporting_signal_sources=supporting_sources,
        pattern_ids_involved=pattern_ids,
        diff_signal_types_involved=diff_signal_types,
        status="PENDING",
        registered_at=registered_at,
        resolved_at=None,
        resolution=None,
        derived_from=_dedupe_refs(list(perception.derived_from)),
        registered_market_price=_load_registered_market_price(
            ticker=perception.ticker, as_of_date=perception.as_of_date
        ),
    )
    _json_write(path, record.model_dump(mode="json"))
    return record


def register_perceptions_from_report(report: VariantPerceptionReport, cfg=None) -> int:
    count = 0
    for perception in report.perceptions:
        if not str(perception.testable_prediction or "").strip():
            continue
        register_perception(perception, cfg=cfg)
        count += 1
    return count


def _load_tracking_records(cfg=None) -> list[PerceptionTrackingRecord]:
    cfg = cfg or get_config()
    records: list[PerceptionTrackingRecord] = []
    for path in sorted(_tracking_dir(cfg).glob("*.json")):
        payload = _safe_json(path)
        if not payload:
            continue
        try:
            records.append(PerceptionTrackingRecord.model_validate(payload))
        except Exception:
            continue
    return records


def list_pending_perceptions(cfg=None) -> list[PerceptionTrackingRecord]:
    return [record for record in _load_tracking_records(cfg=cfg) if record.status == "PENDING"]


def list_resolvable_perceptions(as_of_date: str, cfg=None) -> list[PerceptionTrackingRecord]:
    cutoff = _parse_iso_date(as_of_date)
    return [
        record
        for record in list_pending_perceptions(cfg=cfg)
        if _parse_iso_date(record.expected_resolution_date) <= cutoff
    ]


def load_variant_reports_for_run(
    *, run_id: str, ticker: str | None = None, cfg=None
) -> list[VariantPerceptionReport]:
    cfg = cfg or get_config()
    run_id_norm = str(run_id or "").strip()
    ticker_norm = str(ticker or "").strip().upper()
    paths: list[Path] = []
    local_dir = Path(cfg.outputs_dir) / "universe" / run_id_norm / "variant_perceptions"
    if local_dir.exists():
        paths.extend(sorted(local_dir.glob("*.json")))
    global_dir = Path(cfg.outputs_dir) / "variant_perceptions"
    if global_dir.exists():
        paths.extend(sorted(global_dir.glob("*.json")))

    reports: list[VariantPerceptionReport] = []
    seen: set[tuple[str, str, str]] = set()
    for path in paths:
        payload = _safe_json(path)
        if not payload or str(payload.get("run_id") or "").strip() != run_id_norm:
            continue
        if ticker_norm and str(payload.get("ticker") or "").strip().upper() != ticker_norm:
            continue
        key = (
            str(payload.get("ticker") or "").strip().upper(),
            str(payload.get("run_id") or "").strip(),
            str(payload.get("as_of_date") or "").strip(),
        )
        if key in seen:
            continue
        seen.add(key)
        try:
            reports.append(VariantPerceptionReport.model_validate(payload))
        except Exception:
            continue
    return reports
