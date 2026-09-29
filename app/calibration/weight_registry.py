from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from app.calibration.schemas import CalibrationWeights, DiffSignalWeight, PatternWeight, SectorAccuracy
from app.config import get_config
from app.db import utc_now_iso
from app.universe.research_memory import load_research_memory


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


def _slug(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value or "").strip()).strip("_").lower()


def _weights_path(cfg: Any) -> Path:
    return Path(cfg.outputs_dir) / "calibration" / "calibration_weights.json"


def _tracking_dir(cfg: Any) -> Path:
    return Path(cfg.outputs_dir) / "perception_tracking"


def _load_sector_map(cfg: Any) -> dict[str, str]:
    mapping: dict[str, str] = {}
    path = Path(cfg.sector_taxonomy_path)
    if path.exists():
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    ticker = str(row.get("ticker") or "").strip().upper()
                    sector = _slug(row.get("sector"))
                    if ticker and sector:
                        mapping[ticker] = sector
        except Exception:
            mapping = {}
    memory = load_research_memory()
    tickers = memory.get("tickers") if isinstance(memory.get("tickers"), dict) else {}
    for ticker, entry in tickers.items():
        if str(ticker).upper() in mapping or not isinstance(entry, dict):
            continue
        latest = entry.get("latest") if isinstance(entry.get("latest"), dict) else {}
        token = latest.get("sector") or latest.get("sector_id") or latest.get("last_seen_universe_run_id") or entry.get("last_seen_run_id")
        if "__" in str(token or ""):
            mapping[str(ticker).upper()] = _slug(str(token).rsplit("__", 1)[-1])
        else:
            mapping[str(ticker).upper()] = _slug(token)
    return {ticker: sector for ticker, sector in mapping.items() if sector}


def _load_resolved_tracking_records(cfg: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(_tracking_dir(cfg).glob("*.json")):
        payload = _safe_json(path)
        status = str(payload.get("status") or "").upper()
        if status not in {"CONFIRMED", "DISCONFIRMED", "INCONCLUSIVE"}:
            continue
        rows.append(payload)
    return rows


def _accuracy(confirmed: int, disconfirmed: int) -> float | None:
    sample = confirmed + disconfirmed
    if sample <= 0:
        return None
    return float(confirmed) / float(sample)


def compute_calibration_weights(cfg=None) -> CalibrationWeights:
    cfg = cfg or get_config()
    resolved = _load_resolved_tracking_records(cfg)
    last_calibrated = utc_now_iso()
    sector_map = _load_sector_map(cfg)

    pattern_buckets: dict[str, dict[str, int]] = defaultdict(lambda: {"confirmed": 0, "disconfirmed": 0, "resolved": 0})
    diff_buckets: dict[str, dict[str, int]] = defaultdict(lambda: {"confirmed": 0, "disconfirmed": 0, "resolved": 0})
    sector_buckets: dict[str, dict[str, int]] = defaultdict(lambda: {"confirmed": 0, "disconfirmed": 0, "resolved": 0})
    total_confirmed = 0
    total_disconfirmed = 0
    total_inconclusive = 0

    for row in resolved:
        status = str(row.get("status") or "").upper()
        ticker = str(row.get("ticker") or "").strip().upper()
        is_confirmed = status == "CONFIRMED"
        is_disconfirmed = status == "DISCONFIRMED"
        is_inconclusive = status == "INCONCLUSIVE"
        if is_confirmed:
            total_confirmed += 1
        elif is_disconfirmed:
            total_disconfirmed += 1
        elif is_inconclusive:
            total_inconclusive += 1

        for pattern_id in [str(token) for token in (row.get("pattern_ids_involved") or []) if str(token).strip()]:
            bucket = pattern_buckets[pattern_id]
            bucket["resolved"] += 1
            bucket["confirmed"] += 1 if is_confirmed else 0
            bucket["disconfirmed"] += 1 if is_disconfirmed else 0

        for signal_type in [str(token) for token in (row.get("diff_signal_types_involved") or []) if str(token).strip()]:
            bucket = diff_buckets[signal_type]
            bucket["resolved"] += 1
            bucket["confirmed"] += 1 if is_confirmed else 0
            bucket["disconfirmed"] += 1 if is_disconfirmed else 0

        sector_id = sector_map.get(ticker, "")
        if sector_id:
            bucket = sector_buckets[sector_id]
            bucket["resolved"] += 1
            bucket["confirmed"] += 1 if is_confirmed else 0
            bucket["disconfirmed"] += 1 if is_disconfirmed else 0

    # sample_size is the DECISIVE denominator (confirmed + disconfirmed) — the
    # same denominator hit_rate/accuracy is computed over. Counting INCONCLUSIVE
    # resolutions here would let a 1-confirmed/99-inconclusive bucket advertise
    # hit_rate=1.0 with sample_size=100, hiding a thin sample from downstream
    # confidence gates.
    weights = CalibrationWeights(
        pattern_weights={
            pattern_id: PatternWeight(
                pattern_id=pattern_id,
                hit_rate=_accuracy(bucket["confirmed"], bucket["disconfirmed"]) or 0.5,
                sample_size=int(bucket["confirmed"] + bucket["disconfirmed"]),
                last_calibrated=last_calibrated,
            )
            for pattern_id, bucket in sorted(pattern_buckets.items())
        },
        diff_signal_weights={
            signal_type: DiffSignalWeight(
                signal_type=signal_type,
                predictive_rate=_accuracy(bucket["confirmed"], bucket["disconfirmed"]) or 0.5,
                sample_size=int(bucket["confirmed"] + bucket["disconfirmed"]),
                last_calibrated=last_calibrated,
            )
            for signal_type, bucket in sorted(diff_buckets.items())
        },
        sector_accuracy={
            sector_id: SectorAccuracy(
                sector_id=sector_id,
                accuracy_rate=_accuracy(bucket["confirmed"], bucket["disconfirmed"]) or 0.5,
                sample_size=int(bucket["confirmed"] + bucket["disconfirmed"]),
                last_calibrated=last_calibrated,
            )
            for sector_id, bucket in sorted(sector_buckets.items())
        },
        overall_accuracy=_accuracy(total_confirmed, total_disconfirmed),
        total_resolved=len(resolved),
        total_confirmed=total_confirmed,
        total_disconfirmed=total_disconfirmed,
        total_inconclusive=total_inconclusive,
        last_calibrated=last_calibrated,
    )
    return weights


def write_calibration_weights(weights: CalibrationWeights | dict[str, Any], cfg=None) -> Path:
    cfg = cfg or get_config()
    model = weights if isinstance(weights, CalibrationWeights) else CalibrationWeights.model_validate(weights)
    _json_write(_weights_path(cfg), model.model_dump(mode="json"))
    return _weights_path(cfg)


def load_calibration_weights(cfg=None) -> dict[str, Any]:
    cfg = cfg or get_config()
    payload = _safe_json(_weights_path(cfg))
    if not payload:
        return CalibrationWeights(
            pattern_weights={},
            diff_signal_weights={},
            sector_accuracy={},
            overall_accuracy=None,
            total_resolved=0,
            total_confirmed=0,
            total_disconfirmed=0,
            total_inconclusive=0,
            last_calibrated=utc_now_iso(),
        ).model_dump(mode="json")
    try:
        return CalibrationWeights.model_validate(payload).model_dump(mode="json")
    except Exception:
        return CalibrationWeights(
            pattern_weights={},
            diff_signal_weights={},
            sector_accuracy={},
            overall_accuracy=None,
            total_resolved=0,
            total_confirmed=0,
            total_disconfirmed=0,
            total_inconclusive=0,
            last_calibrated=utc_now_iso(),
        ).model_dump(mode="json")
