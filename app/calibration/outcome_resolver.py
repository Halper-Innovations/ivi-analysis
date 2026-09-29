from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.calibration.perception_tracker import list_resolvable_perceptions
from app.calibration.schemas import PerceptionResolution, PerceptionTrackingRecord
from app.config import get_config
from app.db import utc_now_iso


def _json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _tracking_path(*, ticker: str, perception_id: str, cfg: Any) -> Path:
    return Path(cfg.outputs_dir) / "perception_tracking" / f"{str(ticker).strip().upper()}_{str(perception_id).strip()}.json"


# The CLI --prices contract (cli.py) supplies values already expressed as PERCENT
# (e.g. "TICKER=12.5" means +12.5%). Do NOT apply a magnitude heuristic: a real
# +0.8% move must stay +0.8%, not be rescaled to +80%. Values are clamped to a
# sane band to guard against malformed payloads.
_MAX_ABS_CHANGE_PCT = 100000.0


def _normalize_change_pct(value: Any) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    pct = float(value)
    if pct != pct:  # NaN guard
        return None
    if pct > _MAX_ABS_CHANGE_PCT:
        return _MAX_ABS_CHANGE_PCT
    if pct < -_MAX_ABS_CHANGE_PCT:
        return -_MAX_ABS_CHANGE_PCT
    return pct


def _infer_price_change_pct(record: PerceptionTrackingRecord, outcome_data: dict[str, Any], cfg: Any) -> float | None:
    direct = _normalize_change_pct(outcome_data.get("price_change_pct"))
    if direct is not None:
        return direct

    current_valuation = outcome_data.get("current_valuation")
    if isinstance(current_valuation, dict):
        for key in ["market_price", "price"]:
            value = current_valuation.get(key)
            if isinstance(value, (int, float)) and isinstance(record.registered_market_price, (int, float)) and record.registered_market_price:
                return ((float(value) - float(record.registered_market_price)) / float(record.registered_market_price)) * 100.0

    if not isinstance(record.registered_market_price, (int, float)) or float(record.registered_market_price) <= 0:
        return None

    # Point-in-time price MUST come from the date-aware market provider, which
    # selects the close on/before the target date from full daily history. The
    # valuation get_quote provider ignores as_of_date and returns the latest
    # price stamped with the historical date, contaminating grades with
    # look-ahead (future) information. If no true point-in-time price exists,
    # return None so the caller resolves INSUFFICIENT_DATA rather than
    # substituting a current quote.
    try:
        from app.market.price_provider import get_default_provider

        target_date = outcome_data.get("as_of_date") or utc_now_iso()[:10]
        snapshot = get_default_provider(cfg).get_price_asof(record.ticker, target_date)
        price = getattr(snapshot, "price", None)
        if isinstance(price, (int, float)) and float(price) > 0:
            return ((float(price) - float(record.registered_market_price)) / float(record.registered_market_price)) * 100.0
    except Exception:
        return None
    return None


def _resolution_for_direction(direction: str, price_change_pct: float | None) -> tuple[str, str]:
    if price_change_pct is None:
        return "INSUFFICIENT_DATA", "No usable price change data was available for resolution."
    direction_norm = str(direction or "").strip().upper()
    if direction_norm == "UNDERVALUED":
        if price_change_pct > 10.0:
            return "CONFIRMED", "Price appreciated more than 10% after registration."
        if price_change_pct < -15.0:
            return "DISCONFIRMED", "Price declined more than 15% after registration."
        return "INCONCLUSIVE", "Price move did not cross the confirmation or disconfirmation thresholds."
    if direction_norm == "OVERVALUED":
        if price_change_pct < -10.0:
            return "CONFIRMED", "Price declined more than 10% after registration."
        if price_change_pct > 15.0:
            return "DISCONFIRMED", "Price appreciated more than 15% after registration."
        return "INCONCLUSIVE", "Price move did not cross the confirmation or disconfirmation thresholds."
    return "INSUFFICIENT_DATA", "Unknown perception direction."


def resolve_perception(tracking_record: PerceptionTrackingRecord | dict[str, Any], *, outcome_data: dict[str, Any], cfg=None) -> PerceptionTrackingRecord:
    cfg = cfg or get_config()
    record = tracking_record if isinstance(tracking_record, PerceptionTrackingRecord) else PerceptionTrackingRecord.model_validate(tracking_record)
    price_change_pct = _infer_price_change_pct(record, outcome_data or {}, cfg)
    outcome_status, notes = _resolution_for_direction(record.direction, price_change_pct)
    resolved_at = utc_now_iso()
    resolution = PerceptionResolution(
        outcome_status=outcome_status,
        price_change_pct=price_change_pct,
        resolution_method="price_directional_v1",
        notes=notes,
        resolved_at=resolved_at,
    )
    updated = record.model_copy(
        update={
            "status": outcome_status,
            "resolved_at": resolved_at,
            "resolution": resolution,
        }
    )
    _json_write(
        _tracking_path(ticker=updated.ticker, perception_id=updated.perception_id, cfg=cfg),
        updated.model_dump(mode="json"),
    )
    return updated


def resolve_all_due(as_of_date: str, *, price_data: dict[str, float] | None = None, cfg=None) -> dict[str, Any]:
    cfg = cfg or get_config()
    price_lookup = {str(ticker).strip().upper(): value for ticker, value in (price_data or {}).items()}
    due = list_resolvable_perceptions(as_of_date, cfg=cfg)
    summary = {
        "total_due": len(due),
        "confirmed": 0,
        "disconfirmed": 0,
        "inconclusive": 0,
        "insufficient_data": 0,
    }
    for record in due:
        updated = resolve_perception(
            record,
            outcome_data={
                "price_change_pct": price_lookup.get(record.ticker),
                "as_of_date": as_of_date,
            },
            cfg=cfg,
        )
        status = str(updated.status or "").lower()
        if status == "confirmed":
            summary["confirmed"] += 1
        elif status == "disconfirmed":
            summary["disconfirmed"] += 1
        elif status == "inconclusive":
            summary["inconclusive"] += 1
        else:
            summary["insufficient_data"] += 1
    return summary
