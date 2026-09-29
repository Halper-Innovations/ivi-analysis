from __future__ import annotations

import csv
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


_REGULATORY_TYPES = {
    "edgar",
    "sec_exhibit",
    "8-k",
    "8-k/a",
    "10-k",
    "10-k/a",
    "10-q",
    "10-q/a",
    "20-f",
    "20-f/a",
    "40-f",
    "40-f/a",
}
_COMPANY_CONTROLLED_TYPES = {"ir_press", "company_news"}
_TRANSCRIPT_TYPES = {"transcript"}
_SECONDARY_TYPES = {"external_news"}
_REFERENCE_TYPES = {"wikipedia"}


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = datetime.fromisoformat(f"{str(value)[:10]}T00:00:00+00:00")
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _freshness_days(*, published_at: str | None, as_of_date: str | None) -> int | None:
    published = _parse_dt(published_at)
    if published is None or not as_of_date:
        return None
    try:
        anchor = date.fromisoformat(str(as_of_date)[:10])
    except ValueError:
        return None
    return (anchor - published.date()).days


def _freshness_bucket(days: int | None) -> tuple[str, str]:
    if days is None:
        return "undated", "FRESHNESS_UNDATED"
    if days < 0:
        return "future_dated", "FRESHNESS_FUTURE_DATED"
    if days == 0:
        return "same_day", "FRESHNESS_SAME_DAY"
    if days <= 7:
        return "recent_7d", "FRESHNESS_RECENT_7D"
    if days <= 30:
        return "current_30d", "FRESHNESS_CURRENT_30D"
    if days <= 90:
        return "current_90d", "FRESHNESS_CURRENT_90D"
    return "stale_over_90d", "FRESHNESS_STALE_OVER_90D"


def _normalize_domain(value: object) -> str | None:
    raw = str(value or "").lower().strip().rstrip(".")
    if not raw:
        return None
    parsed = urlparse(raw if "://" in raw else f"//{raw}")
    host = (parsed.hostname or raw).lower().strip().rstrip(".")
    return host or None


def _domain(source_url: str | None) -> str | None:
    return _normalize_domain(source_url)


def _family(source_type: str | None) -> tuple[str, str, str, float, list[str]]:
    normalized = str(source_type or "").strip().lower()
    if normalized in _REGULATORY_TYPES:
        return (
            "regulatory_filing",
            "primary",
            "regulatory",
            1.0,
            ["SOURCE_PRIMARY_REGULATORY"],
        )
    if normalized in _COMPANY_CONTROLLED_TYPES:
        return (
            "company_controlled",
            "primary_company_controlled",
            "company_controlled",
            0.82,
            ["SOURCE_COMPANY_CONTROLLED", "SOURCE_ISSUER_BIAS_POSSIBLE"],
        )
    if normalized in _TRANSCRIPT_TYPES:
        return (
            "management_transcript",
            "primary_company_controlled",
            "company_controlled",
            0.86,
            ["SOURCE_MANAGEMENT_TRANSCRIPT", "SOURCE_ISSUER_BIAS_POSSIBLE"],
        )
    if normalized in _SECONDARY_TYPES:
        return (
            "external_news",
            "secondary",
            "independent_or_third_party",
            0.68,
            ["SOURCE_EXTERNAL_SECONDARY", "SECONDARY_SOURCE_CALIBRATION_PENDING"],
        )
    if normalized in _REFERENCE_TYPES:
        return (
            "reference",
            "tertiary_reference",
            "mixed",
            0.45,
            ["SOURCE_REFERENCE", "SOURCE_NOT_PRIMARY"],
        )
    return (
        "unknown",
        "unknown",
        "unknown",
        0.35,
        ["SOURCE_UNCLASSIFIED"],
    )


def _float_from_record(record: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = record.get(key)
        if value is None or str(value).strip() == "":
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        return max(0.0, min(1.0, parsed))
    return None


def _int_from_record(record: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = record.get(key)
        if value is None or str(value).strip() == "":
            continue
        try:
            parsed = int(float(value))
        except (TypeError, ValueError):
            continue
        return max(0, parsed)
    return None


def _date_from_record(record: dict[str, Any]) -> str | None:
    for key in ("as_of_date", "source_reputation_as_of_date", "assessed_at", "observed_at", "updated_at"):
        value = record.get(key)
        if not value:
            continue
        text = str(value).strip()
        try:
            return date.fromisoformat(text[:10]).isoformat()
        except ValueError:
            continue
    return None


def _record_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _reason_codes_from_record(record: dict[str, Any]) -> list[str]:
    raw = record.get("reason_codes") or record.get("reason_code") or ""
    if isinstance(raw, (list, tuple, set)):
        values = raw
    else:
        values = str(raw).replace(";", ",").replace("|", ",").split(",")
    out: list[str] = []
    for token in values:
        code = str(token).strip().upper()
        if code and code not in out:
            out.append(code)
    return out


def _normalize_reputation_record(record: dict[str, Any]) -> dict[str, Any] | None:
    domain = _normalize_domain(record.get("domain") or record.get("source_domain"))
    score = _float_from_record(
        record,
        (
            "source_reputation_score",
            "reputation_score",
            "reliability_score",
            "accuracy_score",
        ),
    )
    if domain is None or score is None:
        return None
    sample_size = _int_from_record(
        record,
        ("sample_size", "source_reputation_sample_size", "observation_count", "observations"),
    )
    calibration_status = str(record.get("calibration_status") or "").strip()
    return {
        "domain": domain,
        "source_reputation_score": score,
        "source_reputation_as_of_date": _date_from_record(record),
        "source_reputation_sample_size": sample_size,
        "calibration_status": calibration_status or None,
        "reason_codes": _reason_codes_from_record(record),
    }


def load_source_reputation_history(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    history_path = Path(path)
    if not history_path.exists():
        return []
    try:
        with history_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            records = [_normalize_reputation_record(dict(row)) for row in reader]
    except OSError:
        return []
    return [record for record in records if record is not None]


def _domain_match_specificity(source_domain: str, record_domain: str) -> int | None:
    if record_domain.startswith("*."):
        suffix = record_domain[2:]
        if source_domain == suffix or source_domain.endswith(f".{suffix}"):
            return len(suffix)
        return None
    if source_domain == record_domain:
        return len(record_domain) + 1000
    if source_domain.endswith(f".{record_domain}"):
        return len(record_domain)
    return None


def _select_reputation_record(
    *,
    source_domain: str | None,
    as_of_date: str | None,
    history: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not source_domain:
        return None
    anchor = _record_date(as_of_date)
    best_record: dict[str, Any] | None = None
    best_key: tuple[date, int, int] | None = None
    for record in history:
        record_domain = str(record.get("domain") or "").strip().lower()
        specificity = _domain_match_specificity(source_domain, record_domain)
        if specificity is None:
            continue
        reputation_date = _record_date(record.get("source_reputation_as_of_date"))
        if anchor is not None and reputation_date is not None and reputation_date > anchor:
            continue
        key = (
            reputation_date or date.min,
            specificity,
            int(record.get("source_reputation_sample_size") or 0),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_record = record
    return best_record


def _reputation_reason(score: float) -> str:
    if score >= 0.8:
        return "SOURCE_REPUTATION_HIGH"
    if score >= 0.55:
        return "SOURCE_REPUTATION_MEDIUM"
    return "SOURCE_REPUTATION_LOW"


def _reputation_calibration_status(record: dict[str, Any]) -> str:
    explicit = str(record.get("calibration_status") or "").strip()
    if explicit:
        return explicit
    sample_size = record.get("source_reputation_sample_size")
    if isinstance(sample_size, int) and sample_size >= 10:
        return "domain_reputation_calibrated"
    return "domain_reputation_limited"


def classify_source_quality(
    *,
    source_type: str | None,
    source_url: str | None = None,
    published_at: str | None = None,
    as_of_date: str | None = None,
    source_reputation_path: str | Path | None = None,
    source_reputation_history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    family, origin, independence, base_score, reason_codes = _family(source_type)
    days = _freshness_days(published_at=published_at, as_of_date=as_of_date)
    freshness_bucket, freshness_reason = _freshness_bucket(days)
    source_domain = _domain(source_url)
    score = base_score
    if freshness_bucket == "same_day":
        score += 0.04
    elif freshness_bucket == "recent_7d":
        score += 0.03
    elif freshness_bucket == "current_30d":
        score += 0.02
    elif freshness_bucket == "current_90d":
        score += 0.0
    elif freshness_bucket == "stale_over_90d":
        score -= 0.12
    elif freshness_bucket == "future_dated":
        score -= 0.25
    elif freshness_bucket == "undated":
        score -= 0.05

    if family == "external_news":
        calibration_status = "heuristic_unvalidated"
    elif family == "unknown":
        calibration_status = "unclassified"
    else:
        calibration_status = "deterministic_heuristic"

    reputation_registry_present = source_reputation_history is not None
    if source_reputation_history is None:
        reputation_registry_present = source_reputation_path is not None and Path(source_reputation_path).exists()
        reputation_history = load_source_reputation_history(source_reputation_path)
    else:
        reputation_history = [
            record
            for record in (_normalize_reputation_record(item) for item in source_reputation_history)
            if record is not None
        ]
    reputation_record = _select_reputation_record(
        source_domain=source_domain,
        as_of_date=as_of_date,
        history=reputation_history,
    )
    quality_reason_codes = list(dict.fromkeys([*reason_codes, freshness_reason]))

    reputation_fields: dict[str, Any] = {}
    if reputation_record is not None:
        reputation_score = float(reputation_record["source_reputation_score"])
        score += (reputation_score - base_score) * 0.45
        calibration_status = _reputation_calibration_status(reputation_record)
        quality_reason_codes = [
            code
            for code in quality_reason_codes
            if code != "SECONDARY_SOURCE_CALIBRATION_PENDING"
        ]
        quality_reason_codes.extend(
            [
                "SOURCE_REPUTATION_HISTORY",
                _reputation_reason(reputation_score),
                *list(reputation_record.get("reason_codes") or []),
            ]
        )
        reputation_fields = {
            "source_reputation_status": "known",
            "source_reputation_score": round(reputation_score, 2),
            "source_reputation_as_of_date": reputation_record.get("source_reputation_as_of_date"),
            "source_reputation_sample_size": reputation_record.get("source_reputation_sample_size"),
        }
    elif reputation_registry_present and family in {"external_news", "reference", "unknown"}:
        quality_reason_codes.append("SOURCE_REPUTATION_MISSING")
        reputation_fields = {"source_reputation_status": "missing"}

    return {
        "source_family": family,
        "source_origin": origin,
        "source_independence": independence,
        "source_domain": source_domain,
        "freshness_days": days,
        "freshness_bucket": freshness_bucket,
        "source_quality_score": round(max(0.0, min(1.0, score)), 2),
        "calibration_status": calibration_status,
        "reason_codes": list(dict.fromkeys(quality_reason_codes)),
        **reputation_fields,
    }


def source_quality_label(source_quality: dict[str, Any] | None) -> str:
    if not source_quality:
        return ""
    family = str(source_quality.get("source_family") or "unknown")
    freshness = str(source_quality.get("freshness_bucket") or "unknown")
    calibration = str(source_quality.get("calibration_status") or "unknown")
    score = source_quality.get("source_quality_score")
    score_text = f"{float(score):.2f}" if isinstance(score, (int, float)) and not isinstance(score, bool) else "unknown"
    label = f"{family}; freshness={freshness}; score={score_text}; calibration={calibration}"
    reputation_status = source_quality.get("source_reputation_status")
    if reputation_status:
        reputation_score = source_quality.get("source_reputation_score")
        if isinstance(reputation_score, (int, float)) and not isinstance(reputation_score, bool):
            label = f"{label}; reputation={reputation_status}:{float(reputation_score):.2f}"
        else:
            label = f"{label}; reputation={reputation_status}"
    return label


__all__ = ["classify_source_quality", "load_source_reputation_history", "source_quality_label"]
