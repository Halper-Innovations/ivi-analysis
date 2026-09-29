"""Insurance operating metric extraction from cached primary filings.

V1 is intentionally narrow and deterministic. It extracts P&C-oriented
operating signals that the generic scorecard cannot supply: underwriting
ratios, reserve-development language, catastrophe exposure, and reinsurance
structure. It also extracts mortgage-insurance operating signals such as
PMIERs sufficiency, IIF/RIF, NIW, and default-rate data. Missing values remain
explicit rather than becoming zeroes.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence

from app.config import AppConfig
from app.insurance.routing import ISSUER_INSURANCE_UNDERWRITER
from app.insurance.sources import latest_cached_filing_text
from app.util.html_strip import strip_html

UNKNOWN = "UNKNOWN"

_PC_SUBTYPES = {"pc_insurer", "reinsurer"}
_MORTGAGE_SUBTYPES = {"title_mortgage_specialty"}
_MAX_PC_FILING_CHARS = 3_000_000
_MAX_MORTGAGE_FILING_CHARS = 3_000_000

_REINSURANCE_PATTERNS: dict[str, str] = {
    "quota_share": r"\bquota\s+share\b",
    "excess_of_loss": r"\bexcess[-\s]of[-\s]loss\b",
    "facultative": r"\bfacultative\b",
    "retrocession": r"\bretrocession|retro(?:ceded|cessional)\b",
    "catastrophe_treaty": r"\bcatastrophe\s+(?:treaty|reinsurance|program)\b",
}

_CATASTROPHE_PATTERNS: dict[str, str] = {
    "catastrophe": r"\bcatastroph(?:e|ic|es)\b",
    "hurricane": r"\bhurricanes?\b",
    "wildfire": r"\bwildfires?\b",
    "earthquake": r"\bearthquakes?\b",
    "convective_storm": r"\bconvective\s+storms?\b",
    "flood": r"\bfloods?\b",
}

_MORTGAGE_REINSURANCE_PATTERNS: dict[str, str] = {
    "quota_share": r"\bquota\s+share\b|\bqsr\b",
    "excess_of_loss": r"\bexcess[-\s]of[-\s]loss\b|\bxol\b",
    "insurance_linked_notes": r"\binsurance[-\s]linked\s+notes?\b|\biln\b",
}


def _routing_dict(routing: Any) -> dict[str, Any]:
    if hasattr(routing, "to_dict"):
        return routing.to_dict()
    return routing if isinstance(routing, dict) else {}


def _is_pc_applicable(routing: dict[str, Any]) -> bool:
    return (
        routing.get("issuer_type") == ISSUER_INSURANCE_UNDERWRITER
        and str(routing.get("insurance_subtype") or "") in _PC_SUBTYPES
    )


def _is_mortgage_applicable(routing: dict[str, Any]) -> bool:
    return (
        routing.get("issuer_type") == ISSUER_INSURANCE_UNDERWRITER
        and str(routing.get("insurance_subtype") or "") in _MORTGAGE_SUBTYPES
    )


def _excerpt(text: str, start: int, end: int, *, context: int = 180) -> str:
    left = max(0, start - context)
    right = min(len(text), end + context)
    return re.sub(r"\s+", " ", text[left:right]).strip()


def _normalize_ratio(raw_value: str) -> float | None:
    try:
        value = float(raw_value.replace(",", ""))
    except ValueError:
        return None
    if value <= 0 or value > 300:
        return None
    return round(value / 100.0 if value > 2 else value, 6)


def _normalize_percent(raw_value: str) -> float | None:
    try:
        value = float(raw_value.replace(",", ""))
    except ValueError:
        return None
    if value < 0 or value > 300:
        return None
    return round(value / 100.0, 6)


def _normalize_money_to_billions(raw_value: str, unit: str | None = None) -> float | None:
    try:
        value = float(raw_value.replace(",", ""))
    except ValueError:
        return None
    if value <= 0:
        return None
    unit_token = re.sub(r"[\s_-]+", " ", str(unit or "").strip().lower())
    if unit_token in {"trillion", "trillions", "usd trillion", "usd trillions"}:
        return round(value * 1000.0, 3)
    if unit_token in {"billion", "billions", "usd billion", "usd billions"}:
        return round(value, 3)
    if unit_token in {"million", "millions", "usd million", "usd millions"}:
        return round(value / 1000.0, 3)
    if unit_token in {"thousand", "thousands", "usd thousand", "usd thousands"}:
        return round(value / 1_000_000.0, 9)
    if unit_token in {"usd", "dollar", "dollars"}:
        return round(value / 1_000_000_000.0, 12)
    return None


def _normalize_count(raw_value: str) -> int | None:
    try:
        value = int(raw_value.replace(",", ""))
    except ValueError:
        return None
    return value if value >= 0 else None


def _ratio_context_is_noise(label: str, value: float, excerpt: str) -> bool:
    lower = excerpt.lower()
    if value >= 0.995 and any(
        token in lower
        for token in (
            "under 100%",
            "over 100%",
            "below 100%",
            "exceeding 100%",
            "greater than 100%",
            "indicates an underwriting profit",
            "indicates an underwriting loss",
        )
    ):
        return True
    if "loss" in label.lower() and any(
        token in lower
        for token in (
            "attritional loss ratio",
            "catastrophe loss ratio",
            "underlying loss and loss adjustment expense ratio",
            "gross underlying",
            "initial expected loss ratio",
            "estimated loss ratio",
        )
    ):
        return True
    if (
        "expense" in label.lower()
        and "loss" not in label.lower()
        and any(
            token in lower
            for token in (
                "other underwriting expense ratio",
                "acquisition cost ratio",
                "combined ratio is the sum",
            )
        )
    ):
        return True
    return False


def _extract_ratio(text: str, labels: str | list[str]) -> dict[str, Any]:
    label_list = [labels] if isinstance(labels, str) else labels
    for label in label_list:
        escaped = re.escape(label)
        label_pattern = rf"(?<![A-Za-z]){escaped}s?(?![A-Za-z])"
        patterns = [
            rf"{label_pattern}[^%]{{0,160}}?(\d{{1,3}}(?:\.\d+)?)\s*%",
            rf"(\d{{1,3}}(?:\.\d+)?)\s*%[^.]{{0,120}}{label_pattern}",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                value = _normalize_ratio(match.group(1))
                if value is None:
                    continue
                local_context = text[max(0, match.start() - 20) : min(len(text), match.end() + 100)]
                if _ratio_context_is_noise(label, value, local_context):
                    continue
                return {
                    "value": value,
                    "source_label": label,
                    "source_excerpt": _excerpt(text, match.start(), match.end()),
                }
    return {"value": None, "source_label": None, "source_excerpt": None}


def _derived_expense_ratio(
    combined: dict[str, Any], loss: dict[str, Any], expense: dict[str, Any]
) -> dict[str, Any]:
    combined_value = combined.get("value")
    loss_value = loss.get("value")
    expense_value = expense.get("value")
    if not isinstance(combined_value, (int, float)) or not isinstance(loss_value, (int, float)):
        return expense
    derived = round(float(combined_value) - float(loss_value), 6)
    if derived <= 0 or derived > 1:
        return expense
    if expense_value is None or abs(float(expense_value) - derived) > 0.05:
        return {
            "value": derived,
            "source_label": "derived_combined_minus_loss",
            "source_excerpt": "Derived as combined ratio minus loss ratio.",
        }
    return expense


def _reserve_development(text: str) -> dict[str, Any]:
    lower = text.lower()
    favorable_patterns = [
        "favorable prior year reserve development",
        "favorable prior year development",
        "favorable reserve development",
        "favorable development of prior year reserves",
        "redundant reserves",
    ]
    adverse_patterns = [
        "adverse prior year reserve development",
        "adverse reserve development",
        "unfavorable reserve development",
        "reserve strengthening",
        "deficient reserves",
    ]
    favorable = [phrase for phrase in favorable_patterns if phrase in lower]
    adverse = [phrase for phrase in adverse_patterns if phrase in lower]
    if favorable and adverse:
        status = "MIXED"
    elif favorable:
        status = "FAVORABLE"
    elif adverse:
        status = "ADVERSE"
    else:
        status = UNKNOWN
    return {
        "status": status,
        "favorable_mentions": favorable,
        "adverse_mentions": adverse,
    }


def _matched_terms(text: str, patterns: dict[str, str]) -> list[str]:
    return [
        label
        for label, pattern in patterns.items()
        if re.search(pattern, text, flags=re.IGNORECASE)
    ]


def _combined_ratio_assessment(value: float | None) -> str:
    if value is None:
        return UNKNOWN
    if value < 0.95:
        return "STRONG_UNDERWRITING_PROFIT"
    if value < 1.0:
        return "UNDERWRITING_PROFIT"
    if value < 1.05:
        return "UNDERWRITING_LOSS_WATCH"
    return "UNDERWRITING_LOSS"


def _extract_pmier_sufficiency(text: str) -> dict[str, Any]:
    patterns = [
        r"PMIERs\s+available\s+assets\s+exceeded[^.]{0,180}?\bby\s+(\d{1,3}(?:\.\d+)?)\s*%",
        r"PMIERs\s+sufficiency\s+ratio[^.]{0,120}?(\d{1,3}(?:\.\d+)?)\s*%",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        excess_ratio = _normalize_percent(match.group(1))
        if excess_ratio is None:
            continue
        return {
            "pmier_excess_ratio": excess_ratio,
            "pmier_available_to_required_ratio": round(1.0 + excess_ratio, 6),
            "source_excerpt": _excerpt(text, match.start(), match.end()),
        }
    return {
        "pmier_excess_ratio": None,
        "pmier_available_to_required_ratio": None,
        "source_excerpt": None,
    }


def _extract_mortgage_summary_values(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "primary_iif_billion": None,
        "primary_rif_billion": None,
        "new_insurance_written_billion": None,
        "customer_count": None,
        "source_excerpt": None,
    }
    summary_match = re.search(
        r"had\s+\$([\d,.]+)\s+(billion|million)\s+of\s+primary\s+IIF\s+and\s+\$([\d,.]+)\s+(billion|million)\s+of\s+primary\s+RIF[\s\S]{0,220}?generated\s+NIW\s+of\s+\$([\d,.]+)\s+(billion|million)",
        text,
        flags=re.IGNORECASE,
    )
    if summary_match:
        result["primary_iif_billion"] = _normalize_money_to_billions(
            summary_match.group(1), summary_match.group(2)
        )
        result["primary_rif_billion"] = _normalize_money_to_billions(
            summary_match.group(3), summary_match.group(4)
        )
        result["new_insurance_written_billion"] = _normalize_money_to_billions(
            summary_match.group(5), summary_match.group(6)
        )
        result["source_excerpt"] = _excerpt(text, summary_match.start(), summary_match.end())

    customer_match = re.search(
        r"issued\s+master\s+policies\s+with\s+([\d,]+)\s+customers", text, flags=re.IGNORECASE
    )
    if customer_match:
        result["customer_count"] = _normalize_count(customer_match.group(1))
        result["customer_source_excerpt"] = _excerpt(
            text, customer_match.start(), customer_match.end()
        )
    return result


def _extract_mortgage_portfolio_table_values(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "new_insurance_written_billion": None,
        "primary_iif_billion": None,
        "primary_rif_billion": None,
        "policies_in_force": None,
        "loans_in_default": None,
        "default_rate": None,
        "rif_on_defaulted_loans_billion": None,
        "annual_persistency": None,
        "quarterly_runoff": None,
        "source_excerpt": None,
    }
    window_match = re.search(
        r"New\s+insurance\s+written[\s\S]{0,2600}?Quarterly\s+run[-\s]off[\s\S]{0,160}",
        text,
        flags=re.IGNORECASE,
    )
    if not window_match:
        return result
    window = window_match.group(0)
    result["source_excerpt"] = _excerpt(text, window_match.start(), window_match.end(), context=80)

    patterns = {
        "new_insurance_written_billion": r"New\s+insurance\s+written\s*\$?([\d,]+)",
        "primary_iif_billion": r"Insurance-in-force\s*\([^)]*\)\s*\$?([\d,]+)",
        "primary_rif_billion": r"Risk-in-force\s*\([^)]*\)\s*\$?([\d,]+)",
        "policies_in_force": r"Policies\s+in\s+force\s*\(count\)\s*\([^)]*\)\s*([\d,]+)",
        "loans_in_default": r"Loans\s+in\s+default\s*\(count\)\s*\([^)]*\)\s*([\d,]+)",
        "rif_on_defaulted_loans_billion": r"Risk-in-force\s+on\s+defaulted\s+loans\s*\([^)]*\)\s*\$?([\d,]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, window, flags=re.IGNORECASE)
        if not match:
            continue
        if key in {"policies_in_force", "loans_in_default"}:
            result[key] = _normalize_count(match.group(1))
        else:
            result[key] = _normalize_money_to_billions(match.group(1), "million")

    for key, pattern in {
        "default_rate": r"Default\s+rate\s*\([^)]*\)\s*(\d{1,2}(?:\.\d+)?)\s*%",
        "annual_persistency": r"Annual\s+persistency\s*\([^)]*\)\s*(\d{1,3}(?:\.\d+)?)\s*%",
        "quarterly_runoff": r"Quarterly\s+run[-\s]off\s*\([^)]*\)\s*(\d{1,3}(?:\.\d+)?)\s*%",
    }.items():
        match = re.search(pattern, window, flags=re.IGNORECASE)
        if match:
            result[key] = _normalize_percent(match.group(1))
    return result


def _extract_mortgage_default_narrative(text: str) -> dict[str, Any]:
    match = re.search(
        r"had\s+([\d,]+)\s+loans\s+in\s+default[^.]{0,160}?represented\s+a\s+(\d{1,2}(?:\.\d+)?)\s*%\s+default\s+rate\s+against\s+([\d,]+)\s+total\s+policies\s+in-force",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return {
            "loans_in_default": None,
            "default_rate": None,
            "policies_in_force": None,
            "source_excerpt": None,
        }
    return {
        "loans_in_default": _normalize_count(match.group(1)),
        "default_rate": _normalize_percent(match.group(2)),
        "policies_in_force": _normalize_count(match.group(3)),
        "source_excerpt": _excerpt(text, match.start(), match.end()),
    }


def _extract_mortgage_claims_paid(text: str) -> dict[str, Any]:
    match = re.search(
        r"paid\s+([\d,]+)\s+claims\s+totaling\s+\$([\d,.]+)\s+(million|billion)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return {"claims_paid_count": None, "claims_paid_million": None, "source_excerpt": None}
    try:
        amount = float(match.group(2).replace(",", ""))
    except ValueError:
        amount = 0.0
    claims_paid_million = (
        round(amount * 1000.0, 3) if "billion" in match.group(3).lower() else round(amount, 3)
    )
    return {
        "claims_paid_count": _normalize_count(match.group(1)),
        "claims_paid_million": claims_paid_million,
        "source_excerpt": _excerpt(text, match.start(), match.end()),
    }


def _mortgage_default_assessment(
    default_rate: float | None, pmier_excess_ratio: float | None
) -> str:
    if default_rate is None and pmier_excess_ratio is None:
        return UNKNOWN
    if default_rate is not None and default_rate >= 0.03:
        return "MORTGAGE_CREDIT_STRESS"
    if pmier_excess_ratio is not None and pmier_excess_ratio < 0.25:
        return "PMIER_CAPITAL_THIN"
    if (
        default_rate is not None
        and default_rate < 0.015
        and (pmier_excess_ratio is None or pmier_excess_ratio >= 0.5)
    ):
        return "STRONG_CAPITAL_AND_CREDIT_PROFILE"
    return "WATCH"


def build_pc_operating_metrics(
    ticker: str,
    *,
    as_of_date: str | None = None,
    routing: Any = None,
    pipeline_version: str = "v1",
    issuer_cik: str | None = None,
    aliases: Sequence[str] = (),
    db_path: str | Path | None = None,
    cfg: AppConfig | None = None,
) -> dict[str, Any]:
    """Build a deterministic P&C operating evidence packet."""
    routing_dict = _routing_dict(routing)
    if not _is_pc_applicable(routing_dict):
        return {
            "status": "NOT_APPLICABLE",
            "ticker": ticker.upper(),
            "reason_codes": ["NOT_PC_INSURANCE_SUBTYPE"],
        }

    is_v2 = str(pipeline_version or "v1").strip().lower() == "v2"
    if is_v2:
        raw_text, filing_meta = latest_cached_filing_text(
            ticker,
            as_of_date=as_of_date,
            max_chars=_MAX_PC_FILING_CHARS,
            issuer_cik=issuer_cik,
            aliases=aliases,
            db_path=db_path,
            cfg=cfg,
        )
    else:
        raw_text, filing_meta = latest_cached_filing_text(
            ticker,
            as_of_date=as_of_date,
            max_chars=_MAX_PC_FILING_CHARS,
        )
    if not raw_text:
        return {
            "status": "LIMITED",
            "ticker": ticker.upper(),
            "reason_codes": ["NO_CACHED_ANNUAL_FILING"],
            "missing_components": [
                "COMBINED_RATIO",
                "LOSS_RATIO",
                "EXPENSE_RATIO",
                "RESERVE_DEVELOPMENT",
            ],
            "source_references": {},
        }

    text = strip_html(raw_text)
    combined = _extract_ratio(text, "combined ratio")
    loss = _extract_ratio(
        text,
        [
            "loss ratio",
            "loss and loss adjustment expense ratio",
            "loss and LAE ratio",
            "losses and loss adjustment expenses ratio",
        ],
    )
    expense = _derived_expense_ratio(
        combined,
        loss,
        _extract_ratio(text, ["expense ratio", "underwriting expense ratio"]),
    )
    reserve = _reserve_development(text)
    reinsurance_structures = _matched_terms(text, _REINSURANCE_PATTERNS)
    catastrophe_terms = _matched_terms(text, _CATASTROPHE_PATTERNS)

    missing_components: list[str] = []
    if combined["value"] is None:
        missing_components.append("COMBINED_RATIO")
    if loss["value"] is None:
        missing_components.append("LOSS_RATIO")
    if expense["value"] is None:
        missing_components.append("EXPENSE_RATIO")
    if reserve["status"] == UNKNOWN:
        missing_components.append("RESERVE_DEVELOPMENT")

    found_any = (
        any(item is not None for item in (combined["value"], loss["value"], expense["value"]))
        or reserve["status"] != UNKNOWN
        or bool(reinsurance_structures)
        or bool(catastrophe_terms)
    )

    confidence = "LOW"
    if combined["value"] is not None and (
        loss["value"] is not None or expense["value"] is not None
    ):
        confidence = "HIGH"
    elif found_any:
        confidence = "MEDIUM"

    return {
        "status": "OK" if found_any else "LIMITED",
        "ticker": ticker.upper(),
        "as_of_date": as_of_date,
        "insurance_subtype": routing_dict.get("insurance_subtype"),
        "combined_ratio": combined["value"],
        "loss_ratio": loss["value"],
        "expense_ratio": expense["value"],
        "combined_ratio_assessment": _combined_ratio_assessment(combined["value"]),
        "reserve_development": reserve,
        "reinsurance_program": {
            "mentioned": bool(reinsurance_structures),
            "structures": reinsurance_structures,
        },
        "catastrophe_exposure": {
            "mentioned": bool(catastrophe_terms),
            "terms": catastrophe_terms,
        },
        "missing_components": missing_components,
        "confidence": confidence,
        "source_excerpts": {
            "combined_ratio": combined["source_excerpt"],
            "loss_ratio": loss["source_excerpt"],
            "expense_ratio": expense["source_excerpt"],
        },
        "source_labels": {
            "combined_ratio": combined["source_label"],
            "loss_ratio": loss["source_label"],
            "expense_ratio": expense["source_label"],
        },
        "source_references": {
            "filing": filing_meta or UNKNOWN,
            "source_policy": "latest_cached_annual_primary_filing",
        },
        "metric_family": "pc_insurance",
    }


def build_mortgage_insurance_operating_metrics(
    ticker: str,
    *,
    as_of_date: str | None = None,
    routing: Any = None,
    pipeline_version: str = "v1",
    issuer_cik: str | None = None,
    aliases: Sequence[str] = (),
    db_path: str | Path | None = None,
    cfg: AppConfig | None = None,
) -> dict[str, Any]:
    """Build deterministic operating evidence for mortgage/title specialty insurers."""
    routing_dict = _routing_dict(routing)
    if not _is_mortgage_applicable(routing_dict):
        return {
            "status": "NOT_APPLICABLE",
            "ticker": ticker.upper(),
            "reason_codes": ["NOT_MORTGAGE_INSURANCE_SUBTYPE"],
            "metric_family": "mortgage_insurance",
        }

    is_v2 = str(pipeline_version or "v1").strip().lower() == "v2"
    if is_v2:
        raw_text, filing_meta = latest_cached_filing_text(
            ticker,
            as_of_date=as_of_date,
            max_chars=_MAX_MORTGAGE_FILING_CHARS,
            issuer_cik=issuer_cik,
            aliases=aliases,
            db_path=db_path,
            cfg=cfg,
        )
    else:
        raw_text, filing_meta = latest_cached_filing_text(
            ticker,
            as_of_date=as_of_date,
            max_chars=_MAX_MORTGAGE_FILING_CHARS,
        )
    if not raw_text:
        return {
            "status": "LIMITED",
            "ticker": ticker.upper(),
            "reason_codes": ["NO_CACHED_ANNUAL_FILING"],
            "missing_components": [
                "PMIER_EXCESS_RATIO",
                "PRIMARY_IIF",
                "PRIMARY_RIF",
                "NIW",
                "DEFAULT_RATE",
            ],
            "source_references": {},
            "metric_family": "mortgage_insurance",
        }

    text = strip_html(raw_text)
    pmier = _extract_pmier_sufficiency(text)
    summary = _extract_mortgage_summary_values(text)
    table = _extract_mortgage_portfolio_table_values(text)
    default_narrative = _extract_mortgage_default_narrative(text)
    claims_paid = _extract_mortgage_claims_paid(text)
    reserve = _reserve_development(text)
    reinsurance_structures = _matched_terms(text, _MORTGAGE_REINSURANCE_PATTERNS)

    primary_iif = summary.get("primary_iif_billion") or table.get("primary_iif_billion")
    primary_rif = summary.get("primary_rif_billion") or table.get("primary_rif_billion")
    niw = summary.get("new_insurance_written_billion") or table.get("new_insurance_written_billion")
    default_rate = default_narrative.get("default_rate") or table.get("default_rate")
    policies_in_force = default_narrative.get("policies_in_force") or table.get("policies_in_force")
    loans_in_default = default_narrative.get("loans_in_default") or table.get("loans_in_default")

    missing_components: list[str] = []
    if pmier["pmier_excess_ratio"] is None:
        missing_components.append("PMIER_EXCESS_RATIO")
    if primary_iif is None:
        missing_components.append("PRIMARY_IIF")
    if primary_rif is None:
        missing_components.append("PRIMARY_RIF")
    if niw is None:
        missing_components.append("NIW")
    if default_rate is None:
        missing_components.append("DEFAULT_RATE")
    if policies_in_force is None:
        missing_components.append("POLICIES_IN_FORCE")

    found_any = (
        any(
            item is not None
            for item in (
                pmier["pmier_excess_ratio"],
                primary_iif,
                primary_rif,
                niw,
                default_rate,
                loans_in_default,
                policies_in_force,
                claims_paid.get("claims_paid_count"),
            )
        )
        or reserve["status"] != UNKNOWN
        or bool(reinsurance_structures)
    )

    confidence = "LOW"
    if (
        pmier["pmier_excess_ratio"] is not None
        and primary_iif is not None
        and default_rate is not None
    ):
        confidence = "HIGH"
    elif found_any:
        confidence = "MEDIUM"

    return {
        "status": "OK" if found_any else "LIMITED",
        "ticker": ticker.upper(),
        "as_of_date": as_of_date,
        "insurance_subtype": routing_dict.get("insurance_subtype"),
        "metric_family": "mortgage_insurance",
        "pmier_excess_ratio": pmier["pmier_excess_ratio"],
        "pmier_available_to_required_ratio": pmier["pmier_available_to_required_ratio"],
        "primary_iif_billion": primary_iif,
        "primary_rif_billion": primary_rif,
        "new_insurance_written_billion": niw,
        "customer_count": summary.get("customer_count"),
        "policies_in_force": policies_in_force,
        "loans_in_default": loans_in_default,
        "default_rate": default_rate,
        "rif_on_defaulted_loans_billion": table.get("rif_on_defaulted_loans_billion"),
        "annual_persistency": table.get("annual_persistency"),
        "quarterly_runoff": table.get("quarterly_runoff"),
        "claims_paid_count": claims_paid.get("claims_paid_count"),
        "claims_paid_million": claims_paid.get("claims_paid_million"),
        "credit_capital_assessment": _mortgage_default_assessment(
            default_rate, pmier["pmier_excess_ratio"]
        ),
        "reserve_development": reserve,
        "reinsurance_program": {
            "mentioned": bool(reinsurance_structures),
            "structures": reinsurance_structures,
        },
        "missing_components": missing_components,
        "confidence": confidence,
        "source_excerpts": {
            "pmier_sufficiency": pmier["source_excerpt"],
            "portfolio_summary": summary.get("source_excerpt"),
            "portfolio_table": table.get("source_excerpt"),
            "default_rate": default_narrative.get("source_excerpt"),
            "claims_paid": claims_paid.get("source_excerpt"),
        },
        "source_references": {
            "filing": filing_meta or UNKNOWN,
            "source_policy": "latest_cached_annual_primary_filing",
        },
    }


def build_insurance_operating_metrics(
    ticker: str,
    *,
    as_of_date: str | None = None,
    routing: Any = None,
    pipeline_version: str = "v1",
    issuer_cik: str | None = None,
    aliases: Sequence[str] = (),
    db_path: str | Path | None = None,
    cfg: AppConfig | None = None,
) -> dict[str, Any]:
    """Dispatch operating evidence extraction by insurance subtype."""
    routing_dict = _routing_dict(routing)
    if _is_pc_applicable(routing_dict):
        return build_pc_operating_metrics(
            ticker,
            as_of_date=as_of_date,
            routing=routing_dict,
            pipeline_version=pipeline_version,
            issuer_cik=issuer_cik,
            aliases=aliases,
            db_path=db_path,
            cfg=cfg,
        )
    if _is_mortgage_applicable(routing_dict):
        return build_mortgage_insurance_operating_metrics(
            ticker,
            as_of_date=as_of_date,
            routing=routing_dict,
            pipeline_version=pipeline_version,
            issuer_cik=issuer_cik,
            aliases=aliases,
            db_path=db_path,
            cfg=cfg,
        )
    return {
        "status": "NOT_APPLICABLE",
        "ticker": ticker.upper(),
        "reason_codes": ["NO_INSURANCE_OPERATING_METRICS_FOR_SUBTYPE"],
        "metric_family": str(routing_dict.get("insurance_subtype") or UNKNOWN),
    }
