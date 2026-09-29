"""Build and persist insurance-specific alpha evidence packets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from app.config import AppConfig
from app.db import get_db, utc_now_iso
from app.insurance.routing import (
    ISSUER_INSURANCE_UNDERWRITER,
    MODEL_BLOCKED,
    MODEL_NOT_APPLICABLE,
    SECURITY_COMMON,
    SECURITY_DEPOSITARY,
    SECURITY_PREFERRED,
    route_security,
)
from app.insurance.operating_metrics import build_insurance_operating_metrics
from app.insurance.sources import latest_scorecard
from app.insurance.valuation import (
    calculate_insurance_common_valuation,
    calculate_insurance_preferred_valuation,
)
from app.util.financial_data_access import normalize_cik
from app.valuation.lineage import latest_decision_eligible_valuation_row


def _current_price_from_scorecard(scorecard: dict[str, Any] | None) -> float | None:
    if not isinstance(scorecard, dict):
        return None
    pzd = scorecard.get("pricing_zone_detail")
    if not isinstance(pzd, dict):
        return None
    price = pzd.get("current_price")
    return float(price) if isinstance(price, (int, float)) and price > 0 else None


def _write_valuation_method(
    *,
    ticker: str,
    as_of_date: str,
    method: str,
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    warnings: list[str],
) -> None:
    now = utc_now_iso()
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO valuations(ticker, as_of_date, method, inputs_json, outputs_json, warnings_json, created_at)
            VALUES(?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker, as_of_date, method) DO UPDATE SET
                inputs_json = excluded.inputs_json,
                outputs_json = excluded.outputs_json,
                warnings_json = excluded.warnings_json,
                created_at = excluded.created_at
            """,
            (
                ticker.upper(),
                as_of_date,
                method,
                json.dumps(inputs, sort_keys=True),
                json.dumps(outputs, sort_keys=True),
                json.dumps(warnings),
                now,
            ),
        )


def _combined_status(routing: dict[str, Any], valuation: dict[str, Any]) -> str:
    if routing.get("model_status") == MODEL_NOT_APPLICABLE:
        return MODEL_NOT_APPLICABLE
    if routing.get("model_status") == MODEL_BLOCKED:
        return MODEL_BLOCKED
    if valuation.get("model_status"):
        return str(valuation["model_status"])
    if valuation.get("status"):
        return str(valuation["status"])
    return MODEL_BLOCKED


def build_insurance_packet(
    ticker: str,
    *,
    as_of_date: str | None = None,
    scorecard: dict[str, Any] | None = None,
    persist: bool = True,
    pipeline_version: str = "v1",
    current_price_override: float | None = None,
    issuer_cik: str | None = None,
    aliases: Sequence[str] = (),
    db_path: str | Path | None = None,
    cfg: AppConfig | None = None,
) -> dict[str, Any]:
    """Build routing, packet, and valuation outputs for alpha scan consumers."""
    upper = ticker.upper()
    normalized_pipeline = str(pipeline_version or "v1").strip().lower()
    if normalized_pipeline not in {"v1", "v2"}:
        raise ValueError("pipeline_version must be v1 or v2")
    is_v2 = normalized_pipeline == "v2"
    fixed_as_of_date = str(as_of_date or "").strip()[:10] or None
    if is_v2 and fixed_as_of_date is None:
        raise ValueError("v2 insurance packet requires as_of_date")
    identity_bound_fixed_asof = fixed_as_of_date is not None
    if identity_bound_fixed_asof and normalize_cik(issuer_cik) is None:
        raise ValueError("fixed-as-of insurance packet requires exact issuer_cik binding")

    if fixed_as_of_date is not None:
        if scorecard is None:
            scorecard_as_of, loaded_scorecard = latest_scorecard(
                upper,
                as_of_date=fixed_as_of_date,
                issuer_cik=issuer_cik,
                aliases=aliases,
                require_exact_issuer_binding=True,
                db_path=db_path,
                cfg=cfg,
            )
        else:
            scorecard_as_of, loaded_scorecard = fixed_as_of_date, {}
    else:
        scorecard_as_of, loaded_scorecard = latest_scorecard(upper)
    if scorecard is None:
        scorecard = loaded_scorecard
    resolved_as_of = fixed_as_of_date or scorecard_as_of or utc_now_iso()[:10]
    if current_price_override is not None or identity_bound_fixed_asof:
        current_price = (
            float(current_price_override)
            if isinstance(current_price_override, (int, float))
            and not isinstance(current_price_override, bool)
            and current_price_override > 0
            else None
        )
    else:
        current_price = _current_price_from_scorecard(scorecard)

    if identity_bound_fixed_asof:
        routing_result = route_security(
            upper,
            as_of_date=resolved_as_of,
            pipeline_version="v2",
            issuer_cik=issuer_cik,
            aliases=aliases,
            db_path=db_path,
            cfg=cfg,
        )
    else:
        routing_result = route_security(upper, as_of_date=resolved_as_of)
    routing = routing_result.to_dict()
    security_type = routing.get("security_type")
    issuer_type = routing.get("issuer_type")

    valuation: dict[str, Any]
    if security_type in (SECURITY_PREFERRED, SECURITY_DEPOSITARY):
        if identity_bound_fixed_asof:
            valuation = calculate_insurance_preferred_valuation(
                upper,
                as_of_date=resolved_as_of,
                routing=routing,
                current_price=current_price,
                pipeline_version="v2",
                issuer_cik=issuer_cik,
                aliases=aliases,
                db_path=db_path,
                cfg=cfg,
            )
        else:
            valuation = calculate_insurance_preferred_valuation(
                upper,
                as_of_date=resolved_as_of,
                routing=routing,
                current_price=current_price,
            )
    elif security_type == SECURITY_COMMON and issuer_type == ISSUER_INSURANCE_UNDERWRITER:
        if identity_bound_fixed_asof:
            valuation = calculate_insurance_common_valuation(
                upper,
                as_of_date=resolved_as_of,
                routing=routing,
                current_price=current_price,
                pipeline_version="v2",
                issuer_cik=issuer_cik,
                aliases=aliases,
                db_path=db_path,
                cfg=cfg,
            )
        else:
            valuation = calculate_insurance_common_valuation(
                upper,
                as_of_date=resolved_as_of,
                routing=routing,
                current_price=current_price,
            )
    else:
        valuation = {
            "status": "NOT_APPLICABLE",
            "model_status": MODEL_NOT_APPLICABLE,
            "method": None,
            "ticker": upper,
            "reason_codes": ["NOT_INSURANCE_VALUATION_TARGET"],
        }

    if identity_bound_fixed_asof:
        operating_metrics = build_insurance_operating_metrics(
            upper,
            as_of_date=resolved_as_of,
            routing=routing,
            pipeline_version="v2",
            issuer_cik=issuer_cik,
            aliases=aliases,
            db_path=db_path,
            cfg=cfg,
        )
    else:
        operating_metrics = build_insurance_operating_metrics(
            upper,
            as_of_date=resolved_as_of,
            routing=routing,
        )
    generic_invalid = (
        security_type not in (SECURITY_COMMON, "SECURITY_TYPE_UNKNOWN")
        or issuer_type == ISSUER_INSURANCE_UNDERWRITER
    )
    operating_metric_warnings: list[str] = []
    if operating_metrics.get("status") == "LIMITED":
        metric_family = str(operating_metrics.get("metric_family") or "")
        if metric_family == "mortgage_insurance":
            operating_metric_warnings.append("MORTGAGE_OPERATING_METRICS_LIMITED")
        elif metric_family == "pc_insurance":
            operating_metric_warnings.append("PC_OPERATING_METRICS_LIMITED")
    if operating_metrics.get("combined_ratio_assessment") in (
        "UNDERWRITING_LOSS_WATCH",
        "UNDERWRITING_LOSS",
    ):
        operating_metric_warnings.append(str(operating_metrics["combined_ratio_assessment"]))
    if operating_metrics.get("credit_capital_assessment") in (
        "MORTGAGE_CREDIT_STRESS",
        "PMIER_CAPITAL_THIN",
    ):
        operating_metric_warnings.append(str(operating_metrics["credit_capital_assessment"]))
    model_status = _combined_status(routing, valuation)
    if identity_bound_fixed_asof and model_status == MODEL_BLOCKED:
        # A fixed-as-of run treats missing identity, facts, terms, or price as
        # an explicit evidence gap, never as a business-quality rejection.
        model_status = "NEEDS_DATA"
    reason_codes = list(
        dict.fromkeys(
            [str(item) for item in routing.get("reason_codes") or []]
            + [str(item) for item in valuation.get("reason_codes") or []]
        )
    )
    warnings = list(
        dict.fromkeys(
            [str(item) for item in routing.get("model_fit_warnings") or []]
            + [str(item) for item in valuation.get("missing_components") or []]
            + operating_metric_warnings
            + (["GENERIC_DCF_EPV_SUPPRESSED"] if generic_invalid else [])
        )
    )
    packet = {
        "ticker": upper,
        "as_of_date": resolved_as_of,
        "routing": routing,
        "valuation": valuation,
        "operating_metrics": operating_metrics,
        "model_status": model_status,
        "model_blockers": reason_codes if model_status in {MODEL_BLOCKED, "NEEDS_DATA"} else [],
        "model_fit_warnings": warnings,
        "generic_valuation_valid": not generic_invalid,
        "generic_valuation_policy": (
            "generic_dcf_epv_blocked_for_insurance_or_non_common"
            if generic_invalid
            else "generic_dcf_epv_allowed"
        ),
        "evidence_policy": (
            f"issuer_bound_fixed_asof_primary_source_{normalized_pipeline}"
            if identity_bound_fixed_asof
            else "primary_source_only_v1"
        ),
    }
    if is_v2:
        packet["pipeline_version"] = "v2"

    # V2 packet assembly is a pure read. Its caller owns artifact persistence;
    # it must never mutate the configured/global valuation store.
    if persist and not is_v2 and model_status != MODEL_NOT_APPLICABLE:
        _write_valuation_method(
            ticker=upper,
            as_of_date=resolved_as_of,
            method="security_routing",
            inputs={"ticker": upper, "as_of_date": resolved_as_of},
            outputs=routing,
            warnings=list(routing.get("model_fit_warnings") or []),
        )
        if valuation.get("method"):
            _write_valuation_method(
                ticker=upper,
                as_of_date=resolved_as_of,
                method=str(valuation["method"]),
                inputs={"ticker": upper, "as_of_date": resolved_as_of, "routing": routing},
                outputs=valuation,
                warnings=[str(item) for item in valuation.get("missing_components") or []],
            )
        _write_valuation_method(
            ticker=upper,
            as_of_date=resolved_as_of,
            method="insurance_packet",
            inputs={"ticker": upper, "as_of_date": resolved_as_of},
            outputs=packet,
            warnings=warnings,
        )
    return packet


def load_latest_insurance_packet(ticker: str) -> dict[str, Any]:
    """Load the latest persisted insurance packet, if present."""
    with get_db() as conn:
        row = latest_decision_eligible_valuation_row(
            conn,
            ticker=ticker,
            method="insurance_packet",
        )
    if not row:
        return {}
    try:
        return json.loads(row["outputs_json"] or "{}")
    except json.JSONDecodeError:
        return {}
