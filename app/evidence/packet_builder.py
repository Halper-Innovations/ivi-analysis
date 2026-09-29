from __future__ import annotations

import json
import math
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from app.config import get_config
from app.db import get_db, utc_now_iso
from app.evidence.schemas import (
    Citation,
    EvidencePacket,
    ExtractedFactRecord,
    FilingMetadata,
    FinancialRecord,
)
from app.ingest.filings import default_filing_windows_days, select_filing_stubs_for_policy
from app.ingest.companyfacts import normalize_annual_facts_from_raw
from app.ingest.sec_client import SecClient
from app.logging import get_logger
from app.market.company_facts_extract import (
    extract_company_facts_asof,
)
from app.market.company_facts_provider import companyfacts_cache_path
from app.market.price_provider import resolve_price_from_historical_runs
from app.util.hashing import sha256_file
from app.util.financial_data_access import (
    ANNUAL_COMPANYFACTS_PERIOD_TYPES,
    issuer_companyfacts_rows,
)
from app.valuation.facts import resolve_cik_for_ticker
from app.valuation.fundamentals import UNKNOWN, build_fundamentals_frame
from app.valuation.share_splits import (
    split_adjust_share_series,
    split_ratio_rows_from_companyfacts,
)
from app.valuation.lineage import (
    latest_decision_eligible_valuation_row,
    latest_decision_eligible_valuation_rows,
)


logger = get_logger(__name__)

_PROMPT_FINANCIAL_PROVENANCE_KEY = "_prompt_financial_provenance"
_COMPANYFACTS_SHARES_UNIT = "shares_millions"
_COMPANYFACTS_MONETARY_UNIT = "USD_millions"


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float))


def _literal_companyfacts_unit(
    row: dict[str, Any],
    *,
    metric: str,
) -> str | None:
    """Return the exact canonical unit only when the source row states it."""

    unit = str(row.get("units") or "").strip()
    expected = (
        _COMPANYFACTS_SHARES_UNIT if metric == "shares_outstanding" else _COMPANYFACTS_MONETARY_UNIT
    )
    return unit if unit == expected else None


def _companyfacts_row_has_prompt_lineage(
    row: dict[str, Any],
    *,
    metric: str,
) -> bool:
    """Require literal units and filing lineage before a fact reaches a prompt."""

    value = row.get("value")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or _literal_companyfacts_unit(row, metric=metric) is None
    ):
        return False
    return all(
        str(row.get(field) or "").strip()
        for field in ("period_end", "filed_date", "source_url", "accession")
    )


def _companyfacts_fact_provenance(
    row: dict[str, Any],
    *,
    metric: str,
    value: float,
    unit: str,
) -> dict[str, Any] | None:
    literal_unit = _literal_companyfacts_unit(row, metric=metric)
    if (
        literal_unit is None
        or unit != literal_unit
        or not _companyfacts_row_has_prompt_lineage(row, metric=metric)
    ):
        return None
    source_url = str(row.get("source_url") or "").strip()
    period_end = str(row.get("period_end") or "").strip()[:10]
    filed_date = str(row.get("filed_date") or "").strip()[:10]
    accession = str(row.get("accession") or "").strip()
    form = str(row.get("form") or "").strip()
    reference_suffix = accession or form or period_end
    return {
        "value": float(value),
        "unit": str(unit),
        "source": "SEC_companyfacts",
        "period_end": period_end,
        "filed_date": filed_date,
        "source_reference": (
            f"{source_url}#{metric}:{reference_suffix}"
            if source_url
            else f"companyfacts_facts.{metric}:{reference_suffix}"
        ),
        "source_url": source_url,
        "accession": accession,
        "form": form,
    }


def _derived_companyfacts_provenance(
    *,
    metric: str,
    value: float,
    unit: str,
    formula: str,
    inputs: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    input_provenance: dict[str, dict[str, Any]] = {}
    for name, row in inputs.items():
        input_metric = str(row.get("line_item") or name)
        if not _companyfacts_row_has_prompt_lineage(
            row,
            metric=input_metric,
        ):
            return None
        input_unit = _literal_companyfacts_unit(
            row,
            metric=input_metric,
        )
        if input_unit is None:
            return None
        provenance = _companyfacts_fact_provenance(
            row,
            metric=input_metric,
            value=float(row["value"]),
            unit=input_unit,
        )
        if provenance is None:
            return None
        input_provenance[name] = provenance
    filed_dates = [
        str(record.get("filed_date") or "")
        for record in input_provenance.values()
        if str(record.get("filed_date") or "")
    ]
    period_ends = [
        str(record.get("period_end") or "")
        for record in input_provenance.values()
        if str(record.get("period_end") or "")
    ]
    references = [
        str(record.get("source_reference") or "")
        for record in input_provenance.values()
        if str(record.get("source_reference") or "")
    ]
    return {
        "value": float(value),
        "unit": str(unit),
        "source": "derived:SEC_companyfacts",
        "period_end": max(period_ends, default=""),
        "filed_date": max(filed_dates, default=""),
        "source_reference": "|".join(references),
        "formula": formula,
        "input_provenance": input_provenance,
    }


def _restore_prompt_financial_provenance(
    payload: dict[str, Any],
    *,
    source_extracted_facts: list[dict[str, Any]],
    source_financials: list[dict[str, Any]],
) -> dict[str, Any]:
    """Restore provenance fields after Pydantic normalizes evidence records."""

    for output_row, source_row in zip(
        payload.get("extracted_facts") or [],
        source_extracted_facts,
        strict=False,
    ):
        if isinstance(output_row, dict) and isinstance(source_row.get("provenance"), dict):
            output_row["provenance"] = dict(source_row["provenance"])
    for output_row, source_row in zip(
        payload.get("financials") or [],
        source_financials,
        strict=False,
    ):
        if isinstance(output_row, dict) and isinstance(source_row.get("provenance"), dict):
            output_row["provenance"] = dict(source_row["provenance"])

    fundamentals = payload.get("fundamentals")
    if isinstance(fundamentals, dict):
        raw_provenance = fundamentals.pop(_PROMPT_FINANCIAL_PROVENANCE_KEY, {})
        if isinstance(raw_provenance, dict) and raw_provenance:
            payload["fundamentals_provenance"] = raw_provenance
    return payload


def _safe_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _get_latest_as_of(conn, table: str, ticker: str) -> str | None:
    row = conn.execute(
        f"SELECT MAX(as_of_date) as as_of_date FROM {table} WHERE ticker = ?",
        (ticker,),
    ).fetchone()
    return row["as_of_date"] if row and row["as_of_date"] else None


def _cached_filing_metadata_for_ticker(
    *, ticker: str, as_of_date: str, limit: int = 4
) -> list[dict[str, Any]]:
    cik = resolve_cik_for_ticker(ticker, refresh_if_missing=False)
    if not cik:
        return []
    try:
        end_date = date.fromisoformat(as_of_date)
    except Exception:
        return []
    start_date = end_date - timedelta(days=500)
    client = SecClient()
    stubs = client.list_cached_filings_window(
        str(cik),
        start_date=start_date,
        end_date=end_date,
        forms=["10-K", "10-Q", "8-K", "20-F", "DEF 14A"],
    )
    selected = select_filing_stubs_for_policy(
        stubs,
        as_of_date=end_date,
        windows_days=default_filing_windows_days(),
    )
    if not selected:
        selected = stubs[:limit]
    selected = selected[:limit]
    return [
        FilingMetadata(
            accession=stub.accession,
            form_type=stub.form_type,
            filing_date=stub.filing_date.isoformat(),
            period_end=stub.period_end,
            primary_doc_url=stub.primary_doc_url,
        ).model_dump()
        for stub in selected
    ]


def _latest_valuation_as_of(conn, ticker: str) -> str | None:
    row = latest_decision_eligible_valuation_row(conn, ticker=ticker)
    return row["as_of_date"] if row and row["as_of_date"] else None


def _build_packet_payload(conn, ticker: str, as_of_date: str) -> dict[str, Any] | None:
    fundamentals_row = conn.execute(
        "SELECT metrics_json FROM fundamentals WHERE ticker = ? AND as_of_date = ?",
        (ticker, as_of_date),
    ).fetchone()
    if not fundamentals_row:
        return None

    valuation_rows = latest_decision_eligible_valuation_rows(
        conn,
        ticker=ticker,
        as_of_date=as_of_date,
        exact_as_of_date=True,
    )

    filing_rows = conn.execute(
        """
        SELECT id, accession, form_type, filing_date, period_end, primary_doc_url
        FROM filings
        WHERE ticker = ?
          AND filing_date IS NOT NULL AND filing_date <= ?
        ORDER BY filing_date DESC
        LIMIT 4
        """,
        (ticker, as_of_date),
    ).fetchall()

    filing_ids = [row["id"] for row in filing_rows]
    extracted_rows = []
    financial_rows = []
    if filing_ids:
        placeholders = ",".join(["?"] * len(filing_ids))
        extracted_rows = conn.execute(
            f"""
            SELECT fact_type, value_json, source_url, snippet, section_label
            FROM extracted_facts
            WHERE filing_id IN ({placeholders})
            ORDER BY id DESC
            """,
            filing_ids,
        ).fetchall()
        financial_rows = conn.execute(
            f"""
            SELECT statement_type, line_item, value, units, period, source_url, snippet
            FROM financials
            WHERE filing_id IN ({placeholders})
            ORDER BY id DESC
            """,
            filing_ids,
        ).fetchall()

    filings = [
        FilingMetadata(
            accession=row["accession"],
            form_type=row["form_type"],
            filing_date=row["filing_date"],
            period_end=row["period_end"],
            primary_doc_url=row["primary_doc_url"],
        ).model_dump()
        for row in filing_rows
    ]
    if not filings:
        filings = _cached_filing_metadata_for_ticker(ticker=ticker, as_of_date=as_of_date)

    facts = [
        ExtractedFactRecord(
            fact_type=row["fact_type"],
            value=json.loads(row["value_json"]),
            citation=Citation(
                source_url=row["source_url"],
                snippet=row["snippet"] or "",
                section_label=row["section_label"],
            ),
        ).model_dump()
        for row in extracted_rows
    ]

    financials = [
        FinancialRecord(
            statement_type=row["statement_type"],
            line_item=row["line_item"],
            value=row["value"],
            units=row["units"],
            period=row["period"],
            citation=Citation(
                source_url=row["source_url"] or "",
                snippet=row["snippet"] or "",
                section_label=None,
            ),
        ).model_dump()
        for row in financial_rows
    ]
    current_metrics = json.loads(fundamentals_row["metrics_json"])
    valuations: dict[str, Any] = {}
    for row in valuation_rows:
        valuations[row["method"]] = {
            "inputs": json.loads(row["inputs_json"]),
            "outputs": json.loads(row["outputs_json"]),
            "warnings": json.loads(row["warnings_json"]),
        }
    _overlay_missing_fundamentals(
        ticker=ticker,
        as_of_date=as_of_date,
        fundamentals=current_metrics,
        extracted_facts=facts,
        financials=financials,
    )
    _overlay_companyfacts_trends(
        conn,
        ticker=ticker,
        as_of_date=as_of_date,
        fundamentals=current_metrics,
    )
    _overlay_companyfacts_support(
        conn,
        ticker=ticker,
        as_of_date=as_of_date,
        fundamentals=current_metrics,
        extracted_facts=facts,
        financials=financials,
    )
    _overlay_bank_credit_filing_fallbacks(
        fundamentals=current_metrics,
        financials=financials,
    )
    _overlay_missing_price_inputs(
        conn,
        ticker=ticker,
        as_of_date=as_of_date,
        valuations=valuations,
    )
    _overlay_missing_valuation_inputs(
        fundamentals=current_metrics,
        valuations=valuations,
    )
    _overlay_tech_adjusted_valuation_summary(
        fundamentals=current_metrics,
        valuations=valuations,
    )

    prior_row = conn.execute(
        """
        SELECT metrics_json
        FROM fundamentals
        WHERE ticker = ? AND as_of_date < ?
        ORDER BY as_of_date DESC
        LIMIT 1
        """,
        (ticker, as_of_date),
    ).fetchone()
    prior_metrics = json.loads(prior_row["metrics_json"]) if prior_row else {}
    deltas: dict[str, Any] = {}
    for k, current in current_metrics.items():
        prior = prior_metrics.get(k)
        if isinstance(current, (int, float)) and isinstance(prior, (int, float)):
            deltas[k] = current - prior

    packet = EvidencePacket(
        ticker=ticker,
        as_of_date=as_of_date,
        filings_used=filings,
        extracted_facts=facts,
        financials=financials,
        fundamentals=current_metrics,
        valuations=valuations,
        deltas_vs_prior_period=deltas,
    )
    payload = packet.model_dump()
    return _restore_prompt_financial_provenance(
        payload,
        source_extracted_facts=facts,
        source_financials=financials,
    )


def _overlay_tech_adjusted_valuation_summary(
    *, fundamentals: dict[str, Any], valuations: dict[str, Any]
) -> None:
    if not isinstance(fundamentals, dict) or not isinstance(valuations, dict):
        return

    tech_adjustment = valuations.get("tech_adjustment")
    if isinstance(tech_adjustment, dict):
        outputs = (
            tech_adjustment.get("outputs")
            if isinstance(tech_adjustment.get("outputs"), dict)
            else tech_adjustment
        )
        category_payload = (
            outputs.get("category_classification")
            if isinstance(outputs.get("category_classification"), dict)
            else {}
        )
        rnd_payload = (
            outputs.get("rnd_adjustment") if isinstance(outputs.get("rnd_adjustment"), dict) else {}
        )
        divergence = outputs.get("tech_valuation_divergence")
        if category_payload:
            fundamentals["tech_category"] = category_payload.get("category", UNKNOWN)
            fundamentals["tech_category_confidence"] = category_payload.get("confidence", UNKNOWN)
        if rnd_payload:
            fundamentals["rnd_amortization_life"] = rnd_payload.get("amortization_life", UNKNOWN)
            fundamentals["rnd_adjustment"] = rnd_payload.get("rnd_adjustment", UNKNOWN)
            fundamentals["rnd_capitalized_asset"] = rnd_payload.get(
                "rnd_capitalized_asset", UNKNOWN
            )
            fundamentals["rnd_adjustment_status"] = rnd_payload.get("status", UNKNOWN)
        if isinstance(divergence, (int, float)):
            fundamentals["tech_valuation_divergence"] = float(divergence)

    dcf_adjusted = valuations.get("dcf_adjusted")
    if isinstance(dcf_adjusted, dict):
        outputs = (
            dcf_adjusted.get("outputs")
            if isinstance(dcf_adjusted.get("outputs"), dict)
            else dcf_adjusted
        )
        if isinstance(outputs.get("base"), (int, float)):
            fundamentals["dcf_adjusted_base"] = float(outputs["base"])
    epv_adjusted = valuations.get("epv_adjusted")
    if isinstance(epv_adjusted, dict):
        outputs = (
            epv_adjusted.get("outputs")
            if isinstance(epv_adjusted.get("outputs"), dict)
            else epv_adjusted
        )
        if isinstance(outputs.get("value_per_share"), (int, float)):
            fundamentals["epv_adjusted_value_per_share"] = float(outputs["value_per_share"])


def _attach_pattern_scan_summary(
    payload: dict[str, Any],
    *,
    ticker: str,
    run_id: str | None,
) -> dict[str, Any]:
    if not isinstance(payload, dict) or not str(run_id or "").strip():
        return payload
    from app.patterns.scanner import load_pattern_scan_report, summarize_pattern_scan_for_ticker

    report = load_pattern_scan_report(str(run_id).strip())
    if report is None:
        return payload
    summary = summarize_pattern_scan_for_ticker(report, ticker)
    if summary.get("pattern_hit_count"):
        payload["cross_filing_patterns"] = summary
    return payload


def _attach_variant_perceptions(
    payload: dict[str, Any],
    *,
    ticker: str,
    as_of_date: str,
    run_id: str | None,
) -> dict[str, Any]:
    if not isinstance(payload, dict) or not str(run_id or "").strip():
        return payload
    from app.synthesis.variant_builder import build_variant_perceptions

    report = build_variant_perceptions(
        ticker=ticker,
        as_of_date=as_of_date,
        run_id=str(run_id).strip(),
        valuation_payload=payload.get("valuations")
        if isinstance(payload.get("valuations"), dict)
        else None,
    )
    payload["variant_perceptions"] = report.model_dump(mode="json")
    return payload


def _latest_dossier_payload(
    *, ticker: str, as_of_date: str, run_id: str | None = None
) -> dict[str, Any] | None:
    cfg = get_config()
    ticker_norm = ticker.upper()
    candidates: list[tuple[float, Path]] = []
    if run_id:
        path = cfg.dossiers_dir / run_id / ticker_norm / "dossier.json"
        if path.exists():
            candidates.append((path.stat().st_mtime, path))
    else:
        for path in cfg.dossiers_dir.glob(f"*/{ticker_norm}/dossier.json"):
            if path.exists():
                candidates.append((path.stat().st_mtime, path))
    for _, path in sorted(candidates, key=lambda item: item[0], reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(payload, dict):
            continue
        payload_as_of = str(payload.get("as_of_date") or "")
        if payload_as_of and payload_as_of <= as_of_date:
            return payload
    return None


_DOSSIER_EXPECTED_UNITS_BY_METRIC = {
    "shares_outstanding": "shares_millions",
    "gross_margin": "ratio",
    "op_margin": "ratio",
    "operating_margin": "ratio",
    "fcf_margin": "ratio",
    "cfo_margin": "ratio",
    "r_and_d_pct_revenue": "ratio",
    "sales_marketing_pct_revenue": "ratio",
    "g_and_a_pct_revenue": "ratio",
    "customer_concentration_pct": "ratio",
}


def _literal_dossier_item_unit(
    item: dict[str, Any],
    *,
    metric: str,
) -> str | None:
    """Read a stated dossier unit without inferring one from the metric."""

    existing = item.get("provenance")
    unit = str(item.get("units") or item.get("unit") or "").strip()
    if not unit and isinstance(existing, dict):
        unit = str(existing.get("unit") or "").strip()
    expected = _DOSSIER_EXPECTED_UNITS_BY_METRIC.get(metric)
    if not unit or (expected is not None and unit != expected):
        return None
    return unit


def _dossier_item_provenance(
    item: dict[str, Any],
    *,
    metric: str,
    value: Any,
    unit: str | None,
) -> dict[str, Any] | None:
    """Preserve exact dossier provenance without fabricating missing fields."""

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or unit is None
    ):
        return None
    existing = item.get("provenance")
    if isinstance(existing, dict):
        required = (
            "value",
            "unit",
            "source",
            "period_end",
            "filed_date",
            "source_reference",
            "source_url",
            "accession",
        )
        existing_value = existing.get("value")
        if (
            all(str(existing.get(field) or "").strip() for field in required[1:])
            and not isinstance(existing_value, bool)
            and isinstance(existing_value, (int, float))
            and math.isfinite(float(existing_value))
            and math.isclose(
                float(existing_value),
                float(value),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            and str(existing.get("unit") or "").strip() == unit
        ):
            return dict(existing)
        return None
    source_url = str(item.get("source_url") or "").strip()
    accession = str(item.get("filing_accession") or item.get("accession") or "").strip()
    filed_date = str(item.get("filing_date") or "").strip()[:10]
    period_end = str(item.get("period_end") or "").strip()[:10]
    if not all((source_url, accession, filed_date, period_end)):
        return None
    return {
        "value": float(value),
        "unit": unit,
        "source": "SEC_filing_extractor",
        "period_end": period_end,
        "filed_date": filed_date,
        "source_reference": f"{source_url}#{metric}:{accession}",
        "source_url": source_url,
        "accession": accession,
    }


def _financials_from_dossier(dossier: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in dossier.get("items") or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("section_label") or "") != "financial_statements":
            continue
        metric = str(item.get("metric") or "")
        value = item.get("value")
        unit = _literal_dossier_item_unit(item, metric=metric)
        payload = FinancialRecord(
            statement_type="companyfacts",
            line_item=metric,
            value=float(value) if isinstance(value, (int, float)) else None,
            units=unit,
            period=str(item.get("year") or ""),
            citation=Citation(
                source_url=str(item.get("source_url") or ""),
                snippet=str(item.get("snippet") or ""),
                section_label="financial_statements",
            ),
        ).model_dump()
        provenance = _dossier_item_provenance(
            item,
            metric=metric,
            value=value,
            unit=unit,
        )
        if provenance is not None:
            payload["provenance"] = provenance
        rows.append(payload)
    return rows


def _facts_from_dossier(dossier: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in (dossier.get("items") or [])[:200]:
        if not isinstance(item, dict):
            continue
        metric = str(item.get("metric") or "unknown_fact")
        value = item.get("value")
        payload = ExtractedFactRecord(
            fact_type=metric,
            value={
                "year": item.get("year"),
                "metric": item.get("metric"),
                "value": value,
                "derived_from": [
                    str(x) for x in (item.get("derived_from") or []) if str(x).strip()
                ],
                "citations": [x for x in (item.get("citations") or []) if isinstance(x, dict)],
            },
            citation=Citation(
                source_url=str(item.get("source_url") or ""),
                snippet=str(item.get("snippet") or ""),
                section_label=str(item.get("section_label") or "") or None,
            ),
        ).model_dump()
        provenance = _dossier_item_provenance(
            item,
            metric=metric,
            value=value,
            unit=_literal_dossier_item_unit(item, metric=metric),
        )
        if provenance is not None:
            payload["provenance"] = provenance
        out.append(payload)
    return out


def _flat_fundamentals_from_frame(frame: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in (frame.get("rows") or []) if isinstance(row, dict)]
    latest = rows[-1] if rows else {}
    derived = frame.get("derived_signals") or {}
    payload: dict[str, Any] = {
        "ticker": frame.get("ticker"),
        "run_id": frame.get("run_id"),
        "as_of_date": frame.get("as_of_date"),
        "rows": rows,
        "row_traces": frame.get("row_traces") or {},
        "derived_signals": derived,
        "gaps": frame.get("gaps") or [],
    }
    for key in (
        "revenue",
        "gross_profit",
        "operating_income",
        "net_income",
        "cfo",
        "capex",
        "fcf",
        "shares_outstanding",
        "net_debt",
        "gross_margin",
        "op_margin",
        "fcf_margin",
        "cfo_margin",
        "r_and_d_total",
        "sales_marketing_total",
        "g_and_a_total",
        "deferred_revenue_amount",
        "rpo_amount",
        "customer_concentration_pct",
        "customer_concentration_present",
        "segment_count",
        "share_repurchases_amount",
        "dividends_paid_amount",
        "deposits",
        "loans",
        "investment_securities",
        "total_assets",
        "assets_under_management",
        "allowance_for_credit_losses",
        "provision_for_credit_losses",
        "net_charge_offs",
        "nonaccrual_loans",
        "deposits_to_assets",
        "loans_to_deposits",
        "allowance_to_loans",
        "provision_for_credit_losses",
        "net_charge_offs",
        "nonaccrual_loans",
        "provision_to_loans",
        "net_charge_offs_to_loans",
        "debt_to_assets",
    ):
        payload[key] = latest.get(key, UNKNOWN)
    payload["operating_margin"] = payload.get("op_margin", UNKNOWN)
    for signal_name, signal_payload in derived.items():
        if isinstance(signal_payload, dict):
            payload[signal_name] = signal_payload.get("value", UNKNOWN)
    return payload


def _safe_div(a: Any, b: Any) -> float | str:
    if not _is_num(a) or not _is_num(b):
        return UNKNOWN
    if float(b) == 0.0:
        return UNKNOWN
    return float(a) / float(b)


def _series_slope(series: list[tuple[int, float]]) -> float | str:
    if len(series) < 2:
        return UNKNOWN
    start_year, start_value = series[0]
    end_year, end_value = series[-1]
    span = max(1, int(end_year) - int(start_year))
    return (float(end_value) - float(start_value)) / float(span)


def _series_cagr(series: list[tuple[int, float]], years: int) -> float | str:
    if len(series) < 2:
        return UNKNOWN
    end_year, end_value = series[-1]
    if float(end_value) <= 0:
        return UNKNOWN
    target_year = int(end_year) - int(years)
    candidates = [
        (year, value) for year, value in series if int(year) <= target_year and float(value) > 0
    ]
    if candidates:
        start_year, start_value = candidates[-1]
    else:
        start_year, start_value = series[0]
    if float(start_value) <= 0:
        return UNKNOWN
    span = max(1, int(end_year) - int(start_year))
    return (float(end_value) / float(start_value)) ** (1.0 / float(span)) - 1.0


def _share_count_cagr(
    series: list[tuple[int, float]],
    years: int,
    split_rows: list[dict[str, Any]] | None = None,
) -> float | str:
    """Growth rate of the share count over ``years``, split-adjusted.

    A year-over-year move outside [2/3, 3/2], or within 3% of a clean split
    factor, is a break (the shared rule in app.valuation.share_splits). A break
    a filed split ratio (``split_rows``) corroborates is split-adjusted; any
    other break inside the measured window — a 50% raise looks exactly like a
    3-for-2 split — makes the rate UNKNOWN. Before 2026-09-29 every break was
    taken for a split and only the years after it were measured.
    """
    ordered = sorted(
        ((int(year), float(value)) for year, value in series if float(value) > 0),
        key=lambda item: item[0],
    )
    if len(ordered) < 2:
        return UNKNOWN
    target_year = ordered[-1][0] - int(years)
    start = 0
    for index, (year, _value) in enumerate(ordered):
        if year <= target_year:
            start = index
    window, _split_years, breaks = split_adjust_share_series(ordered[start:], split_rows)
    if breaks:
        return UNKNOWN
    return _series_cagr(window, years)


def _series_history(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    for row in rows:
        value = row.get(key)
        year = row.get("year")
        if not _is_num(value) or not isinstance(year, int):
            continue
        history.append({"year": int(year), "value": float(value)})
    return history


def _companyfacts_frame(
    conn,
    *,
    ticker: str,
    as_of_date: str,
    max_years: int = 10,
) -> dict[str, Any]:
    _scope, db_rows = issuer_companyfacts_rows(
        conn,
        ticker,
        columns=(
            "fiscal_year",
            "period_end",
            "period_type",
            "line_item",
            "value",
            "units",
            "source_url",
            "filed_date",
            "form",
            "accession",
        ),
        period_types=ANNUAL_COMPANYFACTS_PERIOD_TYPES,
        as_of_date=as_of_date,
        require_filed_asof=True,
        order_by="fiscal_year ASC, period_end ASC, line_item ASC",
    )
    rows: list[dict[str, Any]] = [
        dict(row) for row in db_rows if str(row["period_end"] or "") <= str(row["filed_date"] or "")
    ]
    rows.extend(
        _companyfacts_rows_from_cache(ticker=ticker, as_of_date=as_of_date, existing_rows=rows)
    )
    if not rows:
        return {}

    by_year: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        fiscal_year = row["fiscal_year"]
        if fiscal_year is None:
            continue
        year = int(fiscal_year)
        metric = str(row["line_item"] or "").strip()
        if not metric:
            continue
        bucket = by_year.setdefault(year, {})
        bucket[metric] = dict(row)

    years = sorted(by_year)
    if max_years > 0:
        years = years[-int(max_years) :]

    frame_rows: list[dict[str, Any]] = []
    row_traces: dict[str, dict[str, dict[str, Any]]] = {}
    for year in years:
        metric_rows = by_year.get(year, {})

        def _value(
            metric: str,
            *,
            year_metric_rows: dict[str, dict[str, Any]] = metric_rows,
        ) -> float | str:
            row = year_metric_rows.get(metric)
            value = row.get("value") if isinstance(row, dict) else None
            return (
                float(value)
                if isinstance(row, dict)
                and _companyfacts_row_has_prompt_lineage(row, metric=metric)
                else UNKNOWN
            )

        revenue = _value("revenue")
        gross_profit = _value("gross_profit")
        operating_income = _value("operating_income")
        net_income = _value("net_income")
        cfo = _value("cfo")
        capex = _value("capex")
        shares_outstanding = _value("shares_outstanding")
        cash = _value("cash")
        total_debt = _value("total_debt")
        r_and_d_total = _value("r_and_d_total")
        share_repurchases_amount = _value("share_repurchases_amount")
        dividends_paid_amount = _value("dividends_paid_amount")
        deposits = _value("deposits")
        loans = _value("loans")
        investment_securities = _value("investment_securities")
        total_assets = _value("total_assets")
        assets_under_management = _value("assets_under_management")
        allowance_for_credit_losses = _value("allowance_for_credit_losses")
        provision_for_credit_losses = _value("provision_for_credit_losses")
        net_charge_offs = _value("net_charge_offs")
        nonaccrual_loans = _value("nonaccrual_loans")
        fcf = float(cfo) - float(capex) if _is_num(cfo) and _is_num(capex) else UNKNOWN
        net_debt = (
            float(total_debt) - float(cash) if _is_num(total_debt) and _is_num(cash) else UNKNOWN
        )

        row_payload = {
            "year": year,
            "revenue": revenue,
            "gross_profit": gross_profit,
            "operating_income": operating_income,
            "net_income": net_income,
            "cfo": cfo,
            "capex": capex,
            "fcf": fcf,
            "shares_outstanding": shares_outstanding,
            "net_debt": net_debt,
            "r_and_d_total": r_and_d_total,
            "share_repurchases_amount": share_repurchases_amount,
            "dividends_paid_amount": dividends_paid_amount,
            "deposits": deposits,
            "loans": loans,
            "investment_securities": investment_securities,
            "total_assets": total_assets,
            "assets_under_management": assets_under_management,
            "allowance_for_credit_losses": allowance_for_credit_losses,
            "provision_for_credit_losses": provision_for_credit_losses,
            "net_charge_offs": net_charge_offs,
            "nonaccrual_loans": nonaccrual_loans,
            "deposits_to_assets": _safe_div(deposits, total_assets),
            "loans_to_deposits": _safe_div(loans, deposits),
            "allowance_to_loans": _safe_div(allowance_for_credit_losses, loans),
            "provision_to_loans": _safe_div(provision_for_credit_losses, loans),
            "net_charge_offs_to_loans": _safe_div(net_charge_offs, loans),
            "debt_to_assets": _safe_div(total_debt, total_assets),
            "gross_margin": _safe_div(gross_profit, revenue),
            "op_margin": _safe_div(operating_income, revenue),
            "fcf_margin": _safe_div(fcf, revenue),
            "cfo_margin": _safe_div(cfo, revenue),
        }
        frame_rows.append(row_payload)

        trace_bucket: dict[str, dict[str, Any]] = {}
        for metric in (
            "revenue",
            "gross_profit",
            "operating_income",
            "net_income",
            "cfo",
            "capex",
            "shares_outstanding",
            "cash",
            "total_debt",
            "r_and_d_total",
            "share_repurchases_amount",
            "dividends_paid_amount",
            "deposits",
            "loans",
            "investment_securities",
            "total_assets",
            "assets_under_management",
            "allowance_for_credit_losses",
            "provision_for_credit_losses",
            "net_charge_offs",
            "nonaccrual_loans",
        ):
            metric_row = metric_rows.get(metric) or {}
            metric_value = metric_row.get("value")
            metric_unit = _literal_companyfacts_unit(
                metric_row,
                metric=metric,
            )
            provenance = (
                _companyfacts_fact_provenance(
                    metric_row,
                    metric=metric,
                    value=float(metric_value),
                    unit=metric_unit,
                )
                if _is_num(metric_value) and metric_unit is not None
                else None
            )
            trace_bucket[metric] = {
                "derived_from": [f"companyfacts_facts.{metric}"],
                "citations": [
                    {
                        "source_url": str(metric_row.get("source_url") or ""),
                        "snippet": f"{metric}: {metric_row.get('value')}",
                        "section_label": "financial_statements",
                    }
                ]
                if metric_row
                else [],
                "provenance": provenance,
            }
        trace_bucket["fcf"] = {
            "derived_from": ["companyfacts_facts.cfo", "companyfacts_facts.capex"],
            "citations": trace_bucket["cfo"]["citations"] or trace_bucket["capex"]["citations"],
            "provenance": (
                _derived_companyfacts_provenance(
                    metric="fcf",
                    value=float(fcf),
                    unit="USD_millions",
                    formula="cfo - capex",
                    inputs={
                        "cfo": metric_rows["cfo"],
                        "capex": metric_rows["capex"],
                    },
                )
                if _is_num(fcf)
                and isinstance(metric_rows.get("cfo"), dict)
                and isinstance(metric_rows.get("capex"), dict)
                else None
            ),
        }
        trace_bucket["net_debt"] = {
            "derived_from": ["companyfacts_facts.total_debt", "companyfacts_facts.cash"],
            "citations": trace_bucket["total_debt"]["citations"]
            or trace_bucket["cash"]["citations"],
            "provenance": (
                _derived_companyfacts_provenance(
                    metric="net_debt",
                    value=float(net_debt),
                    unit="USD_millions",
                    formula="total_debt - cash",
                    inputs={
                        "total_debt": metric_rows["total_debt"],
                        "cash": metric_rows["cash"],
                    },
                )
                if _is_num(net_debt)
                and isinstance(metric_rows.get("total_debt"), dict)
                and isinstance(metric_rows.get("cash"), dict)
                else None
            ),
        }
        trace_bucket["gross_margin"] = {
            "derived_from": ["companyfacts_facts.gross_profit", "companyfacts_facts.revenue"],
            "citations": [],
            "provenance": (
                _derived_companyfacts_provenance(
                    metric="gross_margin",
                    value=float(row_payload["gross_margin"]),
                    unit="ratio",
                    formula="gross_profit / revenue",
                    inputs={
                        "gross_profit": metric_rows["gross_profit"],
                        "revenue": metric_rows["revenue"],
                    },
                )
                if _is_num(row_payload["gross_margin"])
                and isinstance(metric_rows.get("gross_profit"), dict)
                and isinstance(metric_rows.get("revenue"), dict)
                else None
            ),
        }
        trace_bucket["op_margin"] = {
            "derived_from": ["companyfacts_facts.operating_income", "companyfacts_facts.revenue"],
            "citations": [],
            "provenance": (
                _derived_companyfacts_provenance(
                    metric="op_margin",
                    value=float(row_payload["op_margin"]),
                    unit="ratio",
                    formula="operating_income / revenue",
                    inputs={
                        "operating_income": metric_rows["operating_income"],
                        "revenue": metric_rows["revenue"],
                    },
                )
                if _is_num(row_payload["op_margin"])
                and isinstance(metric_rows.get("operating_income"), dict)
                and isinstance(metric_rows.get("revenue"), dict)
                else None
            ),
        }
        trace_bucket["fcf_margin"] = {
            "derived_from": [
                "companyfacts_facts.cfo",
                "companyfacts_facts.capex",
                "companyfacts_facts.revenue",
            ],
            "citations": [],
            "provenance": (
                _derived_companyfacts_provenance(
                    metric="fcf_margin",
                    value=float(row_payload["fcf_margin"]),
                    unit="ratio",
                    formula="(cfo - capex) / revenue",
                    inputs={
                        "cfo": metric_rows["cfo"],
                        "capex": metric_rows["capex"],
                        "revenue": metric_rows["revenue"],
                    },
                )
                if _is_num(row_payload["fcf_margin"])
                and isinstance(metric_rows.get("cfo"), dict)
                and isinstance(metric_rows.get("capex"), dict)
                and isinstance(metric_rows.get("revenue"), dict)
                else None
            ),
        }
        trace_bucket["cfo_margin"] = {
            "derived_from": ["companyfacts_facts.cfo", "companyfacts_facts.revenue"],
            "citations": [],
            "provenance": (
                _derived_companyfacts_provenance(
                    metric="cfo_margin",
                    value=float(row_payload["cfo_margin"]),
                    unit="ratio",
                    formula="cfo / revenue",
                    inputs={
                        "cfo": metric_rows["cfo"],
                        "revenue": metric_rows["revenue"],
                    },
                )
                if _is_num(row_payload["cfo_margin"])
                and isinstance(metric_rows.get("cfo"), dict)
                and isinstance(metric_rows.get("revenue"), dict)
                else None
            ),
        }
        row_traces[str(year)] = trace_bucket

    revenue_series = [
        (int(row["year"]), float(row["revenue"]))
        for row in frame_rows
        if _is_num(row.get("revenue")) and float(row["revenue"]) > 0
    ]
    shares_series = [
        (int(row["year"]), float(row["shares_outstanding"]))
        for row in frame_rows
        if _is_num(row.get("shares_outstanding")) and float(row["shares_outstanding"]) > 0
    ]
    gm_series = [
        (int(row["year"]), float(row["gross_margin"]))
        for row in frame_rows
        if _is_num(row.get("gross_margin"))
    ]
    om_series = [
        (int(row["year"]), float(row["op_margin"]))
        for row in frame_rows
        if _is_num(row.get("op_margin"))
    ]
    fcfm_series = [
        (int(row["year"]), float(row["fcf_margin"]))
        for row in frame_rows
        if _is_num(row.get("fcf_margin"))
    ]

    derived_signals = {
        "revenue_cagr_3y": {
            "value": _series_cagr(revenue_series, 3),
            "derived_from": ["companyfacts_facts.revenue"],
        },
        "revenue_cagr_5y": {
            "value": _series_cagr(revenue_series, 5),
            "derived_from": ["companyfacts_facts.revenue"],
        },
        "revenue_cagr_10y": {
            "value": _series_cagr(revenue_series, 10),
            "derived_from": ["companyfacts_facts.revenue"],
        },
        "gross_margin_trend_slope": {
            "value": _series_slope(gm_series),
            "derived_from": ["companyfacts_facts.gross_profit", "companyfacts_facts.revenue"],
        },
        "operating_margin_trend_slope": {
            "value": _series_slope(om_series),
            "derived_from": ["companyfacts_facts.operating_income", "companyfacts_facts.revenue"],
        },
        "fcf_margin_trend_slope": {
            "value": _series_slope(fcfm_series),
            "derived_from": [
                "companyfacts_facts.cfo",
                "companyfacts_facts.capex",
                "companyfacts_facts.revenue",
            ],
        },
        "dilution_rate_shares_cagr": {
            "value": _share_count_cagr(
                shares_series,
                10,
                split_ratio_rows_from_companyfacts(
                    _raw_companyfacts_from_cache(ticker), as_of_date
                ),
            ),
            "derived_from": ["companyfacts_facts.shares_outstanding"],
        },
    }
    latest = frame_rows[-1] if frame_rows else {}
    if _is_num(latest.get("r_and_d_total")) and _is_num(latest.get("revenue")):
        derived_signals["r_and_d_intensity_latest"] = {
            "value": _safe_div(latest.get("r_and_d_total"), latest.get("revenue")),
            "derived_from": ["companyfacts_facts.r_and_d_total", "companyfacts_facts.revenue"],
        }
    if _is_num(latest.get("deposits")) and _is_num(latest.get("total_assets")):
        derived_signals["deposits_to_assets_latest"] = {
            "value": _safe_div(latest.get("deposits"), latest.get("total_assets")),
            "derived_from": ["companyfacts_facts.deposits", "companyfacts_facts.total_assets"],
        }
    if _is_num(latest.get("loans")) and _is_num(latest.get("deposits")):
        derived_signals["loans_to_deposits_latest"] = {
            "value": _safe_div(latest.get("loans"), latest.get("deposits")),
            "derived_from": ["companyfacts_facts.loans", "companyfacts_facts.deposits"],
        }
    if _is_num(latest.get("allowance_for_credit_losses")) and _is_num(latest.get("loans")):
        derived_signals["allowance_to_loans_latest"] = {
            "value": _safe_div(latest.get("allowance_for_credit_losses"), latest.get("loans")),
            "derived_from": [
                "companyfacts_facts.allowance_for_credit_losses",
                "companyfacts_facts.loans",
            ],
        }
    if _is_num(latest.get("provision_for_credit_losses")) and _is_num(latest.get("loans")):
        derived_signals["provision_to_loans_latest"] = {
            "value": _safe_div(latest.get("provision_for_credit_losses"), latest.get("loans")),
            "derived_from": [
                "companyfacts_facts.provision_for_credit_losses",
                "companyfacts_facts.loans",
            ],
        }
    if _is_num(latest.get("net_charge_offs")) and _is_num(latest.get("loans")):
        derived_signals["net_charge_offs_to_loans_latest"] = {
            "value": _safe_div(latest.get("net_charge_offs"), latest.get("loans")),
            "derived_from": ["companyfacts_facts.net_charge_offs", "companyfacts_facts.loans"],
        }
    if _is_num(latest.get("total_debt")) and _is_num(latest.get("total_assets")):
        derived_signals["debt_to_assets_latest"] = {
            "value": _safe_div(latest.get("total_debt"), latest.get("total_assets")),
            "derived_from": ["companyfacts_facts.total_debt", "companyfacts_facts.total_assets"],
        }
    signal_formulas = {
        "revenue_cagr_3y": "CAGR(revenue, 3 years)",
        "revenue_cagr_5y": "CAGR(revenue, 5 years)",
        "revenue_cagr_10y": "CAGR(revenue, 10 years)",
        "gross_margin_trend_slope": "linear_slope(gross_profit / revenue)",
        "operating_margin_trend_slope": "linear_slope(operating_income / revenue)",
        "fcf_margin_trend_slope": "linear_slope((cfo - capex) / revenue)",
        "dilution_rate_shares_cagr": "CAGR(shares_outstanding, 10 years, split-adjusted by filed split ratios; UNKNOWN across an uncorroborated break)",
        "r_and_d_intensity_latest": "r_and_d_total / revenue",
        "deposits_to_assets_latest": "deposits / total_assets",
        "loans_to_deposits_latest": "loans / deposits",
        "allowance_to_loans_latest": "allowance_for_credit_losses / loans",
        "provision_to_loans_latest": "provision_for_credit_losses / loans",
        "net_charge_offs_to_loans_latest": "net_charge_offs / loans",
        "debt_to_assets_latest": "total_debt / total_assets",
    }
    for signal_name, signal_payload in derived_signals.items():
        signal_value = signal_payload.get("value")
        if not _is_num(signal_value):
            continue
        metric_names = [
            str(reference).rsplit(".", maxsplit=1)[-1]
            for reference in signal_payload.get("derived_from") or []
            if str(reference).startswith("companyfacts_facts.")
        ]
        selected_years = years[-1:] if signal_name.endswith("_latest") else years
        input_rows = {
            f"{metric}_{year}": by_year[year][metric]
            for year in selected_years
            for metric in metric_names
            if isinstance(by_year.get(year, {}).get(metric), dict)
        }
        if not input_rows:
            continue
        signal_payload["provenance"] = _derived_companyfacts_provenance(
            metric=signal_name,
            value=float(signal_value),
            unit="ratio",
            formula=signal_formulas[signal_name],
            inputs=input_rows,
        )
    for signal_name, row_key, refs in (
        (
            "allowance_to_loans_history",
            "allowance_to_loans",
            ["companyfacts_facts.allowance_for_credit_losses", "companyfacts_facts.loans"],
        ),
        (
            "provision_to_loans_history",
            "provision_to_loans",
            ["companyfacts_facts.provision_for_credit_losses", "companyfacts_facts.loans"],
        ),
        (
            "net_charge_offs_to_loans_history",
            "net_charge_offs_to_loans",
            ["companyfacts_facts.net_charge_offs", "companyfacts_facts.loans"],
        ),
    ):
        history = _series_history(frame_rows, row_key)
        if history:
            derived_signals[signal_name] = {
                "value": history,
                "derived_from": refs,
            }

    gaps: list[dict[str, Any]] = []
    for field in (
        "revenue",
        "gross_profit",
        "operating_income",
        "net_income",
        "cfo",
        "capex",
        "fcf",
        "shares_outstanding",
        "net_debt",
    ):
        missing = len([row for row in frame_rows if not _is_num(row.get(field))])
        if missing > 0:
            gaps.append(
                {
                    "field": field,
                    "missing_count": int(missing),
                    "derived_from": [f"companyfacts_facts.{field}"],
                }
            )

    return {
        "rows": frame_rows,
        "row_traces": row_traces,
        "derived_signals": derived_signals,
        "gaps": gaps,
    }


def _raw_companyfacts_from_cache(ticker: str) -> dict[str, Any]:
    """The issuer's cached SEC companyfacts payload, or {} when there is none."""
    cik = resolve_cik_for_ticker(ticker, refresh_if_missing=False)
    if not cik:
        return {}
    payload = _safe_json(companyfacts_cache_path(cik))
    raw = payload.get("companyfacts") if isinstance(payload.get("companyfacts"), dict) else payload
    return raw if isinstance(raw, dict) else {}


def _companyfacts_rows_from_cache(
    *,
    ticker: str,
    as_of_date: str,
    existing_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cik = resolve_cik_for_ticker(ticker, refresh_if_missing=False)
    if not cik:
        return []
    raw = _raw_companyfacts_from_cache(ticker)
    if not raw:
        return []
    try:
        normalized = normalize_annual_facts_from_raw(
            raw,
            cik=str(cik).zfill(10),
            years_back=10,
            filed_as_of=as_of_date,
        )
    except Exception:
        return []
    existing_keys = {
        (int(row.get("fiscal_year") or 0), str(row.get("line_item") or ""))
        for row in existing_rows
        if int(row.get("fiscal_year") or 0) > 0
        and str(row.get("line_item") or "").strip()
        and _companyfacts_row_has_prompt_lineage(
            row,
            metric=str(row.get("line_item") or ""),
        )
    }
    supplemented: list[dict[str, Any]] = []
    for row in normalized:
        period_end = str(row.get("period_end") or "")
        fiscal_year = int(row.get("fiscal_year") or 0)
        line_item = str(row.get("line_item") or "")
        key = (fiscal_year, line_item)
        filed_date = str(row.get("filed_date") or "")
        if (
            not period_end
            or period_end > as_of_date
            or not filed_date
            or filed_date > as_of_date
            or period_end > filed_date
            or key in existing_keys
        ):
            continue
        supplemented.append(dict(row))
    return supplemented


def _overlay_companyfacts_trends(
    conn,
    *,
    ticker: str,
    as_of_date: str,
    fundamentals: dict[str, Any],
) -> None:
    existing_rows = [row for row in (fundamentals.get("rows") or []) if isinstance(row, dict)]
    existing_signals = fundamentals.get("derived_signals")

    frame = _companyfacts_frame(conn, ticker=ticker, as_of_date=as_of_date)
    if not frame:
        return

    if not existing_rows:
        fundamentals["rows"] = frame.get("rows") or []
    if not isinstance(fundamentals.get("row_traces"), dict) or not fundamentals.get("row_traces"):
        fundamentals["row_traces"] = frame.get("row_traces") or {}
    else:
        frame_traces = frame.get("row_traces") or {}
        rows_by_year = {
            str(row.get("year") or ""): row for row in existing_rows if str(row.get("year") or "")
        }
        for year, traces in fundamentals["row_traces"].items():
            if not isinstance(traces, dict):
                continue
            frame_year_traces = frame_traces.get(str(year)) or {}
            for metric, trace in traces.items():
                frame_trace = frame_year_traces.get(metric)
                row_value = (rows_by_year.get(str(year)) or {}).get(metric)
                provenance = (
                    frame_trace.get("provenance") if isinstance(frame_trace, dict) else None
                )
                if (
                    isinstance(trace, dict)
                    and _is_num(row_value)
                    and isinstance(provenance, dict)
                    and _is_num(provenance.get("value"))
                    and math.isclose(
                        float(row_value),
                        float(provenance.get("value")),
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                ):
                    trace["provenance"] = dict(provenance)
    if not isinstance(existing_signals, dict) or not existing_signals:
        fundamentals["derived_signals"] = frame.get("derived_signals") or {}
    if not isinstance(fundamentals.get("gaps"), list) or not fundamentals.get("gaps"):
        fundamentals["gaps"] = frame.get("gaps") or []
    latest = existing_rows[-1] if existing_rows else {}
    if not latest:
        frame_rows = [row for row in (frame.get("rows") or []) if isinstance(row, dict)]
        latest = frame_rows[-1] if frame_rows else {}
    latest_year = str(latest.get("year") or "")
    latest_trace_bucket = (
        (frame.get("row_traces") or {}).get(latest_year, {}) if latest_year else {}
    )
    fundamentals_provenance = fundamentals.setdefault(
        _PROMPT_FINANCIAL_PROVENANCE_KEY,
        {},
    )
    for key in (
        "revenue",
        "gross_profit",
        "operating_income",
        "net_income",
        "cfo",
        "capex",
        "fcf",
        "shares_outstanding",
        "net_debt",
        "r_and_d_total",
        "share_repurchases_amount",
        "dividends_paid_amount",
        "deposits",
        "loans",
        "investment_securities",
        "total_assets",
        "assets_under_management",
        "allowance_for_credit_losses",
        "provision_for_credit_losses",
        "net_charge_offs",
        "nonaccrual_loans",
        "deposits_to_assets",
        "loans_to_deposits",
        "allowance_to_loans",
        "provision_to_loans",
        "net_charge_offs_to_loans",
        "debt_to_assets",
        "gross_margin",
        "op_margin",
        "fcf_margin",
        "cfo_margin",
    ):
        # A key the filing route stored as UNKNOWN is as absent as a missing one: the
        # companyfacts value fills both, or business quality stays the neutral score.
        # (Free cash flow stays UNKNOWN where the filing route marked it sector-limited.)
        fcf_not_applicable = (
            key in {"fcf", "fcf_margin"} and fundamentals.get("fcf_applicability") == "sector_limited"
        )
        if (
            (fundamentals.get(key, UNKNOWN) in (UNKNOWN, None))
            and key in latest
            and not fcf_not_applicable
        ):
            fundamentals[key] = latest.get(key, UNKNOWN)
        trace = latest_trace_bucket.get(key) if isinstance(latest_trace_bucket, dict) else None
        if (
            _is_num(fundamentals.get(key))
            and isinstance(trace, dict)
            and isinstance(trace.get("provenance"), dict)
        ):
            fundamentals_provenance[key] = dict(trace["provenance"])
    if fundamentals.get("operating_margin", UNKNOWN) in (UNKNOWN, None):
        fundamentals["operating_margin"] = fundamentals.get("op_margin", UNKNOWN)
    if _is_num(fundamentals.get("operating_margin")) and isinstance(
        fundamentals_provenance.get("op_margin"), dict
    ):
        fundamentals_provenance["operating_margin"] = dict(fundamentals_provenance["op_margin"])

    for signal_name, signal_payload in (fundamentals.get("derived_signals") or {}).items():
        if isinstance(signal_payload, dict) and signal_name not in fundamentals:
            fundamentals[signal_name] = signal_payload.get("value", UNKNOWN)
        frame_signal = (frame.get("derived_signals") or {}).get(signal_name)
        if (
            isinstance(signal_payload, dict)
            and _is_num(signal_payload.get("value"))
            and isinstance(frame_signal, dict)
            and _is_num(frame_signal.get("value"))
            and math.isclose(
                float(signal_payload["value"]),
                float(frame_signal["value"]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            and isinstance(frame_signal.get("provenance"), dict)
        ):
            signal_payload["provenance"] = dict(frame_signal["provenance"])
        if (
            isinstance(signal_payload, dict)
            and _is_num(fundamentals.get(signal_name))
            and isinstance(signal_payload.get("provenance"), dict)
        ):
            fundamentals_provenance[signal_name] = dict(signal_payload["provenance"])


def _has_fact(rows: list[dict[str, Any]], fact_type: str) -> bool:
    return any(
        str(row.get("fact_type") or "") == fact_type for row in rows if isinstance(row, dict)
    )


def _has_financial(rows: list[dict[str, Any]], line_item: str) -> bool:
    return any(
        str(row.get("line_item") or "") == line_item for row in rows if isinstance(row, dict)
    )


def _append_supplemental_fact(
    rows: list[dict[str, Any]],
    *,
    fact_type: str,
    metric: str,
    value: float,
    source_url: str,
    snippet: str,
    section_label: str,
    derived_from: list[str],
    extra_value_fields: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
) -> None:
    if _has_fact(rows, fact_type):
        return
    value_payload = {
        "year": None,
        "metric": metric,
        "value": float(value),
        "derived_from": [str(ref) for ref in derived_from if str(ref).strip()],
        "citations": [
            {
                "source_url": source_url,
                "snippet": snippet,
                "section_label": section_label,
            }
        ]
        if source_url or snippet
        else [],
    }
    if isinstance(extra_value_fields, dict):
        value_payload.update(extra_value_fields)
    record = ExtractedFactRecord(
        fact_type=fact_type,
        value=value_payload,
        citation=Citation(
            source_url=source_url,
            snippet=snippet,
            section_label=section_label,
        ),
    )
    payload = record.model_dump()
    if isinstance(provenance, dict):
        payload["provenance"] = dict(provenance)
    rows.append(payload)


def _append_supplemental_financial(
    rows: list[dict[str, Any]],
    *,
    line_item: str,
    value: float,
    units: str,
    period: str,
    source_url: str,
    snippet: str,
    section_label: str,
    statement_type: str,
    provenance: dict[str, Any] | None = None,
) -> None:
    if _has_financial(rows, line_item):
        return
    record = FinancialRecord(
        statement_type=statement_type,
        line_item=line_item,
        value=float(value),
        units=units,
        period=period,
        citation=Citation(
            source_url=source_url,
            snippet=snippet,
            section_label=section_label,
        ),
    )
    payload = record.model_dump()
    if isinstance(provenance, dict):
        payload["provenance"] = dict(provenance)
    rows.append(payload)


def _latest_companyfacts_rows(
    conn,
    *,
    ticker: str,
    as_of_date: str,
) -> dict[str, dict[str, Any]]:
    _scope, candidate_rows = issuer_companyfacts_rows(
        conn,
        ticker,
        columns=(
            "fiscal_year",
            "period_end",
            "period_type",
            "line_item",
            "value",
            "units",
            "source_url",
            "filed_date",
            "form",
            "accession",
        ),
        period_types=ANNUAL_COMPANYFACTS_PERIOD_TYPES,
        as_of_date=as_of_date,
        require_filed_asof=True,
        order_by="period_end DESC, filed_date DESC, line_item ASC",
    )
    valid_rows = [
        dict(row)
        for row in candidate_rows
        if str(row["period_end"] or "") <= str(row["filed_date"] or "")
    ]
    period_end = max(
        (str(row.get("period_end") or "").strip() for row in valid_rows),
        default="",
    )
    if not period_end:
        return {}
    return {
        str(row["line_item"]): row
        for row in valid_rows
        if str(row.get("period_end") or "") == period_end
        and str(row.get("line_item") or "").strip()
        and _companyfacts_row_has_prompt_lineage(
            row,
            metric=str(row.get("line_item") or ""),
        )
    }


def _overlay_companyfacts_support(
    conn,
    *,
    ticker: str,
    as_of_date: str,
    fundamentals: dict[str, Any],
    extracted_facts: list[dict[str, Any]],
    financials: list[dict[str, Any]],
) -> None:
    companyfacts_rows = _latest_companyfacts_rows(conn, ticker=ticker, as_of_date=as_of_date)
    if not companyfacts_rows:
        return
    fundamentals_provenance = fundamentals.setdefault(
        _PROMPT_FINANCIAL_PROVENANCE_KEY,
        {},
    )

    for metric in (
        "revenue",
        "net_income",
        "cfo",
        "capex",
        "cash",
        "total_debt",
        "shares_outstanding",
        "deposits",
        "loans",
        "investment_securities",
        "total_assets",
        "assets_under_management",
        "allowance_for_credit_losses",
        "provision_for_credit_losses",
        "net_charge_offs",
        "nonaccrual_loans",
        "r_and_d_total",
        "share_repurchases_amount",
        "dividends_paid_amount",
    ):
        row = companyfacts_rows.get(metric)
        value = row.get("value") if isinstance(row, dict) else None
        if not _is_num(value):
            continue
        numeric = float(value)
        source_url = str(row.get("source_url") or "")
        units = _literal_companyfacts_unit(row, metric=metric)
        if units is None:
            continue
        period = str(row.get("period_end") or as_of_date)
        snippet = f"{metric}: {numeric}"
        provenance = _companyfacts_fact_provenance(
            row,
            metric=metric,
            value=numeric,
            unit=units,
        )
        if provenance is None:
            continue
        fundamentals_provenance[metric] = provenance
        _append_supplemental_fact(
            extracted_facts,
            fact_type=metric,
            metric=metric,
            value=numeric,
            source_url=source_url,
            snippet=snippet,
            section_label="financial_statements",
            derived_from=[f"companyfacts_facts.{metric}"],
            provenance=provenance,
        )
        _append_supplemental_financial(
            financials,
            line_item=metric,
            value=numeric,
            units=units,
            period=period,
            source_url=source_url,
            snippet=snippet,
            section_label="financial_statements",
            statement_type="companyfacts_fallback",
            provenance=provenance,
        )

    latest_loans = fundamentals.get("loans", UNKNOWN)
    latest_allowance = fundamentals.get("allowance_for_credit_losses", UNKNOWN)
    latest_provision = fundamentals.get("provision_for_credit_losses", UNKNOWN)
    latest_charge_offs = fundamentals.get("net_charge_offs", UNKNOWN)
    if (
        "provision_to_loans_latest" not in fundamentals
        and _is_num(latest_provision)
        and _is_num(latest_loans)
        and isinstance(companyfacts_rows.get("provision_for_credit_losses"), dict)
        and isinstance(companyfacts_rows.get("loans"), dict)
    ):
        fundamentals["provision_to_loans_latest"] = _safe_div(latest_provision, latest_loans)
        fundamentals_provenance["provision_to_loans_latest"] = _derived_companyfacts_provenance(
            metric="provision_to_loans_latest",
            value=float(fundamentals["provision_to_loans_latest"]),
            unit="ratio",
            formula="provision_for_credit_losses / loans",
            inputs={
                "provision_for_credit_losses": companyfacts_rows["provision_for_credit_losses"],
                "loans": companyfacts_rows["loans"],
            },
        )
    if (
        "net_charge_offs_to_loans_latest" not in fundamentals
        and _is_num(latest_charge_offs)
        and _is_num(latest_loans)
        and isinstance(companyfacts_rows.get("net_charge_offs"), dict)
        and isinstance(companyfacts_rows.get("loans"), dict)
    ):
        fundamentals["net_charge_offs_to_loans_latest"] = _safe_div(
            latest_charge_offs, latest_loans
        )
        fundamentals_provenance["net_charge_offs_to_loans_latest"] = (
            _derived_companyfacts_provenance(
                metric="net_charge_offs_to_loans_latest",
                value=float(fundamentals["net_charge_offs_to_loans_latest"]),
                unit="ratio",
                formula="net_charge_offs / loans",
                inputs={
                    "net_charge_offs": companyfacts_rows["net_charge_offs"],
                    "loans": companyfacts_rows["loans"],
                },
            )
        )
    if (
        "allowance_to_loans_latest" not in fundamentals
        and _is_num(latest_allowance)
        and _is_num(latest_loans)
        and isinstance(companyfacts_rows.get("allowance_for_credit_losses"), dict)
        and isinstance(companyfacts_rows.get("loans"), dict)
    ):
        fundamentals["allowance_to_loans_latest"] = _safe_div(latest_allowance, latest_loans)
        fundamentals_provenance["allowance_to_loans_latest"] = _derived_companyfacts_provenance(
            metric="allowance_to_loans_latest",
            value=float(fundamentals["allowance_to_loans_latest"]),
            unit="ratio",
            formula="allowance_for_credit_losses / loans",
            inputs={
                "allowance_for_credit_losses": companyfacts_rows["allowance_for_credit_losses"],
                "loans": companyfacts_rows["loans"],
            },
        )

    cfo_row = companyfacts_rows.get("cfo")
    capex_row = companyfacts_rows.get("capex")
    if (
        not _has_fact(extracted_facts, "fcf")
        and _is_num(fundamentals.get("fcf", UNKNOWN))
        and isinstance(cfo_row, dict)
        and isinstance(capex_row, dict)
        and _is_num(cfo_row.get("value"))
        and _is_num(capex_row.get("value"))
    ):
        source_url = str(cfo_row.get("source_url") or capex_row.get("source_url") or "")
        period = str(cfo_row.get("period_end") or capex_row.get("period_end") or as_of_date)
        snippet = (
            f"Derived FCF from CFO {float(cfo_row['value'])} and capex {float(capex_row['value'])}"
        )
        provenance = _derived_companyfacts_provenance(
            metric="fcf",
            value=float(fundamentals["fcf"]),
            unit="USD_millions",
            formula="cfo - capex",
            inputs={"cfo": cfo_row, "capex": capex_row},
        )
        if provenance is not None:
            fundamentals_provenance["fcf"] = provenance
            _append_supplemental_fact(
                extracted_facts,
                fact_type="fcf",
                metric="fcf",
                value=float(fundamentals["fcf"]),
                source_url=source_url,
                snippet=snippet,
                section_label="financial_statements",
                derived_from=[
                    "companyfacts_facts.cfo",
                    "companyfacts_facts.capex",
                ],
                provenance=provenance,
            )
            _append_supplemental_financial(
                financials,
                line_item="fcf",
                value=float(fundamentals["fcf"]),
                units="USD_millions",
                period=period,
                source_url=source_url,
                snippet=snippet,
                section_label="financial_statements",
                statement_type="derived_fallback",
                provenance=provenance,
            )

    debt_row = companyfacts_rows.get("total_debt")
    cash_row = companyfacts_rows.get("cash")
    if (
        not _has_fact(extracted_facts, "net_debt")
        and _is_num(fundamentals.get("net_debt", UNKNOWN))
        and isinstance(debt_row, dict)
        and isinstance(cash_row, dict)
        and _is_num(debt_row.get("value"))
        and _is_num(cash_row.get("value"))
    ):
        source_url = str(debt_row.get("source_url") or cash_row.get("source_url") or "")
        period = str(debt_row.get("period_end") or cash_row.get("period_end") or as_of_date)
        snippet = f"Derived net debt from total_debt {float(debt_row['value'])} and cash {float(cash_row['value'])}"
        provenance = _derived_companyfacts_provenance(
            metric="net_debt",
            value=float(fundamentals["net_debt"]),
            unit="USD_millions",
            formula="total_debt - cash",
            inputs={"total_debt": debt_row, "cash": cash_row},
        )
        if provenance is not None:
            fundamentals_provenance["net_debt"] = provenance
            _append_supplemental_fact(
                extracted_facts,
                fact_type="net_debt",
                metric="net_debt",
                value=float(fundamentals["net_debt"]),
                source_url=source_url,
                snippet=snippet,
                section_label="financial_statements",
                derived_from=[
                    "companyfacts_facts.total_debt",
                    "companyfacts_facts.cash",
                ],
                provenance=provenance,
            )
            _append_supplemental_financial(
                financials,
                line_item="net_debt",
                value=float(fundamentals["net_debt"]),
                units="USD_millions",
                period=period,
                source_url=source_url,
                snippet=snippet,
                section_label="financial_statements",
                statement_type="derived_fallback",
                provenance=provenance,
            )


def _latest_numeric_financial(
    financials: list[dict[str, Any]], line_item: str
) -> dict[str, Any] | None:
    for row in reversed(financials):
        if str(row.get("line_item") or "") != line_item:
            continue
        value = row.get("value")
        if _is_num(value):
            return row
    return None


def _overlay_bank_credit_filing_fallbacks(
    *,
    fundamentals: dict[str, Any],
    financials: list[dict[str, Any]],
) -> None:
    issuer_classification = str(fundamentals.get("issuer_classification") or "").strip().lower()
    if issuer_classification != "financial":
        return

    for metric in (
        "loans",
        "allowance_for_credit_losses",
        "provision_for_credit_losses",
        "net_charge_offs",
        "nonaccrual_loans",
    ):
        if _is_num(fundamentals.get(metric, UNKNOWN)):
            continue
        row = _latest_numeric_financial(financials, metric)
        if row is not None:
            fundamentals[metric] = float(row["value"])

    loans = fundamentals.get("loans", UNKNOWN)
    allowance = fundamentals.get("allowance_for_credit_losses", UNKNOWN)
    provision = fundamentals.get("provision_for_credit_losses", UNKNOWN)
    charge_offs = fundamentals.get("net_charge_offs", UNKNOWN)
    if (
        not _is_num(fundamentals.get("allowance_to_loans_latest"))
        and _is_num(allowance)
        and _is_num(loans)
    ):
        fundamentals["allowance_to_loans_latest"] = _safe_div(allowance, loans)
    if (
        not _is_num(fundamentals.get("provision_to_loans_latest"))
        and _is_num(provision)
        and _is_num(loans)
    ):
        fundamentals["provision_to_loans_latest"] = _safe_div(provision, loans)
    if (
        not _is_num(fundamentals.get("net_charge_offs_to_loans_latest"))
        and _is_num(charge_offs)
        and _is_num(loans)
    ):
        fundamentals["net_charge_offs_to_loans_latest"] = _safe_div(charge_offs, loans)


def _resolve_local_facts_backfill(*, ticker: str, as_of_date: str) -> dict[str, Any]:
    from app.valuation.net_debt import resolve_net_debt_proxy

    cik = resolve_cik_for_ticker(ticker, refresh_if_missing=False)
    if not cik:
        return {}
    cache_path = companyfacts_cache_path(cik)
    payload = _safe_json(cache_path)
    companyfacts = (
        payload.get("companyfacts") if isinstance(payload.get("companyfacts"), dict) else payload
    )
    if not isinstance(companyfacts, dict) or not isinstance(companyfacts.get("facts"), dict):
        return {}

    source_url = str(
        payload.get("source_url") or f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    )
    extracted = extract_company_facts_asof(companyfacts, as_of_date)
    out: dict[str, Any] = {
        "source_url": source_url,
        "derived_from": [str(cache_path)],
        "metric_support": {},
    }
    for metric, key in (
        ("cfo", "cfo_asof"),
        ("capex", "capex_asof"),
        ("fcf", "fcf_asof"),
    ):
        fact = extracted.get(key) if isinstance(extracted, dict) else None
        support = {
            "status": "OK" if isinstance(fact, dict) and _is_num(fact.get("value")) else "UNKNOWN",
            "resolution": str(
                fact.get("resolution")
                or (
                    "RESOLVED"
                    if isinstance(fact, dict) and _is_num(fact.get("value"))
                    else "UNAVAILABLE"
                )
            )
            if isinstance(fact, dict)
            else "UNAVAILABLE",
            "reason_code": str(
                fact.get("reason_code")
                or (
                    "OK"
                    if isinstance(fact, dict) and _is_num(fact.get("value"))
                    else f"MISSING_{metric.upper()}"
                )
            )
            if isinstance(fact, dict)
            else f"MISSING_{metric.upper()}",
            "tag": str(fact.get("tag") or "") if isinstance(fact, dict) else "",
            "as_of_used": str(fact.get("fact_end_date") or "") if isinstance(fact, dict) else "",
            "derived_from": [
                str(ref) for ref in (fact.get("derived_from") or []) if str(ref).strip()
            ]
            if isinstance(fact, dict)
            else [],
        }
        if isinstance(fact, dict) and isinstance(fact.get("bridge_context"), dict):
            support["bridge_context"] = dict(fact["bridge_context"])
        out["metric_support"][metric] = support
        if isinstance(fact, dict) and _is_num(fact.get("value")):
            out[f"{metric}_value"] = float(fact["value"])
            out["derived_from"] = [
                str(ref)
                for ref in ([str(cache_path)] + list(fact.get("derived_from") or []))
                if str(ref).strip()
            ]

    net_debt_row = resolve_net_debt_proxy(
        ticker=ticker,
        as_of_date=as_of_date,
        facts_row={
            "ticker": ticker.upper(),
            "cache_path": str(cache_path),
            "derived_from": [str(cache_path)],
        },
    )
    out["metric_support"]["net_debt"] = {
        "status": str(net_debt_row.get("status") or "UNKNOWN"),
        "resolution": str(net_debt_row.get("net_debt_resolution") or "UNAVAILABLE"),
        "reason_code": str(net_debt_row.get("reason_code") or "UNKNOWN"),
        "issuer_classification": str(net_debt_row.get("issuer_classification") or ""),
        "derived_from": [
            str(ref) for ref in (net_debt_row.get("derived_from") or []) if str(ref).strip()
        ],
        "components": {
            "total_debt": net_debt_row.get("total_debt"),
            "cash_equivalents": net_debt_row.get("cash_equivalents"),
        },
    }
    if _is_num(net_debt_row.get("net_debt_proxy")):
        out["net_debt_proxy"] = float(net_debt_row["net_debt_proxy"])
        out["net_debt_derived_from"] = [
            str(ref) for ref in (net_debt_row.get("derived_from") or []) if str(ref).strip()
        ]
    return out


def _overlay_missing_fundamentals(
    *,
    ticker: str,
    as_of_date: str,
    fundamentals: dict[str, Any],
    extracted_facts: list[dict[str, Any]],
    financials: list[dict[str, Any]],
) -> None:
    missing_fact_metrics = [
        metric
        for metric in ("cfo", "capex", "fcf")
        if not _is_num(fundamentals.get(metric, UNKNOWN))
    ]
    needs_net_debt = not _is_num(fundamentals.get("net_debt", UNKNOWN))
    if not missing_fact_metrics and not needs_net_debt:
        return

    facts_row = _resolve_local_facts_backfill(
        ticker=ticker,
        as_of_date=as_of_date,
    )
    source_url = str(facts_row.get("source_url") or "")
    facts_refs = [str(ref) for ref in (facts_row.get("derived_from") or []) if str(ref).strip()]
    resolution_bucket = fundamentals.setdefault("upstream_evidence_resolution", {})
    if isinstance(facts_row.get("metric_support"), dict):
        resolution_bucket.update(facts_row["metric_support"])
    for metric in missing_fact_metrics:
        current = fundamentals.get(metric, UNKNOWN)
        value = facts_row.get(f"{metric}_value", UNKNOWN)
        if _is_num(current) or not _is_num(value):
            continue
        numeric = float(value)
        fundamentals[metric] = numeric
        _append_supplemental_fact(
            extracted_facts,
            fact_type=metric,
            metric=metric,
            value=numeric,
            source_url=source_url,
            snippet=str(
                (
                    facts_row.get("metric_support", {})
                    .get(metric, {})
                    .get("bridge_context", {})
                    .get("formula")
                )
                or ""
            ),
            section_label="financial_statements",
            derived_from=facts_refs,
            extra_value_fields={
                "resolution": str(
                    (facts_row.get("metric_support", {}).get(metric, {}) or {}).get("resolution")
                    or "RESOLVED"
                ),
                "reason_code": str(
                    (facts_row.get("metric_support", {}).get(metric, {}) or {}).get("reason_code")
                    or "OK"
                ),
                "bridge_context": (facts_row.get("metric_support", {}).get(metric, {}) or {}).get(
                    "bridge_context"
                ),
            },
        )
        _append_supplemental_financial(
            financials,
            line_item=metric,
            value=numeric,
            units="USD_millions",
            period=as_of_date,
            source_url=source_url,
            snippet="",
            section_label="financial_statements",
            statement_type="companyfacts_fallback",
        )

    if needs_net_debt:
        net_debt_value = facts_row.get("net_debt_proxy", UNKNOWN)
        if not _is_num(fundamentals.get("net_debt", UNKNOWN)) and _is_num(net_debt_value):
            numeric = float(net_debt_value)
            fundamentals["net_debt"] = numeric
            net_debt_refs = [
                str(ref)
                for ref in (facts_row.get("net_debt_derived_from") or facts_refs)
                if str(ref).strip()
            ]
            net_debt_support = (
                (facts_row.get("metric_support", {}).get("net_debt") or {})
                if isinstance(facts_row.get("metric_support"), dict)
                else {}
            )
            debt_value = ((net_debt_support.get("components") or {}).get("total_debt") or {}).get(
                "value"
            )
            cash_value = (
                (net_debt_support.get("components") or {}).get("cash_equivalents") or {}
            ).get("value")
            snippet = ""
            if _is_num(debt_value) and _is_num(cash_value):
                snippet = f"Derived net debt from total_debt {float(debt_value)} and cash {float(cash_value)}"
            _append_supplemental_fact(
                extracted_facts,
                fact_type="net_debt",
                metric="net_debt",
                value=numeric,
                source_url=source_url,
                snippet=snippet,
                section_label="financial_statements",
                derived_from=net_debt_refs,
                extra_value_fields={
                    "resolution": str(net_debt_support.get("resolution") or "DERIVED"),
                    "reason_code": str(net_debt_support.get("reason_code") or "OK"),
                    "components": net_debt_support.get("components"),
                },
            )
            _append_supplemental_financial(
                financials,
                line_item="net_debt",
                value=numeric,
                units="USD_millions",
                period=as_of_date,
                source_url=source_url,
                snippet=snippet,
                section_label="financial_statements",
                statement_type="derived_fallback",
            )


def _latest_price_quote(conn, *, ticker: str, as_of_date: str) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT provider, as_of_date, price, currency, source_url, fetched_at
        FROM price_quotes
        WHERE ticker = ?
          AND status = 'OK'
          AND price IS NOT NULL
          AND provider != 'disabled'
          AND as_of_date <= ?
        ORDER BY as_of_date DESC,
                 CASE provider
                     WHEN 'stooq' THEN 0
                     ELSE 1
                 END,
                 fetched_at DESC
        LIMIT 1
        """,
        (ticker, as_of_date),
    ).fetchone()
    return dict(row) if row else None


def _latest_disk_cache_price_quote(*, ticker: str, as_of_date: str) -> dict[str, Any] | None:
    cache_path = get_config().cache_dir / "prices" / f"{ticker.upper()}.json"
    payload = _safe_json(cache_path)
    entries = payload.get("entries") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        return None
    candidates: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        snapshot = entry.get("snapshot") if isinstance(entry.get("snapshot"), dict) else None
        if not isinstance(snapshot, dict):
            continue
        price = snapshot.get("price")
        snapshot_as_of = str(snapshot.get("as_of_date") or "").strip()
        if not _is_num(price) or not snapshot_as_of or snapshot_as_of > as_of_date:
            continue
        candidates.append(
            {
                "provider": str(snapshot.get("source") or entry.get("source") or "disk_cache"),
                "as_of_date": snapshot_as_of,
                "price": float(price),
                "currency": str(snapshot.get("currency") or "USD"),
                "source_url": str(snapshot.get("url") or ""),
                "fetched_at": str(snapshot.get("retrieved_at") or ""),
                "requested_as_of_date": str(entry.get("requested_as_of_date") or ""),
            }
        )
    if not candidates:
        return None
    candidates.sort(
        key=lambda row: (
            str(row.get("as_of_date") or ""),
            1 if str(row.get("provider") or "") == "stooq" else 0,
            str(row.get("requested_as_of_date") or ""),
            str(row.get("fetched_at") or ""),
        ),
        reverse=True,
    )
    return candidates[0]


def _overlay_missing_price_inputs(
    conn,
    *,
    ticker: str,
    as_of_date: str,
    valuations: dict[str, Any],
) -> None:
    quote = _latest_price_quote(conn, ticker=ticker, as_of_date=as_of_date)
    if quote is None:
        quote = _latest_disk_cache_price_quote(ticker=ticker, as_of_date=as_of_date)
    if quote is None:
        historical = resolve_price_from_historical_runs(ticker=ticker, as_of_date=as_of_date)
        if historical is not None:
            quote = {
                "provider": str(historical.source or "historical_run_artifacts"),
                "as_of_date": historical.as_of_date,
                "price": float(historical.price),
                "currency": historical.currency,
                "source_url": str(historical.url or ""),
                "fetched_at": str(historical.retrieved_at or ""),
            }
    if not quote or not _is_num(quote.get("price")):
        return
    price = float(quote["price"])
    provider = str(quote.get("provider") or "")
    price_as_of = str(quote.get("as_of_date") or as_of_date)
    source_url = str(quote.get("source_url") or "")
    fetched_at = str(quote.get("fetched_at") or "")
    for method_payload in valuations.values():
        if not isinstance(method_payload, dict):
            continue
        inputs = method_payload.get("inputs")
        if not isinstance(inputs, dict):
            continue
        price_missing = not _is_num(inputs.get("price"))
        market_price_missing = not _is_num(inputs.get("market_price"))
        metadata_stale = (
            price_missing
            or market_price_missing
            or not str(inputs.get("price_status") or "").strip()
            or str(inputs.get("price_status")).upper() == "UNKNOWN"
            or not str(inputs.get("price_source") or "").strip()
            or str(inputs.get("price_source")) == "disabled"
        )
        if price_missing:
            inputs["price"] = price
        if market_price_missing:
            inputs["market_price"] = price
        if metadata_stale:
            inputs["price_status"] = "OK"
            inputs["price_source"] = provider
            inputs["price_source_url"] = source_url
            inputs["price_fetched_at"] = fetched_at
            inputs["price_as_of_date"] = price_as_of


def _overlay_missing_valuation_inputs(
    *,
    fundamentals: dict[str, Any],
    valuations: dict[str, Any],
) -> None:
    net_debt = fundamentals.get("net_debt", UNKNOWN)
    shares = fundamentals.get("shares_outstanding", UNKNOWN)
    issuer_classification = str(fundamentals.get("issuer_classification") or "").strip().lower()
    for method_payload in valuations.values():
        if not isinstance(method_payload, dict):
            continue
        inputs = method_payload.get("inputs")
        if not isinstance(inputs, dict):
            continue
        if not _is_num(inputs.get("net_debt")) and _is_num(net_debt):
            inputs["net_debt"] = float(net_debt)
            inputs["net_debt_status"] = "OK"
            inputs["net_debt_source"] = "evidence_packet_fundamentals"
        if not _is_num(inputs.get("shares_outstanding")) and _is_num(shares) and float(shares) > 0:
            inputs["shares_outstanding"] = float(shares)
            inputs["shares_status"] = "OK"
            inputs["shares_source"] = str(
                inputs.get("shares_source") or "evidence_packet_fundamentals"
            )
        if issuer_classification:
            inputs["issuer_classification"] = issuer_classification


def _packet_from_dossier_payload(
    conn,
    *,
    ticker: str,
    as_of_date: str,
    dossier: dict[str, Any],
    dossier_run_id: str | None = None,
) -> dict[str, Any]:
    frame = build_fundamentals_frame(dossier)
    fundamentals = _flat_fundamentals_from_frame(frame)
    extracted_facts = _facts_from_dossier(dossier)
    financials = _financials_from_dossier(dossier)
    valuation_rows = latest_decision_eligible_valuation_rows(
        conn,
        ticker=ticker,
        as_of_date=as_of_date,
        exact_as_of_date=True,
    )
    valuations: dict[str, Any] = {}
    for row in valuation_rows:
        valuations[row["method"]] = {
            "inputs": json.loads(row["inputs_json"]),
            "outputs": json.loads(row["outputs_json"]),
            "warnings": json.loads(row["warnings_json"]),
        }
    _overlay_missing_fundamentals(
        ticker=ticker,
        as_of_date=as_of_date,
        fundamentals=fundamentals,
        extracted_facts=extracted_facts,
        financials=financials,
    )
    _overlay_companyfacts_trends(
        conn,
        ticker=ticker,
        as_of_date=as_of_date,
        fundamentals=fundamentals,
    )
    _overlay_companyfacts_support(
        conn,
        ticker=ticker,
        as_of_date=as_of_date,
        fundamentals=fundamentals,
        extracted_facts=extracted_facts,
        financials=financials,
    )
    _overlay_missing_price_inputs(
        conn,
        ticker=ticker,
        as_of_date=as_of_date,
        valuations=valuations,
    )
    _overlay_missing_valuation_inputs(
        fundamentals=fundamentals,
        valuations=valuations,
    )

    filings = [
        FilingMetadata(
            accession=str(row.get("accession") or ""),
            form_type=str(row.get("form_type") or ""),
            filing_date=str(row.get("filing_date") or "") or None,
            period_end=str(row.get("period_end") or "") or None,
            primary_doc_url=str(row.get("primary_doc_url") or ""),
        ).model_dump()
        for row in (dossier.get("docket") or [])
        if isinstance(row, dict)
    ]
    if not filings:
        filings = _cached_filing_metadata_for_ticker(ticker=ticker, as_of_date=as_of_date)

    prior_row = conn.execute(
        """
        SELECT packet_path
        FROM evidence_packets
        WHERE ticker = ? AND as_of_date < ?
        ORDER BY as_of_date DESC
        LIMIT 1
        """,
        (ticker, as_of_date),
    ).fetchone()
    prior_metrics: dict[str, Any] = {}
    if prior_row:
        try:
            prior_payload = json.loads(Path(prior_row["packet_path"]).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            prior_payload = {}
        if isinstance(prior_payload, dict):
            prior_metrics = prior_payload.get("fundamentals") or {}

    deltas: dict[str, Any] = {}
    for key, current in fundamentals.items():
        prior = prior_metrics.get(key)
        if isinstance(current, (int, float)) and isinstance(prior, (int, float)):
            deltas[key] = float(current) - float(prior)

    packet = EvidencePacket(
        ticker=ticker,
        as_of_date=as_of_date,
        filings_used=filings,
        extracted_facts=extracted_facts,
        financials=financials,
        fundamentals=fundamentals,
        valuations=valuations,
        deltas_vs_prior_period=deltas,
    )
    payload = packet.model_dump()
    payload = _restore_prompt_financial_provenance(
        payload,
        source_extracted_facts=extracted_facts,
        source_financials=financials,
    )
    payload = _attach_pattern_scan_summary(
        payload,
        ticker=ticker,
        run_id=dossier_run_id or str(dossier.get("run_id") or "") or None,
    )
    return _attach_variant_perceptions(
        payload,
        ticker=ticker,
        as_of_date=as_of_date,
        run_id=dossier_run_id or str(dossier.get("run_id") or "") or None,
    )


def _latest_dossier_record(
    *, ticker: str, run_id: str | None = None
) -> tuple[str, dict[str, Any]] | None:
    cfg = get_config()
    ticker_norm = ticker.upper()
    candidates: list[tuple[float, Path]] = []
    if run_id:
        path = cfg.dossiers_dir / run_id / ticker_norm / "dossier.json"
        if path.exists():
            candidates.append((path.stat().st_mtime, path))
    else:
        for path in cfg.dossiers_dir.glob(f"*/{ticker_norm}/dossier.json"):
            if path.exists():
                candidates.append((path.stat().st_mtime, path))
    for _, path in sorted(candidates, key=lambda item: item[0], reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(payload, dict):
            continue
        payload_as_of = str(payload.get("as_of_date") or "")
        if payload_as_of:
            return payload_as_of, payload
    return None


def _resolve_packet_as_of(
    conn,
    *,
    ticker: str,
    as_of_date: str | None,
    dossier_run_id: str | None,
) -> tuple[str | None, dict[str, Any] | None]:
    if as_of_date:
        dossier = _latest_dossier_payload(
            ticker=ticker, as_of_date=as_of_date, run_id=dossier_run_id
        )
        return as_of_date, dossier

    latest_dossier = _latest_dossier_record(ticker=ticker, run_id=dossier_run_id)
    dossier_as_of, dossier_payload = latest_dossier if latest_dossier else (None, None)
    candidates = [
        _get_latest_as_of(conn, "fundamentals", ticker),
        _latest_valuation_as_of(conn, ticker),
        dossier_as_of,
    ]
    resolved = max((candidate for candidate in candidates if candidate), default=None)
    if not resolved:
        return None, None
    if dossier_payload and dossier_as_of == resolved:
        return resolved, dossier_payload
    dossier = _latest_dossier_payload(ticker=ticker, as_of_date=resolved, run_id=dossier_run_id)
    return resolved, dossier


def build_packet_for_ticker(
    ticker: str,
    as_of_date: str | None = None,
    *,
    dossier_run_id: str | None = None,
) -> Path | None:
    cfg = get_config()
    ticker = ticker.upper()
    from app.fundamentals.metrics import compute_fundamentals_for_ticker

    with get_db() as conn:
        as_of_date, dossier = _resolve_packet_as_of(
            conn,
            ticker=ticker,
            as_of_date=as_of_date,
            dossier_run_id=dossier_run_id,
        )
        payload = None
        if as_of_date and dossier:
            payload = _packet_from_dossier_payload(
                conn,
                ticker=ticker,
                as_of_date=as_of_date,
                dossier=dossier,
                dossier_run_id=dossier_run_id,
            )
        if payload is None and as_of_date:
            payload = _build_packet_payload(conn, ticker, as_of_date)
        if payload is None and as_of_date:
            if compute_fundamentals_for_ticker(ticker, as_of_date=as_of_date):
                payload = _build_packet_payload(conn, ticker, as_of_date)
        if payload is None:
            return None
        payload = _attach_pattern_scan_summary(payload, ticker=ticker, run_id=dossier_run_id)
        payload = _attach_variant_perceptions(
            payload, ticker=ticker, as_of_date=as_of_date, run_id=dossier_run_id
        )

        out_path = cfg.evidence_dir / f"{ticker}_{as_of_date}.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        packet_hash = sha256_file(out_path)

        conn.execute(
            """
            INSERT INTO evidence_packets(ticker, as_of_date, packet_path, packet_hash, created_at)
            VALUES(?, ?, ?, ?, ?)
            ON CONFLICT(ticker, as_of_date) DO UPDATE SET
                packet_path=excluded.packet_path,
                packet_hash=excluded.packet_hash
            """,
            (ticker, as_of_date, str(out_path), packet_hash, utc_now_iso()),
        )

    return out_path


def build_all_packets() -> int:
    built = 0
    tickers: set[str] = set()
    with get_db() as conn:
        rows = conn.execute("SELECT DISTINCT ticker FROM fundamentals ORDER BY ticker").fetchall()
        tickers.update(str(row["ticker"]).upper() for row in rows if row["ticker"])
        rows = conn.execute("SELECT DISTINCT ticker FROM valuations ORDER BY ticker").fetchall()
        for row in rows:
            ticker = str(row["ticker"]).strip().upper()
            if ticker and latest_decision_eligible_valuation_rows(
                conn,
                ticker=ticker,
            ):
                tickers.add(ticker)
    for path in get_config().dossiers_dir.glob("*/*/dossier.json"):
        ticker = path.parent.name.strip().upper()
        if ticker:
            tickers.add(ticker)
    for ticker in sorted(tickers):
        path = build_packet_for_ticker(ticker)
        if path:
            built += 1
    logger.info(
        "evidence_packets_completed", extra={"stage_name": "evidence", "stage_count": built}
    )
    return built
