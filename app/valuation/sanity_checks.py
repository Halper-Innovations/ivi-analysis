from __future__ import annotations

import json
import math
from datetime import date
from typing import Any

from app.db import get_db, utc_now_iso
from app.fundamentals.normalize import UNKNOWN
from app.logging import get_logger
from app.valuation.dcf_lite import DCFInputs, run_dcf_lite
from app.valuation.multiples import compute_intrinsic_range_from_multiples
from app.valuation.price_provider import get_default_provider
from app.valuation.reverse_dcf import implied_growth_from_price


logger = get_logger(__name__)


def _latest_rows(conn, table: str) -> list:
    return conn.execute(
        f"""
        SELECT t1.*
        FROM {table} t1
        INNER JOIN (
            SELECT ticker, MAX(as_of_date) AS max_date
            FROM {table}
            GROUP BY ticker
        ) t2 ON t1.ticker = t2.ticker AND t1.as_of_date = t2.max_date
        """
    ).fetchall()


def _normalized_shares_value(value: Any, units: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0:
        return None
    normalized_units = str(units or "").strip().lower()
    if normalized_units == "shares_millions":
        return numeric
    if normalized_units == "shares":
        return numeric / 1_000_000.0
    return None


def _latest_shares(
    conn,
    ticker: str,
    *,
    run_as_of_date: str,
) -> dict[str, Any] | None:
    """Resolve shares visible at the run boundary with exact source proof."""

    try:
        boundary = date.fromisoformat(str(run_as_of_date))
    except ValueError as exc:
        raise ValueError("run_as_of_date must be YYYY-MM-DD") from exc

    candidates: list[dict[str, Any]] = []
    companyfacts_rows = conn.execute(
        """
        WITH candidates AS (
            SELECT
                id AS source_id,
                value,
                units,
                period_end,
                filed_date,
                accession,
                source_url,
                0 AS source_rank
            FROM companyfacts_facts
            WHERE UPPER(ticker) = ?
              AND line_item = 'shares_outstanding'
              AND value IS NOT NULL
            UNION ALL
            SELECT
                id AS source_id,
                value,
                units,
                period_end,
                filed_date,
                accession,
                source_url,
                1 AS source_rank
            FROM companyfacts_vintages
            WHERE UPPER(ticker) = ?
              AND line_item = 'shares_outstanding'
              AND value IS NOT NULL
        )
        SELECT *
        FROM candidates
        ORDER BY period_end DESC, filed_date DESC, source_rank ASC, source_id DESC
        """,
        (ticker.upper(), ticker.upper()),
    ).fetchall()
    for row in companyfacts_rows:
        period_end = str(row["period_end"] or "").strip()
        filed_date = str(row["filed_date"] or "").strip()
        accession = str(row["accession"] or "").strip()
        source_url = str(row["source_url"] or "").strip()
        try:
            period = date.fromisoformat(period_end)
            filed = date.fromisoformat(filed_date)
        except ValueError:
            continue
        if (
            period > filed
            or filed > boundary
            or period > boundary
            or not accession
            or not source_url
        ):
            continue
        value = _normalized_shares_value(row["value"], row["units"])
        if value is None:
            continue
        candidates.append(
            {
                "value": value,
                "units": "shares_millions",
                "raw_value": float(row["value"]),
                "raw_units": str(row["units"] or ""),
                "period_end": period_end,
                "filed_date": filed_date,
                "accession": accession,
                "source_url": source_url,
                "source": "companyfacts",
            }
        )

    extracted_rows = conn.execute(
        """
        SELECT
            ef.value_json,
            ef.source_url,
            f.accession,
            f.filing_date,
            f.period_end
        FROM extracted_facts ef
        JOIN filings f ON ef.filing_id = f.id
        WHERE UPPER(f.ticker) = ?
          AND ef.fact_type = 'shares_outstanding'
        ORDER BY f.period_end DESC, f.filing_date DESC, f.id DESC
        """,
        (ticker.upper(),),
    ).fetchall()
    for row in extracted_rows:
        period_end = str(row["period_end"] or "").strip()
        filed_date = str(row["filing_date"] or "").strip()
        accession = str(row["accession"] or "").strip()
        source_url = str(row["source_url"] or "").strip()
        try:
            period = date.fromisoformat(period_end)
            filed = date.fromisoformat(filed_date)
            payload = json.loads(row["value_json"] or "{}")
        except (ValueError, TypeError, json.JSONDecodeError):
            continue
        if (
            period > filed
            or filed > boundary
            or period > boundary
            or not accession
            or not source_url
        ):
            continue
        raw_value = payload.get("value") if isinstance(payload, dict) else None
        value = _normalized_shares_value(raw_value, "shares")
        if value is None:
            continue
        candidates.append(
            {
                "value": value,
                "units": "shares_millions",
                "raw_value": float(raw_value),
                "raw_units": "shares",
                "period_end": period_end,
                "filed_date": filed_date,
                "accession": accession,
                "source_url": source_url,
                "source": "extracted_filing",
            }
        )

    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            str(item["period_end"]),
            str(item["filed_date"]),
            1 if item["source"] == "companyfacts" else 0,
            str(item["accession"]),
        ),
    )


def _historical_multiple_bands(conn, ticker: str) -> tuple[list[float], list[float]]:
    # v0 placeholder: historical filing-derived EV data is usually unavailable without market price.
    # return empty bands and let valuation mark low confidence.
    _ = conn
    _ = ticker
    return [], []


def _insert_valuation(
    conn,
    ticker: str,
    as_of_date: str,
    method: str,
    inputs: dict,
    outputs: dict,
    warnings: list[str],
) -> None:
    # Route through the measurement scope, archive the pre-overwrite row,
    # and stamp the writer version (previously unstamped on this path).
    from app.valuation.measurement import valuations_table
    from app.valuation.valuation_writer import _VERSION, _archive_valuation_row

    outputs_json = json.dumps(outputs)
    _archive_valuation_row(
        conn,
        ticker=ticker,
        as_of_date=as_of_date,
        method=method,
        new_outputs_json=outputs_json,
    )
    conn.execute(
        f"""
        INSERT INTO {valuations_table()}(
            ticker, as_of_date, method, inputs_json, outputs_json,
            warnings_json, created_at, valuation_writer_version)
        VALUES(?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker, as_of_date, method) DO UPDATE SET
            inputs_json=excluded.inputs_json,
            outputs_json=excluded.outputs_json,
            warnings_json=excluded.warnings_json,
            valuation_writer_version=excluded.valuation_writer_version
        """,
        (
            ticker,
            as_of_date,
            method,
            json.dumps(inputs),
            outputs_json,
            json.dumps(warnings),
            utc_now_iso(),
            _VERSION,
        ),
    )


def run_all_valuations(as_of_date: str | None = None) -> int:
    count = 0
    provider = get_default_provider()
    with get_db() as conn:
        rows = _latest_rows(conn, "fundamentals")
    for row in rows:
        if run_valuation_for_ticker(
            row["ticker"],
            provider=provider,
            as_of_date=row["as_of_date"],
            run_as_of_date=as_of_date,
        ):
            count += 1

    logger.info("valuation_completed", extra={"stage_name": "valuation", "stage_count": count})
    return count


def run_valuation_for_ticker(
    ticker: str,
    *,
    conn=None,
    provider=None,
    as_of_date: str | None = None,
    run_as_of_date: str | None = None,
) -> bool:
    own_conn = False
    if conn is None:
        own_conn = True
        db_ctx = get_db()
        conn = db_ctx.__enter__()
    try:
        effective_run_as_of = str(run_as_of_date or date.today().isoformat()).strip()
        try:
            run_boundary = date.fromisoformat(effective_run_as_of)
        except ValueError as exc:
            raise ValueError("run_as_of_date must be YYYY-MM-DD") from exc
        if as_of_date:
            try:
                fundamentals_date = date.fromisoformat(str(as_of_date))
            except ValueError as exc:
                raise ValueError("as_of_date must be YYYY-MM-DD") from exc
            if fundamentals_date > run_boundary:
                return False
            row = conn.execute(
                "SELECT as_of_date, metrics_json FROM fundamentals WHERE ticker = ? AND as_of_date = ?",
                (ticker, as_of_date),
            ).fetchone()
        else:
            row = conn.execute(
                """
                SELECT as_of_date, metrics_json
                FROM fundamentals
                WHERE ticker = ? AND as_of_date <= ?
                ORDER BY as_of_date DESC
                LIMIT 1
                """,
                (ticker, effective_run_as_of),
            ).fetchone()
        if not row:
            return False

        as_of_date = row["as_of_date"]
        metrics = json.loads(row["metrics_json"])
        revenue = metrics.get("revenue") if metrics.get("revenue") != UNKNOWN else None
        op_margin = (
            metrics.get("operating_margin") if metrics.get("operating_margin") != UNKNOWN else None
        )
        fcf = metrics.get("fcf") if metrics.get("fcf") != UNKNOWN else None
        net_debt = metrics.get("net_debt") if metrics.get("net_debt") != UNKNOWN else None
        shares_evidence = _latest_shares(
            conn,
            ticker,
            run_as_of_date=effective_run_as_of,
        )
        if shares_evidence is None:
            logger.warning(
                "valuation_shares_unproven",
                extra={
                    "stage_name": "valuation",
                    "ticker": ticker.upper(),
                    "run_as_of_date": effective_run_as_of,
                },
            )
            return False
        shares = float(shares_evidence["value"])
        provider = provider or get_default_provider()
        quote_as_of = effective_run_as_of
        price_quote = provider.get_quote(ticker, quote_as_of)

        ev_fcf_hist, ev_ebit_hist = _historical_multiple_bands(conn, ticker)
        mult_out, mult_warn = compute_intrinsic_range_from_multiples(
            fcf=fcf,
            ebit_proxy=(revenue * op_margin)
            if revenue is not None and op_margin is not None
            else None,
            historical_ev_fcf=ev_fcf_hist,
            historical_ev_ebit=ev_ebit_hist,
            net_debt=net_debt,
            shares_outstanding=shares,
        )
        from app.valuation.measurement import measurement_scope

        with measurement_scope():
            _insert_valuation(
                conn,
                ticker,
                as_of_date,
                "multiples",
                {
                    "fcf": fcf,
                    "ebit_proxy": (
                        (revenue * op_margin)
                        if revenue is not None and op_margin is not None
                        else None
                    ),
                    "net_debt": net_debt,
                    "shares_outstanding": shares,
                    "shares_evidence": shares_evidence,
                },
                mult_out,
                mult_warn,
            )

        dcf_inputs = DCFInputs(
            revenue=revenue,
            operating_margin=op_margin,
            tax_rate=None,
            reinvestment_rate=None,
            discount_rate=0.10,
            terminal_growth=0.02,
            shares_outstanding=shares,
            net_debt=net_debt,
        )
        dcf_out, dcf_warn = run_dcf_lite(dcf_inputs)
        if fcf is None or (isinstance(fcf, (int, float)) and fcf <= 0):
            dcf_warn.append("Negative or missing FCF; DCF confidence reduced")
            dcf_out["confidence"] = "LOW"
        implied_mult_base = (
            dcf_out.get("implied_terminal_ebit_multiple", {}).get("base")
            if isinstance(dcf_out.get("implied_terminal_ebit_multiple"), dict)
            else None
        )
        if isinstance(implied_mult_base, (int, float)):
            if ev_ebit_hist:
                low_band, high_band = min(ev_ebit_hist), max(ev_ebit_hist)
            else:
                low_band, high_band = 7.0, 14.0
                dcf_warn.append(
                    "Historical EV/EBIT unavailable; using conservative default sanity band [7x,14x]"
                )
            if implied_mult_base < low_band or implied_mult_base > high_band:
                dcf_warn.append(
                    f"Implied terminal EBIT multiple ({implied_mult_base:.2f}x) outside sanity band [{low_band:.2f}x,{high_band:.2f}x]"
                )
        with measurement_scope():
            _insert_valuation(
                conn,
                ticker,
                as_of_date,
                "dcf_lite",
                {
                    **dcf_inputs.__dict__,
                    "shares_evidence": shares_evidence,
                },
                dcf_out,
                dcf_warn,
            )

        rev_out, rev_warn = implied_growth_from_price(
            market_price=price_quote.price,
            shares_outstanding=shares,
            net_debt=net_debt,
            base_revenue=revenue,
            margin=op_margin,
        )
        with measurement_scope():
            _insert_valuation(
                conn,
                ticker,
                as_of_date,
                "reverse_dcf",
                {
                    "market_price": (
                        price_quote.price if price_quote.price is not None else UNKNOWN
                    ),
                    "price_source": price_quote.provider,
                    "price_status": price_quote.status,
                    "price_source_url": price_quote.source_url,
                    "price_fetched_at": price_quote.fetched_at,
                    "price_as_of_date": price_quote.as_of_date,
                    "valuation_run_as_of_date": quote_as_of,
                    "shares_outstanding": shares,
                    "shares_evidence": shares_evidence,
                    "net_debt": net_debt,
                },
                rev_out,
                rev_warn,
            )
        return True
    finally:
        if own_conn:
            db_ctx.__exit__(None, None, None)
