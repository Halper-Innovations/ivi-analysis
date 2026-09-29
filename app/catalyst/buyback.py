"""Buyback acceleration detection from companyfacts FY history.

A secondary, WEAK-only catalyst signal: when annual share repurchases are
accelerating, the company is putting cash behind a value-cheap price. This is
deterministic and reads only the cached ``companyfacts_facts`` FY rows; it never
hits the network.

The catalyst is an orthogonal timing axis: it informs the price-trigger overlay
only, never the conviction grade.
"""

from __future__ import annotations

from datetime import date
import math

# Threshold above which the latest FY repurchase counts as ACCELERATING vs STEADY.
ACCELERATION_RATIO = 1.25

LINE_ITEM = "share_repurchases_amount"


def _no_buyback_signal(
    *,
    status: str = "NO_DATA",
    reason_codes: tuple[str, ...] = ("NO_VISIBLE_FILED_ASOF_FACTS",),
) -> dict:
    return {
        "label": "NONE",
        "latest_fy": None,
        "prior_fy": None,
        "fiscal_year": None,
        "unit": "USD_millions",
        "status": status,
        "reason_codes": list(reason_codes),
        "source_lineage": [],
    }


def detect_buyback_signal(conn, ticker: str, as_of_date: str | date) -> dict:
    """Classify the latest annual buyback trend for ``ticker``.

    Queries ``companyfacts_facts`` for the two most recent visible FY
    ``share_repurchases_amount`` rows. Every accepted row must carry the literal
    normalized unit ``USD_millions`` and satisfy
    ``period_end <= filed_date <= as_of_date`` with nonblank source URL and
    accession. Some issuers report repurchases as a negative cash-flow line, so
    signal amounts are normalized with ``abs()`` while the reported value remains
    in the returned lineage.

    Labels:
      * ``ACCELERATING`` -- latest FY > 0 and latest > prior * 1.25
      * ``STEADY``       -- latest FY > 0 and within +/-25% of prior
      * ``NONE``         -- latest FY == 0 or no FY rows at all

    Missing or malformed provenance returns ``status='NEEDS_DATA'`` and
    ``label='NONE'`` so the fact cannot affect watchlist trigger status.
    """
    raw_as_of = as_of_date.isoformat() if isinstance(as_of_date, date) else str(as_of_date or "")
    try:
        as_of_str = date.fromisoformat(raw_as_of[:10]).isoformat()
    except ValueError:
        return _no_buyback_signal(
            status="NEEDS_DATA",
            reason_codes=("INVALID_AS_OF_DATE",),
        )
    columns = {
        str(row[1]) for row in conn.execute("PRAGMA table_info(companyfacts_facts)").fetchall()
    }
    required_columns = {
        "ticker",
        "fiscal_year",
        "period_type",
        "period_end",
        "filed_date",
        "line_item",
        "value",
        "units",
        "source_url",
        "accession",
    }
    if not required_columns <= columns:
        return _no_buyback_signal(
            status="NEEDS_DATA",
            reason_codes=("MISSING_REQUIRED_COLUMNS",),
        )
    rows = conn.execute(
        "SELECT fiscal_year, period_end, filed_date, value, units, "
        "source_url, accession FROM companyfacts_facts "
        "WHERE ticker = ? AND line_item = ? AND period_type = 'FY' "
        "ORDER BY fiscal_year DESC, period_end DESC, filed_date DESC",
        (ticker.upper(), LINE_ITEM),
    ).fetchall()

    if not rows:
        return _no_buyback_signal()

    field_order = (
        "fiscal_year",
        "period_end",
        "filed_date",
        "value",
        "units",
        "source_url",
        "accession",
    )
    as_of_value = date.fromisoformat(as_of_str)
    visible: list[dict] = []
    invalid_reasons: set[str] = set()
    for row in rows:
        record = dict(row) if hasattr(row, "keys") else dict(zip(field_order, row, strict=True))
        period_end_raw = str(record.get("period_end") or "").strip()
        try:
            period_end = date.fromisoformat(period_end_raw)
        except ValueError:
            invalid_reasons.add("MISSING_OR_INVALID_PERIOD_END")
            continue
        if period_end > as_of_value:
            continue

        filed_date_raw = str(record.get("filed_date") or "").strip()
        try:
            filed_date = date.fromisoformat(filed_date_raw)
        except ValueError:
            invalid_reasons.add("MISSING_OR_INVALID_FILED_DATE")
            continue
        if filed_date > as_of_value:
            continue
        if period_end > filed_date:
            invalid_reasons.add("PERIOD_END_AFTER_FILED_DATE")
            continue

        value = record.get("value")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            invalid_reasons.add("MISSING_OR_NONFINITE_VALUE")
            continue
        if str(record.get("units") or "").strip() != "USD_millions":
            invalid_reasons.add("INVALID_OR_MISSING_UNIT")
            continue
        source_url = str(record.get("source_url") or "").strip()
        accession = str(record.get("accession") or "").strip()
        if not source_url:
            invalid_reasons.add("MISSING_SOURCE_URL")
            continue
        if not accession:
            invalid_reasons.add("MISSING_ACCESSION")
            continue
        try:
            fiscal_year = int(record["fiscal_year"])
        except (TypeError, ValueError):
            invalid_reasons.add("MISSING_OR_INVALID_FISCAL_YEAR")
            continue
        reported_value = float(value)
        visible.append(
            {
                "line_item": LINE_ITEM,
                "fiscal_year": fiscal_year,
                "reported_value": reported_value,
                "normalized_value": abs(reported_value),
                "unit": "USD_millions",
                "period_end": period_end.isoformat(),
                "filed_date": filed_date.isoformat(),
                "accession": accession,
                "source_url": source_url,
                "source_reference": (f"{source_url}#line_item={LINE_ITEM}&accession={accession}"),
            }
        )

    if invalid_reasons:
        return _no_buyback_signal(
            status="NEEDS_DATA",
            reason_codes=tuple(sorted(invalid_reasons)),
        )
    if not visible:
        return _no_buyback_signal()

    fiscal_years = [record["fiscal_year"] for record in visible]
    if len(fiscal_years) != len(set(fiscal_years)):
        return _no_buyback_signal(
            status="NEEDS_DATA",
            reason_codes=("AMBIGUOUS_FISCAL_YEAR",),
        )
    visible.sort(
        key=lambda record: (
            record["fiscal_year"],
            record["period_end"],
            record["filed_date"],
            record["accession"],
        ),
        reverse=True,
    )
    selected = visible[:2]
    for role, record in zip(("latest_fy", "prior_fy"), selected, strict=False):
        record["role"] = role

    latest = selected[0]
    latest_fy_year = latest["fiscal_year"]
    latest_fy = latest["normalized_value"]
    prior_fy = selected[1]["normalized_value"] if len(selected) > 1 else None

    if not latest_fy:
        label = "NONE"
    elif prior_fy is not None and latest_fy > prior_fy * ACCELERATION_RATIO:
        label = "ACCELERATING"
    else:
        label = "STEADY"

    return {
        "label": label,
        "latest_fy": latest_fy,
        "prior_fy": prior_fy,
        "fiscal_year": latest_fy_year,
        "unit": "USD_millions",
        "status": "OK",
        "reason_codes": [],
        "source_lineage": selected,
    }
