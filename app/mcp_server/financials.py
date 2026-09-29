"""Normalized financials and raw XBRL concept series, optionally point-in-time.

Built on ``app.ingest.companyfacts`` (tag map + normalizers), which does the work
this tool relies on: ``filed_as_of`` gives the point-in-time view (a fact filed
after the date, or with no filing date, is never read), the quarterly normalizer
labels every 10-Q fact with the fiscal year/quarter of the period it measures
(not the filing's stamps), and every normalized row names the XBRL tag it came
from -- or, for a summed value, its component tags.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter, defaultdict
from datetime import date
from difflib import get_close_matches
from typing import Any

import requests

from app.config import get_config
from app.ingest.companyfacts import (
    COMPANYFACTS_URL,
    TAG_MAP,
    normalize_annual_facts_from_raw,
    normalize_quarterly_facts_from_raw,
)
from app.mcp_server.companies import Company, parse_iso_date, resolve_company
from app.mcp_server.errors import SecToolError
from app.util.http import HttpClient

logger = logging.getLogger("app.mcp_server")

COMPANYFACTS_TTL_SECONDS = 24 * 3600
_MEMORY_TTL_SECONDS = 600
_MEMORY_SLOTS = 3

ALL_LINE_ITEMS: tuple[str, ...] = tuple(TAG_MAP)
DEFAULT_LINE_ITEMS: tuple[str, ...] = (
    "revenue",
    "gross_profit",
    "operating_income",
    "net_income",
    "cfo",
    "capex",
    "depreciation_amortization",
    "sbc",
    "cash",
    "total_debt",
    "total_assets",
    "total_liabilities",
    "equity",
    "shares_outstanding",
    "share_repurchases_amount",
    "dividends_paid_amount",
)
DEFAULT_YEARS = {"annual": 10, "quarterly": 3}
MAX_YEARS = 30

_memory_lock = threading.Lock()
_memory: dict[str, tuple[float, dict[str, Any]]] = {}


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------


def load_companyfacts(company: Company) -> dict[str, Any]:
    """SEC company-facts JSON for ``company`` (disk-cached 24h, memory-cached briefly)."""

    now = time.monotonic()
    with _memory_lock:
        hit = _memory.get(company.cik)
        if hit and now - hit[0] < _MEMORY_TTL_SECONDS:
            return hit[1]
    url = COMPANYFACTS_URL.format(cik=company.cik)
    try:
        payload = HttpClient(get_config()).get_json(url, cache_ttl_seconds=COMPANYFACTS_TTL_SECONDS)
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            who = company.ticker or company.name
            label = f"{who} (CIK {company.cik})" if who else f"CIK {company.cik}"
            raise SecToolError(
                f"SEC has no XBRL financial data for {label}. Company facts "
                "exist only for filers that tag their statements in XBRL (most operating "
                "companies since 2009-2011); funds, trusts and many older or foreign filers have "
                "none. list_filings and get_filing_text still work."
            ) from None
        raise
    if not isinstance(payload, dict) or not isinstance(payload.get("facts"), dict):
        raise SecToolError(f"SEC returned an empty company-facts record for CIK {company.cik}.")
    with _memory_lock:
        _memory[company.cik] = (now, payload)
        while len(_memory) > _MEMORY_SLOTS:
            oldest = min(_memory, key=lambda key: _memory[key][0])
            del _memory[oldest]
    return payload


def clear_memory_cache() -> None:
    with _memory_lock:
        _memory.clear()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _filed_date(fact: dict[str, Any]) -> date | None:
    text = str(fact.get("filed") or "").strip()[:10]
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _as_date(text: Any) -> date | None:
    try:
        return date.fromisoformat(str(text or "")[:10])
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# tool: get_financials
# ---------------------------------------------------------------------------


def _resolve_line_items(line_items: list[str] | str | None) -> list[str]:
    if line_items is None:
        return list(DEFAULT_LINE_ITEMS)
    requested = line_items.split(",") if isinstance(line_items, str) else list(line_items)
    names = [str(item).strip().lower() for item in requested if str(item).strip()]
    if not names:
        return list(DEFAULT_LINE_ITEMS)
    if names == ["all"]:
        return list(ALL_LINE_ITEMS)
    unknown = [name for name in names if name not in TAG_MAP]
    if unknown:
        raise SecToolError(
            f"Unknown line item(s): {', '.join(unknown)}. Valid names: {', '.join(ALL_LINE_ITEMS)} "
            "(or 'all'). For any other XBRL concept use get_concept."
        )
    return list(dict.fromkeys(names))


def _mode(values: list[str]) -> str | None:
    values = [v for v in values if v]
    if not values:
        return None
    counts = Counter(values)
    return max(counts.items(), key=lambda item: (item[1], item[0]))[0]


def get_financials(
    ticker_or_cik: str,
    period: str = "annual",
    years: int | None = None,
    as_of: str | None = None,
    line_items: list[str] | str | None = None,
) -> dict[str, Any]:
    period = str(period or "annual").strip().lower()
    if period not in DEFAULT_YEARS:
        raise SecToolError("period must be 'annual' or 'quarterly'.")
    n_years = DEFAULT_YEARS[period] if years is None else int(years)
    if not 1 <= n_years <= MAX_YEARS:
        raise SecToolError(f"years must be between 1 and {MAX_YEARS}.")
    as_of_d = parse_iso_date(as_of, "as_of")
    items = _resolve_line_items(line_items)

    company = resolve_company(ticker_or_cik)
    raw = load_companyfacts(company)
    normalize = (
        normalize_annual_facts_from_raw if period == "annual" else normalize_quarterly_facts_from_raw
    )
    rows = normalize(
        raw,
        cik=company.cik,
        years_back=n_years + 1,
        filed_as_of=as_of_d.isoformat() if as_of_d else None,
    )

    wanted = set(items)
    rows = [r for r in rows if r.get("line_item") in wanted]
    if as_of_d:
        leaked = [r for r in rows if not r.get("filed_date") or str(r["filed_date"])[:10] > as_of_d.isoformat()]
        if leaked:  # cannot happen: the normalizer reads nothing filed after as_of
            logger.error("Dropped %d rows filed after as_of %s", len(leaked), as_of_d)
            rows = [r for r in rows if r not in leaked]

    fiscal_years = sorted({int(r["fiscal_year"]) for r in rows}, reverse=True)[:n_years]
    keep_years = set(fiscal_years)
    rows = [r for r in rows if int(r["fiscal_year"]) in keep_years]

    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(int(row["fiscal_year"]), str(row["period_type"]))].append(row)

    fp_order = {"FY": 0, "Q4": 1, "Q3": 2, "Q2": 3, "Q1": 4}
    filings: dict[str, dict[str, Any]] = {}
    periods: list[dict[str, Any]] = []
    untraced = 0
    for (fy, fp) in sorted(groups, key=lambda k: (-k[0], fp_order.get(k[1], 9))):
        group = groups[(fy, fp)]
        non_share_ends = [str(r["period_end"]) for r in group if r["line_item"] != "shares_outstanding"]
        period_end = _mode(non_share_ends) or _mode([str(r["period_end"]) for r in group])
        starts = [str(r.get("period_start") or "") for r in group]
        values: dict[str, Any] = {}
        for row in sorted(group, key=lambda r: items.index(r["line_item"])):
            entry: dict[str, Any] = {"value": round(float(row["value"]), 6)}
            components = row.get("components") or []
            if row.get("tag"):
                entry["tag"] = f"{row.get('taxonomy') or 'us-gaap'}:{row['tag']}"
                entry["accn"] = row.get("accession") or None
            elif components:
                entry["sum_of"] = [
                    {"tag": f"{c['taxonomy']}:{c['tag']}", "value": round(c["value"], 6), "accn": c["accession"]}
                    for c in components
                ]
                for c in components:
                    filings.setdefault(str(c["accession"]), {"form": c["form"], "filed": c["filed_date"]})
            else:
                untraced += 1
                entry["tag"] = None
                entry["accn"] = row.get("accession") or None
            if row.get("accession") and not components:
                filings.setdefault(
                    str(row["accession"]), {"form": row.get("form"), "filed": row.get("filed_date")}
                )
            if str(row["period_end"]) != period_end:
                entry["end"] = str(row["period_end"])
            values[str(row["line_item"])] = entry
        periods.append(
            {
                "fiscal_year": fy,
                "fiscal_period": fp,
                "start": _mode(starts),
                "end": period_end,
                "values": values,
            }
        )

    notes = [
        "Values are in millions: USD millions, except shares_outstanding (millions of shares).",
        "Each value names its XBRL tag and filing accession; 'filings' maps accessions to form "
        "and filed date. capex and payouts are positive cash outflows.",
    ]
    if period == "annual":
        notes.append("fiscal_year is the calendar year in which the fiscal year ends.")
    else:
        notes.append(
            "fiscal_year/fiscal_period follow the company's own fiscal calendar; Q4 is not "
            "reported in 10-Qs (use the annual figures)."
        )
    if as_of_d:
        notes.append(
            f"Point-in-time: only facts filed on or before {as_of_d.isoformat()} are used, so "
            "later restatements are excluded."
        )
    else:
        notes.append("Latest view: where a period was restated, the most recently filed value is shown.")
    if untraced:
        notes.append(f"{untraced} value(s) could not be tied to a single tag; their accession is given.")
    if not periods:
        notes.append("No normalized values found for the requested line items and window.")

    return {
        "company": {"cik": company.cik, "ticker": company.ticker, "name": raw.get("entityName") or company.name},
        "period": period,
        "as_of": as_of_d.isoformat() if as_of_d else None,
        "line_items": items,
        "periods": periods,
        "filings": dict(sorted(filings.items(), key=lambda kv: str(kv[1].get("filed") or ""), reverse=True)),
        "source": COMPANYFACTS_URL.format(cik=company.cik),
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# tool: get_concept
# ---------------------------------------------------------------------------

_PERIOD_FILTERS = {"all", "annual", "quarterly", "instant"}


def _days(fact: dict[str, Any]) -> int | None:
    start, end = _as_date(fact.get("start")), _as_date(fact.get("end"))
    if start is None or end is None:
        return None
    return (end - start).days


def _period_ok(fact: dict[str, Any], period: str) -> bool:
    if period == "all":
        return True
    days = _days(fact)
    if period == "instant":
        return not fact.get("start")
    if days is None:
        return False
    if period == "annual":
        return 350 <= days <= 380
    return 80 <= days <= 100


def _latest_key(fact: dict[str, Any]) -> tuple[str, bool, str]:
    return (
        str(fact.get("filed") or ""),
        str(fact.get("form") or "").upper().endswith("/A"),
        str(fact.get("accn") or ""),
    )


def get_concept(
    ticker_or_cik: str,
    concept: str,
    taxonomy: str = "us-gaap",
    unit: str | None = None,
    as_of: str | None = None,
    period: str = "all",
    limit: int = 40,
) -> dict[str, Any]:
    concept_name = str(concept or "").strip()
    if not concept_name:
        raise SecToolError("concept is required, e.g. 'AccountsPayableCurrent'.")
    if ":" in concept_name and taxonomy == "us-gaap":
        taxonomy, concept_name = concept_name.split(":", 1)
    taxonomy = str(taxonomy or "us-gaap").strip()
    period = str(period or "all").strip().lower()
    if period not in _PERIOD_FILTERS:
        raise SecToolError("period must be one of: all, annual, quarterly, instant.")
    limit = int(limit)
    if not 1 <= limit <= 500:
        raise SecToolError("limit must be between 1 and 500.")
    as_of_d = parse_iso_date(as_of, "as_of")

    company = resolve_company(ticker_or_cik)
    raw = load_companyfacts(company)
    facts_root = raw.get("facts") or {}
    taxonomy_node = facts_root.get(taxonomy)
    if not isinstance(taxonomy_node, dict):
        raise SecToolError(
            f"{company.ticker or company.cik} reports no '{taxonomy}' facts. "
            f"Available taxonomies: {', '.join(sorted(facts_root)) or 'none'}."
        )
    node = taxonomy_node.get(concept_name)
    if node is None:
        folded = {name.lower(): name for name in taxonomy_node}
        if concept_name.lower() in folded:
            concept_name = folded[concept_name.lower()]
            node = taxonomy_node[concept_name]
    if node is None:
        needle = concept_name.lower()
        similar = [name for name in sorted(taxonomy_node) if needle in name.lower()][:10]
        if len(similar) < 5:
            similar += [
                n for n in get_close_matches(concept_name, list(taxonomy_node), n=8, cutoff=0.6)
                if n not in similar
            ]
        hint = f" Similar concepts this company reports: {', '.join(similar[:10])}." if similar else ""
        raise SecToolError(
            f"{company.ticker or company.cik} has no {taxonomy}:{concept_name} facts.{hint}"
        )

    units = node.get("units") or {}
    available_units = sorted(units)
    if unit:
        chosen_unit = next((u for u in units if u.lower() == unit.strip().lower()), None)
        if chosen_unit is None:
            raise SecToolError(
                f"{taxonomy}:{concept_name} has no unit {unit!r}; available: {', '.join(available_units)}."
            )
    elif "USD" in units:
        chosen_unit = "USD"
    elif units:
        chosen_unit = max(units, key=lambda u: len(units[u] or []))
    else:
        raise SecToolError(f"{taxonomy}:{concept_name} has no facts.")

    visible = [
        f
        for f in units.get(chosen_unit) or []
        if isinstance(f, dict)
        and f.get("val") is not None
        and (as_of_d is None or ((filed := _filed_date(f)) is not None and filed <= as_of_d))
        and _period_ok(f, period)
    ]
    by_period: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for fact in visible:
        by_period[(str(fact.get("start") or ""), str(fact.get("end") or ""))].append(fact)

    points: list[dict[str, Any]] = []
    for (start, end), facts in by_period.items():
        latest = max(facts, key=_latest_key)
        first = min(facts, key=_latest_key)
        point: dict[str, Any] = {"start": start or None, "end": end, "value": latest["val"]}
        days = _days(latest)
        if days is not None:
            point["months"] = round(days / 30.44)
        point.update(
            {
                "fy": latest.get("fy"),
                "fp": latest.get("fp"),
                "form": latest.get("form"),
                "filed": latest.get("filed"),
                "accn": latest.get("accn"),
            }
        )
        if first is not latest and first.get("val") != latest.get("val"):
            point["first_reported"] = {
                "value": first.get("val"),
                "filed": first.get("filed"),
                "accn": first.get("accn"),
            }
        points.append({k: v for k, v in point.items() if v is not None})
    points.sort(key=lambda p: (p["end"], p.get("start") or ""), reverse=True)

    description = str(node.get("description") or "").strip()
    if len(description) > 400:
        description = description[:397].rstrip() + "..."
    result: dict[str, Any] = {
        "company": {"cik": company.cik, "ticker": company.ticker, "name": raw.get("entityName") or company.name},
        "concept": f"{taxonomy}:{concept_name}",
        "label": node.get("label"),
        "description": description or None,
        "unit": chosen_unit,
        "available_units": available_units,
        "period": period,
        "as_of": as_of_d.isoformat() if as_of_d else None,
        "count": len(points),
        "returned": min(len(points), limit),
        "points": points[:limit],
        "source": COMPANYFACTS_URL.format(cik=company.cik),
        "notes": [
            "One point per (start, end) period: the latest value filed"
            + (f" on or before {as_of_d.isoformat()}" if as_of_d else "")
            + "; 'first_reported' appears when the original filing said something different. "
            "Values are in the raw unit (not scaled).",
        ],
    }
    if len(points) > limit:
        result["truncated"] = True
    return {k: v for k, v in result.items() if v is not None or k == "as_of"}
