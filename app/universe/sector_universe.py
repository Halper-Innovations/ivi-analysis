from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.autonomous.financial_integrity import InvalidFinancialInputError
from app.config import AppConfig, get_config
from app.db import get_db, init_db, utc_now_iso
from app.diff.engine import build_filing_diff_for_ticker
from app.dossier.runner import run_dossier_for_peer_set
from app.ingest.facts_writer import ensure_facts
from app.logging import get_logger
from app.market.company_facts_provider import normalize_cik
from app.market.price_prewarm import write_prices_prewarm_for_run
from app.patterns.scanner import scan_peer_set
from app.synthesis.variant_builder import build_variant_perceptions
from app.universe.ticker_cik_map import SEC_TICKER_JSON_URL
from app.util.financial_data_access import ANNUAL_COMPANYFACTS_PERIOD_TYPES, companyfacts_rows
from app.util.http import HttpClient
from app.valuation.engine import write_valuations_for_run
from app.valuation.intangible_economics import write_intangible_economics_for_run
from app.valuation.lineage import latest_decision_eligible_valuation_row
from app.valuation.rubric import apply_value_first_overlay
from app.valuation.value_gates import PASS, WATCH, write_value_gates_for_run
from app.valuation.valuation_writer import ensure_valuation


logger = get_logger(__name__)

_DISCOVERY_TTL_SECONDS = 7 * 24 * 3600
_SUBMISSIONS_TTL_SECONDS = 7 * 24 * 3600
_SEC_DELAY_SECONDS = 0.15
_MIN_ANNUAL_HISTORY = 3
_DEFAULT_TIER3_YEARS_BACK = 5
UNKNOWN = "UNKNOWN"
INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
VALUATION_ANOMALY = "VALUATION_ANOMALY"


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


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


def _load_prewarm_prices(
    *, run_id: str, tickers: list[str], as_of_date: str, cfg: AppConfig | None = None
) -> dict[str, float]:
    cfg = cfg or get_config()
    out: dict[str, float] = {}
    for ticker in sorted(
        {str(symbol).strip().upper() for symbol in tickers if str(symbol).strip()}
    ):
        payload = _safe_json(cfg.outputs_dir / "prices" / run_id / f"{ticker}.json")
        if str(payload.get("status") or "").upper() != "OK":
            continue
        if str(payload.get("requested_as_of_date") or "") not in {"", str(as_of_date)}:
            continue
        snapshot = payload.get("snapshot") if isinstance(payload.get("snapshot"), dict) else {}
        diagnostic = (
            payload.get("diagnostic") if isinstance(payload.get("diagnostic"), dict) else {}
        )
        output_fields = (
            diagnostic.get("output_fields")
            if isinstance(diagnostic.get("output_fields"), dict)
            else {}
        )
        price = snapshot.get("price")
        if not _is_num(price):
            price = output_fields.get("current_price")
        if _is_num(price) and float(price) > 0:
            out[ticker] = float(price)
    return out


def _inject_prewarm_prices_into_fundamentals(
    *,
    fundamentals_by_ticker: dict[str, dict[str, Any]],
    prewarm_prices: dict[str, float],
) -> None:
    for ticker, price in prewarm_prices.items():
        fundamentals = fundamentals_by_ticker.get(ticker)
        if not isinstance(fundamentals, dict):
            continue
        rows = fundamentals.get("rows")
        if not isinstance(rows, list) or not rows:
            continue
        latest = rows[-1]
        if not isinstance(latest, dict):
            continue
        latest["current_price"] = float(price)
        row_traces = fundamentals.get("row_traces")
        year_key = str(latest.get("year") or "")
        if isinstance(row_traces, dict) and isinstance(row_traces.get(year_key), dict):
            row_traces[year_key]["current_price"] = {
                "derived_from": [
                    f"outputs/prices/{fundamentals.get('run_id')}/{ticker}.json.snapshot.price"
                ],
                "citations": [],
            }


def _cache_is_fresh(path: Path, *, ttl_seconds: int) -> bool:
    if not path.exists():
        return False
    try:
        return (time.time() - path.stat().st_mtime) <= max(1, int(ttl_seconds))
    except Exception:
        return False


def _normalize_sector_key(value: Any) -> str:
    return str(value or "").strip().lower()


def _default_run_id(prefix: str, sector: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{prefix}_{_normalize_sector_key(sector)}_{stamp}"


def _ticker_rows_cache_path(cfg: AppConfig | None = None) -> Path:
    cfg = cfg or get_config()
    path = cfg.cache_dir / "company_tickers.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _sector_cache_dir(cfg: AppConfig | None = None) -> Path:
    cfg = cfg or get_config()
    path = cfg.cache_dir / "sector_universe"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sector_cache_path(sector: str, cfg: AppConfig | None = None) -> Path:
    return _sector_cache_dir(cfg=cfg) / f"{_normalize_sector_key(sector)}.json"


def _submissions_cache_path(cik: str | int, cfg: AppConfig | None = None) -> Path:
    cfg = cfg or get_config()
    path = cfg.cache_dir / "submissions" / f"{normalize_cik(cik)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _parse_sic_range(token: str) -> tuple[int, int]:
    raw = str(token or "").strip()
    if not raw:
        raise ValueError("empty SIC range token")
    if "-" in raw:
        start_raw, end_raw = raw.split("-", 1)
        start = int(start_raw.strip())
        end = int(end_raw.strip())
        if end < start:
            raise ValueError(f"invalid SIC range: {raw}")
        return start, end
    value = int(raw)
    return value, value


def load_sector_sic_config(
    *, path: Path | None = None, cfg: AppConfig | None = None
) -> dict[str, dict[str, Any]]:
    cfg = cfg or get_config()
    config_path = path or cfg.sector_sic_config_path
    payload = _safe_json(config_path)
    out: dict[str, dict[str, Any]] = {}
    for raw_sector, raw_entry in payload.items():
        sector = _normalize_sector_key(raw_sector)
        if not sector:
            continue
        entry = raw_entry if isinstance(raw_entry, dict) else {}
        sic_ranges = []
        for item in entry.get("sic_ranges") or []:
            try:
                sic_ranges.append(_parse_sic_range(str(item)))
            except Exception:
                continue
        if not sic_ranges:
            continue
        out[sector] = {
            "label": str(entry.get("label") or sector),
            "sic_ranges": sic_ranges,
        }
    return out


def list_configured_sectors(*, cfg: AppConfig | None = None) -> list[str]:
    return sorted(load_sector_sic_config(cfg=cfg).keys())


def has_sector_definition(sector: str, *, cfg: AppConfig | None = None) -> bool:
    return _normalize_sector_key(sector) in load_sector_sic_config(cfg=cfg)


def _sic_in_ranges(sic: int | None, ranges: list[tuple[int, int]]) -> bool:
    if sic is None:
        return False
    return any(start <= int(sic) <= end for start, end in ranges)


def load_company_ticker_rows(
    *, http: HttpClient | None = None, cfg: AppConfig | None = None
) -> list[dict[str, Any]]:
    cfg = cfg or get_config()
    http = http or HttpClient(cfg)
    path = _ticker_rows_cache_path(cfg=cfg)
    payload: dict[str, Any]
    if _cache_is_fresh(path, ttl_seconds=_DISCOVERY_TTL_SECONDS):
        payload = _safe_json(path)
    else:
        payload = http.get_json(
            SEC_TICKER_JSON_URL, use_cache=True, cache_ttl_seconds=_DISCOVERY_TTL_SECONDS
        )
        _json_write(path, payload)
    rows: list[dict[str, Any]] = []
    for _, raw_row in sorted(
        payload.items(), key=lambda item: int(item[0]) if str(item[0]).isdigit() else str(item[0])
    ):
        row = raw_row if isinstance(raw_row, dict) else {}
        ticker = str(row.get("ticker") or "").strip().upper()
        cik_value = row.get("cik_str")
        cik = normalize_cik(cik_value) if cik_value is not None else ""
        company_name = str(row.get("title") or row.get("name") or "").strip()
        if not ticker or not cik:
            continue
        rows.append({"ticker": ticker, "cik": cik, "company_name": company_name})
    return rows


def load_company_submissions(
    cik: str | int,
    *,
    http: HttpClient | None = None,
    cfg: AppConfig | None = None,
    refresh_if_missing: bool = True,
) -> dict[str, Any]:
    cfg = cfg or get_config()
    http = http or HttpClient(cfg)
    cache_path = _submissions_cache_path(cik, cfg=cfg)
    cached = _safe_json(cache_path)
    if cached and _cache_is_fresh(cache_path, ttl_seconds=_SUBMISSIONS_TTL_SECONDS):
        return cached
    if not refresh_if_missing and cached:
        return cached
    url = f"https://data.sec.gov/submissions/CIK{normalize_cik(cik)}.json"
    try:
        payload = http.get_json(url, use_cache=True, cache_ttl_seconds=_SUBMISSIONS_TTL_SECONDS)
        _json_write(cache_path, payload if isinstance(payload, dict) else {})
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return cached


def _extract_sic(submissions: dict[str, Any]) -> int | None:
    for key in ("sic", "sicCode"):
        value = submissions.get(key)
        if value is None:
            continue
        token = str(value).strip()
        if token.isdigit():
            return int(token)
    return None


def _latest_annual_filing_info(submissions: dict[str, Any]) -> dict[str, str]:
    recent = (
        submissions.get("filings", {}).get("recent")
        if isinstance(submissions.get("filings"), dict)
        else {}
    )
    forms = list(recent.get("form") or []) if isinstance(recent, dict) else []
    filing_dates = list(recent.get("filingDate") or []) if isinstance(recent, dict) else []
    accessions = list(recent.get("accessionNumber") or []) if isinstance(recent, dict) else []
    best_date = ""
    best_accession = ""
    for idx, form in enumerate(forms):
        form_token = str(form or "").upper()
        if form_token not in {"10-K", "10-K/A", "20-F", "20-F/A", "40-F", "40-F/A"}:
            continue
        filing_date = str(filing_dates[idx] if idx < len(filing_dates) else "")
        accession = str(accessions[idx] if idx < len(accessions) else "")
        if filing_date >= best_date:
            best_date = filing_date
            best_accession = accession
    return {"filing_date": best_date, "accession": best_accession}


def _load_sector_overrides(*, cfg: AppConfig | None = None) -> dict[str, set[str]]:
    cfg = cfg or get_config()
    path = cfg.sector_overrides_path
    out: dict[str, set[str]] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            sector = _normalize_sector_key(row.get("sector"))
            ticker = str(row.get("ticker") or "").strip().upper()
            if not sector or not ticker:
                continue
            out.setdefault(sector, set()).add(ticker)
    return out


def discover_sector_universe(
    sector: str,
    *,
    http: HttpClient | None = None,
    cfg: AppConfig | None = None,
    include_supplements: bool = True,
    refresh_cache: bool = False,
) -> list[dict[str, Any]]:
    cfg = cfg or get_config()
    sector_key = _normalize_sector_key(sector)
    config = load_sector_sic_config(cfg=cfg)
    if sector_key not in config:
        raise ValueError(f"unknown sector: {sector}")

    cache_path = _sector_cache_path(sector_key, cfg=cfg)
    if not refresh_cache and _cache_is_fresh(cache_path, ttl_seconds=_DISCOVERY_TTL_SECONDS):
        cached = _safe_json(cache_path)
        rows = cached.get("rows")
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]

    ranges = list(config[sector_key]["sic_ranges"])
    http = http or HttpClient(cfg)
    ticker_rows = load_company_ticker_rows(http=http, cfg=cfg)
    matches: list[dict[str, Any]] = []
    by_ticker: dict[str, dict[str, Any]] = {}

    for row in ticker_rows:
        cik = str(row.get("cik") or "")
        if not cik:
            continue
        submissions = load_company_submissions(cik, http=http, cfg=cfg)
        sic = _extract_sic(submissions)
        if not _sic_in_ranges(sic, ranges):
            continue
        entry = {
            "ticker": str(row.get("ticker") or "").upper(),
            "cik": cik,
            "company_name": str(row.get("company_name") or submissions.get("name") or "").strip(),
            "sic": int(sic) if sic is not None else None,
        }
        matches.append(entry)
        by_ticker[entry["ticker"]] = entry

    if include_supplements:
        overrides = _load_sector_overrides(cfg=cfg)
        supplements = overrides.get(sector_key, set())
        for ticker in sorted(supplements):
            if ticker in by_ticker:
                continue
            source = next(
                (row for row in ticker_rows if str(row.get("ticker") or "").upper() == ticker), None
            )
            if not source:
                continue
            submissions = load_company_submissions(str(source.get("cik") or ""), http=http, cfg=cfg)
            sic = _extract_sic(submissions)
            entry = {
                "ticker": ticker,
                "cik": str(source.get("cik") or ""),
                "company_name": str(
                    source.get("company_name") or submissions.get("name") or ""
                ).strip(),
                "sic": int(sic) if sic is not None else None,
                "supplemental": True,
            }
            matches.append(entry)
            by_ticker[ticker] = entry

    matches = sorted(
        matches, key=lambda row: (str(row.get("company_name") or ""), str(row.get("ticker") or ""))
    )
    _json_write(
        cache_path,
        {
            "sector": sector_key,
            "generated_at": utc_now_iso(),
            "ticker_count": len(matches),
            "rows": matches,
        },
    )
    return matches


def load_sector_ticker_map(
    *,
    sectors: list[str] | None = None,
    cfg: AppConfig | None = None,
    discover_missing: bool = True,
) -> dict[str, list[str]]:
    cfg = cfg or get_config()
    target_sectors = sorted(
        {
            _normalize_sector_key(item)
            for item in (sectors or list_configured_sectors(cfg=cfg))
            if _normalize_sector_key(item)
        }
    )
    static_map: dict[str, set[str]] = {}
    taxonomy_path = cfg.sector_taxonomy_path
    if taxonomy_path.exists():
        with taxonomy_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                sector = _normalize_sector_key(row.get("sector"))
                ticker = str(row.get("ticker") or "").strip().upper()
                if sector and ticker:
                    static_map.setdefault(sector, set()).add(ticker)
    overrides = _load_sector_overrides(cfg=cfg)

    out: dict[str, list[str]] = {}
    for sector in target_sectors:
        rows: list[dict[str, Any]] = []
        if discover_missing:
            try:
                rows = discover_sector_universe(sector, cfg=cfg)
            except Exception:
                rows = []
        else:
            cached = _safe_json(_sector_cache_path(sector, cfg=cfg))
            cached_rows = cached.get("rows")
            rows = (
                [row for row in cached_rows if isinstance(row, dict)]
                if isinstance(cached_rows, list)
                else []
            )
        tickers = {
            str(row.get("ticker") or "").upper()
            for row in rows
            if str(row.get("ticker") or "").strip()
        }
        tickers.update(static_map.get(sector, set()))
        tickers.update(overrides.get(sector, set()))
        out[sector] = sorted(tickers)
    return out


def _companyfacts_year_rows(ticker: str, *, years_back: int = 10) -> list[dict[str, Any]]:
    try:
        ensure_facts(ticker, years_back=years_back)
    except Exception as exc:
        logger.warning(
            "Tier 1 companyfacts unavailable for ticker=%s; treating as insufficient data",
            ticker.upper(),
            extra={
                "ticker": ticker.upper(),
                "years_back": int(years_back),
                "error": str(exc),
            },
        )
        return []
    with get_db() as conn:
        rows = companyfacts_rows(
            conn,
            ticker,
            columns=("fiscal_year", "line_item", "value", "period_end"),
            period_types=ANNUAL_COMPANYFACTS_PERIOD_TYPES,
            order_by="fiscal_year ASC, line_item ASC",
        )
    by_year: dict[int, dict[str, Any]] = {}
    for row in rows:
        year = int(row["fiscal_year"] or 0)
        if year <= 0:
            continue
        bucket = by_year.setdefault(
            year,
            {
                "year": year,
                "period_end": str(row["period_end"] or ""),
            },
        )
        bucket[str(row["line_item"])] = float(row["value"]) if _is_num(row["value"]) else UNKNOWN
    years = sorted(by_year.keys())
    if years_back > 0 and len(years) > years_back:
        years = years[-years_back:]
    normalized: list[dict[str, Any]] = []
    for year in years:
        row = dict(by_year[year])
        cfo = row.get("cfo")
        capex = row.get("capex")
        debt = row.get("total_debt")
        cash = row.get("cash")
        revenue = row.get("revenue")
        gross_profit = row.get("gross_profit")
        operating_income = row.get("operating_income")
        shares = row.get("shares_outstanding")
        row["fcf"] = (float(cfo) - float(capex)) if _is_num(cfo) and _is_num(capex) else UNKNOWN
        row["net_debt"] = (
            (float(debt) - float(cash)) if _is_num(debt) and _is_num(cash) else UNKNOWN
        )
        row["gross_margin"] = (
            (float(gross_profit) / float(revenue))
            if _is_num(gross_profit) and _is_num(revenue) and float(revenue)
            else UNKNOWN
        )
        row["op_margin"] = (
            (float(operating_income) / float(revenue))
            if _is_num(operating_income) and _is_num(revenue) and float(revenue)
            else UNKNOWN
        )
        row["fcf_margin"] = (
            (float(row["fcf"]) / float(revenue))
            if _is_num(row.get("fcf")) and _is_num(revenue) and float(revenue)
            else UNKNOWN
        )
        row["cfo_margin"] = (
            (float(cfo) / float(revenue))
            if _is_num(cfo) and _is_num(revenue) and float(revenue)
            else UNKNOWN
        )
        row["shares_outstanding"] = shares if _is_num(shares) else UNKNOWN
        normalized.append(row)
    return normalized


def _cagr(series: list[tuple[int, float]], years: int) -> float | str:
    if len(series) < 2:
        return UNKNOWN
    end_year, end_value = series[-1]
    if end_value <= 0:
        return UNKNOWN
    target_year = end_year - int(years)
    candidates = [(year, value) for year, value in series if year <= target_year and value > 0]
    if candidates:
        start_year, start_value = candidates[-1]
    else:
        start_year, start_value = series[0]
    if start_value <= 0:
        return UNKNOWN
    span = max(1, end_year - start_year)
    return (end_value / start_value) ** (1.0 / span) - 1.0


def _slope(series: list[tuple[int, float]]) -> float | str:
    if len(series) < 2:
        return UNKNOWN
    start_year, start_value = series[0]
    end_year, end_value = series[-1]
    span = max(1, end_year - start_year)
    return (end_value - start_value) / float(span)


def build_companyfacts_fundamentals_frame(
    ticker: str,
    *,
    as_of_date: str,
    run_id: str,
    years_back: int = 10,
) -> dict[str, Any]:
    rows = _companyfacts_year_rows(ticker, years_back=years_back)
    row_traces: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        year = int(row.get("year") or 0)
        row_traces[str(year)] = {}
        for metric in (
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
            "share_repurchases_amount",
            "dividends_paid_amount",
            "total_assets",
        ):
            row_traces[str(year)][metric] = {
                "derived_from": [f"companyfacts_facts[{ticker.upper()},{year},{metric}]"],
                "citations": [],
            }

    revenue_series = [
        (int(row["year"]), float(row["revenue"]))
        for row in rows
        if _is_num(row.get("revenue")) and float(row["revenue"]) > 0
    ]
    shares_series = [
        (int(row["year"]), float(row["shares_outstanding"]))
        for row in rows
        if _is_num(row.get("shares_outstanding")) and float(row["shares_outstanding"]) > 0
    ]
    gm_series = [
        (int(row["year"]), float(row["gross_margin"]))
        for row in rows
        if _is_num(row.get("gross_margin"))
    ]
    om_series = [
        (int(row["year"]), float(row["op_margin"])) for row in rows if _is_num(row.get("op_margin"))
    ]
    fcfm_series = [
        (int(row["year"]), float(row["fcf_margin"]))
        for row in rows
        if _is_num(row.get("fcf_margin"))
    ]

    derived_signals = {
        "revenue_cagr_3y": {
            "value": _cagr(revenue_series, 3),
            "derived_from": [f"companyfacts_facts[{ticker.upper()}].revenue"],
        },
        "revenue_cagr_5y": {
            "value": _cagr(revenue_series, 5),
            "derived_from": [f"companyfacts_facts[{ticker.upper()}].revenue"],
        },
        "revenue_cagr_10y": {
            "value": _cagr(revenue_series, 10),
            "derived_from": [f"companyfacts_facts[{ticker.upper()}].revenue"],
        },
        "gross_margin_trend_slope": {
            "value": _slope(gm_series),
            "derived_from": [f"companyfacts_facts[{ticker.upper()}].gross_margin"],
        },
        "operating_margin_trend_slope": {
            "value": _slope(om_series),
            "derived_from": [f"companyfacts_facts[{ticker.upper()}].op_margin"],
        },
        "fcf_margin_trend_slope": {
            "value": _slope(fcfm_series),
            "derived_from": [f"companyfacts_facts[{ticker.upper()}].fcf_margin"],
        },
        "dilution_rate_shares_cagr": {
            "value": _cagr(shares_series, 10),
            "derived_from": [f"companyfacts_facts[{ticker.upper()}].shares_outstanding"],
        },
        "r_and_d_intensity_latest": {
            "value": (
                (float(rows[-1]["r_and_d_total"]) / float(rows[-1]["revenue"]))
                if rows
                and _is_num(rows[-1].get("r_and_d_total"))
                and _is_num(rows[-1].get("revenue"))
                and float(rows[-1]["revenue"])
                else UNKNOWN
            ),
            "derived_from": [
                f"companyfacts_facts[{ticker.upper()}].r_and_d_total",
                f"companyfacts_facts[{ticker.upper()}].revenue",
            ],
        },
    }

    return {
        "fundamentals_version": "v1.1",
        "ticker": ticker.upper(),
        "run_id": run_id,
        "as_of_date": as_of_date,
        "rows": rows,
        "row_traces": row_traces,
        "derived_signals": derived_signals,
        "gaps": [],
        "generated_at": utc_now_iso(),
    }


def quick_screen_sector_universe(
    rows: list[dict[str, Any]],
    *,
    as_of_date: str,
    cfg: AppConfig | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cfg = cfg or get_config()
    results: list[dict[str, Any]] = []
    passed: list[dict[str, Any]] = []
    revenue_floor = float(cfg.universe_tier1_revenue_min_musd)
    for row in rows:
        ticker = str(row.get("ticker") or "").upper()
        if not ticker:
            continue
        annual_rows = _companyfacts_year_rows(ticker, years_back=10)
        annual_count = len(annual_rows)
        latest = annual_rows[-1] if annual_rows else {}
        revenue = latest.get("revenue", UNKNOWN)
        net_income_history = [
            float(item["net_income"])
            for item in annual_rows[-3:]
            if _is_num(item.get("net_income"))
        ]
        profitable_years = len([value for value in net_income_history if value > 0])
        has_any_positive_net_income = profitable_years > 0
        revenue_ok = _is_num(revenue) and float(revenue) >= revenue_floor
        history_ok = annual_count >= _MIN_ANNUAL_HISTORY
        profitability_ok = has_any_positive_net_income
        passed_screen = revenue_ok and history_ok and profitability_ok
        result = {
            "ticker": ticker,
            "cik": row.get("cik"),
            "company_name": row.get("company_name"),
            "sic": row.get("sic"),
            "annual_history_count": annual_count,
            "latest_revenue_musd": float(revenue) if _is_num(revenue) else UNKNOWN,
            "latest_net_income_musd": float(latest["net_income"])
            if _is_num(latest.get("net_income"))
            else UNKNOWN,
            "latest_shares_outstanding_musd": float(latest["shares_outstanding"])
            if _is_num(latest.get("shares_outstanding"))
            else UNKNOWN,
            "latest_total_assets_musd": float(latest["total_assets"])
            if _is_num(latest.get("total_assets"))
            else UNKNOWN,
            "profitable_years_last_3": profitable_years,
            "pass_screen": passed_screen,
            "screen_flags": {
                "revenue_ok": revenue_ok,
                "history_ok": history_ok,
                "profitability_ok": profitability_ok,
            },
            "screen_status": "PASS" if passed_screen else "FILTERED_OUT",
            "screen_reason": (
                "PASSED"
                if passed_screen
                else (
                    "COMPANYFACTS_UNAVAILABLE"
                    if annual_count == 0
                    else "REVENUE_BELOW_FLOOR"
                    if not revenue_ok
                    else "INSUFFICIENT_HISTORY"
                    if not history_ok
                    else "NO_POSITIVE_NET_INCOME"
                    if not profitability_ok
                    else "SCREEN_FILTERED"
                )
            ),
        }
        results.append(result)
        if passed_screen:
            passed.append(result)
    return results, passed


def _write_json_per_ticker(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _seed_scoreboard(run_dir: Path, tickers: list[str]) -> None:
    scoreboard = {
        "rows": [
            {"ticker": ticker, "metric_values": {}, "metric_traces": {}}
            for ticker in sorted(tickers)
        ],
        "metrics": [],
        "ranking_mode": "tier2_seed",
        "generated_at": utc_now_iso(),
    }
    rankings = {
        "rankings": [{"ticker": ticker, "metric_ranks": {}} for ticker in sorted(tickers)],
        "generated_at": utc_now_iso(),
    }
    _json_write(run_dir / "peer_scoreboard.json", scoreboard)
    _json_write(run_dir / "peer_rankings.json", rankings)


def _load_sector_run_dir(run_id: str, cfg: AppConfig | None = None) -> Path:
    cfg = cfg or get_config()
    path = cfg.sectors_dir / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _status_rank(value: Any) -> int:
    token = str(value or "").upper()
    if token == PASS:
        return 0
    if token == WATCH:
        return 1
    return 2


def _load_gate_rows(run_dir: Path) -> list[dict[str, Any]]:
    payload = _safe_json(run_dir / "value_gates.json")
    return [row for row in (payload.get("entries") or []) if isinstance(row, dict)]


def _load_valuation_payload(run_dir: Path, ticker: str) -> dict[str, Any]:
    return _safe_json(run_dir / f"valuation_{ticker.upper()}.json")


def _load_writer_scorecard_payload(*, ticker: str, as_of_date: str) -> dict[str, Any]:
    try:
        with get_db() as conn:
            row = latest_decision_eligible_valuation_row(
                conn,
                ticker=ticker,
                method="scorecard",
                as_of_date=as_of_date,
                exact_as_of_date=True,
            )
    except Exception:
        return {}
    if not row or not str(row["outputs_json"] or "").strip():
        return {}
    try:
        payload = json.loads(str(row["outputs_json"]))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _augment_tier2_pricing_zone_artifacts(
    *, run_dir: Path, tickers: list[str], as_of_date: str
) -> None:
    prewarm_prices = _load_prewarm_prices(
        run_id=run_dir.name, tickers=tickers, as_of_date=as_of_date
    )
    for ticker in tickers:
        valuation_path = run_dir / f"valuation_{ticker.upper()}.json"
        valuation = _safe_json(valuation_path)
        if not valuation:
            continue

        try:
            ensure_valuation(
                ticker.upper(),
                as_of_date,
                run_id=run_dir.name,
                price_override=prewarm_prices.get(ticker.upper()),
                force_refresh=ticker.upper() in prewarm_prices,
                require_filed_asof=True,
            )
        except Exception:
            pass

        zone = INSUFFICIENT_DATA
        detail: dict[str, Any] = {
            "reason": "Pricing zone unavailable from valuation scorecard.",
        }
        scorecard = _load_writer_scorecard_payload(ticker=ticker, as_of_date=as_of_date)
        if isinstance(scorecard, dict) and str(scorecard.get("pricing_zone") or "").strip():
            zone = str(scorecard.get("pricing_zone") or INSUFFICIENT_DATA).upper()
            detail = (
                scorecard.get("pricing_zone_detail")
                if isinstance(scorecard.get("pricing_zone_detail"), dict)
                else {}
            )

        current_price = (
            (valuation.get("input_snapshot") or {}).get("current_price")
            if isinstance(valuation.get("input_snapshot"), dict)
            else UNKNOWN
        )
        if not _is_num(current_price) or float(current_price) <= 0:
            zone = INSUFFICIENT_DATA
            detail = dict(detail)
            detail.setdefault("reason", "Current price unavailable.")

        if str(valuation.get("valuation_reason_code") or "").upper() == VALUATION_ANOMALY:
            zone = VALUATION_ANOMALY
            anomaly_detail = (
                valuation.get("valuation_anomaly_detail")
                if isinstance(valuation.get("valuation_anomaly_detail"), dict)
                else {}
            )
            detail = dict(anomaly_detail)
            detail.setdefault("reason", "Implied return exceeded anomaly threshold.")

        valuation["pricing_zone"] = zone
        valuation["pricing_zone_detail"] = detail
        _write_json_per_ticker(valuation_path, valuation)


def _tier2_ranked_candidates(run_dir: Path) -> list[dict[str, Any]]:
    gate_rows = _load_gate_rows(run_dir)
    out: list[dict[str, Any]] = []
    for row in gate_rows:
        ticker = str(row.get("ticker") or "").upper()
        valuation = _load_valuation_payload(run_dir, ticker)
        pricing_zone = str(valuation.get("pricing_zone") or INSUFFICIENT_DATA).upper()
        implied_return = valuation.get("implied_return_base", UNKNOWN)
        valuation_reason_code = str(valuation.get("valuation_reason_code") or UNKNOWN).upper()
        anomaly_flag = valuation_reason_code == VALUATION_ANOMALY or (
            _is_num(implied_return) and abs(float(implied_return)) > 100.0
        )
        if anomaly_flag:
            pricing_zone = VALUATION_ANOMALY
            implied_value: float | str = UNKNOWN
        else:
            implied_value = float(implied_return) if _is_num(implied_return) else UNKNOWN
        out.append(
            {
                "ticker": ticker,
                "value_gate_status": str(row.get("gate_status") or UNKNOWN).upper(),
                "primary_blocker": str(row.get("primary_blocker") or UNKNOWN),
                "implied_return_base": implied_value,
                "pricing_zone": pricing_zone,
                "pricing_zone_detail": valuation.get("pricing_zone_detail")
                if isinstance(valuation.get("pricing_zone_detail"), dict)
                else {},
                "valuation_reason_code": valuation_reason_code,
                "anomaly_flag": anomaly_flag,
            }
        )
    out.sort(
        key=lambda row: (
            1 if row.get("anomaly_flag") else 0,
            _status_rank(row.get("value_gate_status")),
            -float(row.get("implied_return_base"))
            if _is_num(row.get("implied_return_base"))
            else float("inf"),
            str(row.get("ticker") or ""),
        )
    )
    return out


def _zone_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        zone = str(row.get("pricing_zone") or INSUFFICIENT_DATA).upper()
        counts[zone] = counts.get(zone, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def _select_tier3_candidates(
    ranked_candidates: list[dict[str, Any]],
    *,
    tier2_rank_limit: int,
    tier3_limit: int,
) -> tuple[list[str], list[str]]:
    if tier3_limit <= 0 or tier2_rank_limit <= 0:
        return [], []

    tier2_window = ranked_candidates[: max(0, int(tier2_rank_limit))]
    selected: list[str] = []
    margin_of_safety_selected: list[str] = []

    for row in tier2_window:
        ticker = str(row.get("ticker") or "").upper()
        if not ticker:
            continue
        if str(row.get("pricing_zone") or "").upper() != "MARGIN_OF_SAFETY":
            continue
        if ticker in selected:
            continue
        selected.append(ticker)
        margin_of_safety_selected.append(ticker)
        if len(selected) >= tier3_limit:
            return selected, margin_of_safety_selected

    for row in tier2_window:
        ticker = str(row.get("ticker") or "").upper()
        if not ticker or ticker in selected:
            continue
        if str(row.get("value_gate_status") or "").upper() not in {"PASS", "WATCH"}:
            continue
        selected.append(ticker)
        if len(selected) >= tier3_limit:
            break

    return selected, margin_of_safety_selected


def _margin_of_safety_alerts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("pricing_zone") or "").upper() != "MARGIN_OF_SAFETY":
            continue
        detail = (
            row.get("pricing_zone_detail")
            if isinstance(row.get("pricing_zone_detail"), dict)
            else {}
        )
        alerts.append(
            {
                "alert_type": "MARGIN_OF_SAFETY_DETECTED",
                "ticker": str(row.get("ticker") or "").upper(),
                "price": detail.get("current_price"),
                "epv_adjusted": detail.get("epv_adjusted"),
                # TEXTBOOK convention: (intrinsic - price) / intrinsic
                # (audit: dual-mos-convention-same-name)
                "margin_of_safety_pct": detail.get("margin_of_safety_vs_epv_adjusted"),
                "margin_of_safety_convention": "TEXTBOOK (intrinsic-price)/intrinsic",
                "gate_status": str(row.get("value_gate_status") or UNKNOWN).upper(),
                "primary_blocker": str(row.get("primary_blocker") or UNKNOWN),
                "epv_quality": detail.get("epv_quality"),
                "revenue_cagr_5y": detail.get("revenue_cagr_5y"),
                "earnings_quality": detail.get("earnings_quality"),
                "gate_action": detail.get("gate_action"),
            }
        )
    return alerts


def _build_tier2_artifacts(
    *,
    sector: str,
    run_id: str,
    tickers: list[str],
    as_of_date: str,
    cfg: AppConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or get_config()
    run_dir = _load_sector_run_dir(run_id, cfg=cfg)
    fundamentals_by_ticker: dict[str, dict[str, Any]] = {}
    for ticker in tickers:
        fundamentals = build_companyfacts_fundamentals_frame(
            ticker,
            as_of_date=as_of_date,
            run_id=run_id,
            years_back=10,
        )
        fundamentals_by_ticker[ticker] = fundamentals
        _write_json_per_ticker(run_dir / f"fundamentals_{ticker}.json", fundamentals)

    price_prewarm_summary: dict[str, Any] = {
        "run_id": run_id,
        "as_of_date": as_of_date,
        "ok_count": 0,
        "unknown_count": 0,
        "tickers_ok": [],
        "tickers_unknown": [],
        "reason_counts": {},
    }
    # No network override here: VOE_NET_PROVIDER alone decides, and the LLM
    # provider setting no longer implies offline.
    try:
        price_prewarm_summary = write_prices_prewarm_for_run(
            run_id=run_id,
            as_of_date=as_of_date,
            tickers=tickers,
            fallback_days=max(0, int(cfg.price_fallback_days)),
            cfg=cfg,
        )
    except Exception as exc:
        logger.warning(
            "tier2 price prewarm failed", extra={"run_id": run_id, "error": str(exc)}
        )
    prewarm_prices = _load_prewarm_prices(
        run_id=run_id, tickers=tickers, as_of_date=as_of_date, cfg=cfg
    )
    _inject_prewarm_prices_into_fundamentals(
        fundamentals_by_ticker=fundamentals_by_ticker,
        prewarm_prices=prewarm_prices,
    )
    for ticker, fundamentals in fundamentals_by_ticker.items():
        _write_json_per_ticker(run_dir / f"fundamentals_{ticker}.json", fundamentals)
    for ticker in sorted(
        {str(symbol).strip().upper() for symbol in tickers if str(symbol).strip()}
    ):
        price_path = cfg.outputs_dir / "prices" / run_id / f"{ticker}.json"
        payload = _safe_json(price_path)
        diagnostic = (
            payload.get("diagnostic") if isinstance(payload.get("diagnostic"), dict) else {}
        )
        snapshot = payload.get("snapshot") if isinstance(payload.get("snapshot"), dict) else {}
        status = str(payload.get("status") or "MISSING").upper()
        if status == "OK":
            logger.info(
                "Price resolved for %s: $%s (source: %s, cached: %s)",
                ticker,
                snapshot.get("price"),
                snapshot.get("source")
                or (
                    (diagnostic.get("output_fields") or {}).get("price_source")
                    if isinstance(diagnostic.get("output_fields"), dict)
                    else UNKNOWN
                ),
                bool((diagnostic.get("cache") or {}).get("hit"))
                if isinstance(diagnostic, dict)
                else False,
            )
        else:
            result = diagnostic.get("result") if isinstance(diagnostic, dict) else {}
            logger.warning(
                "Price unavailable for %s (reason: %s)",
                ticker,
                (result or {}).get("reason_code")
                if isinstance(result, dict)
                else "PROVIDER_NO_DATA",
            )

    _seed_scoreboard(run_dir, tickers)
    valuation_summary = write_valuations_for_run(
        run_id=run_id,
        tickers=tickers,
        as_of_date=as_of_date,
        output_dir=run_dir,
        with_prices=True,
    )
    intangible_path = run_dir / "intangible_economics.json"
    write_intangible_economics_for_run(
        run_id=run_id,
        as_of_date=as_of_date,
        tickers=tickers,
        output_path=intangible_path,
        fundamentals_by_ticker=fundamentals_by_ticker,
        cfg=cfg,
    )
    apply_value_first_overlay(
        run_id=run_id,
        as_of_date=as_of_date,
        sector_run_dir=run_dir,
        dossier_run_dir=cfg.dossiers_dir / run_id,
        with_prices=True,
    )
    _augment_tier2_pricing_zone_artifacts(run_dir=run_dir, tickers=tickers, as_of_date=as_of_date)
    gates = write_value_gates_for_run(run_id=run_id, output_dir=run_dir, tickers=tickers)
    ranked = _tier2_ranked_candidates(run_dir)
    summary = {
        "sector": _normalize_sector_key(sector),
        "run_id": run_id,
        "as_of_date": as_of_date,
        "ticker_count": len(tickers),
        "price_prewarm_path": str(
            price_prewarm_summary.get("prices_prewarm_path")
            or (cfg.sectors_dir / run_id / "prices_prewarm.json")
        ),
        "price_prewarm_ok_count": int(price_prewarm_summary.get("ok_count") or 0),
        "price_prewarm_unknown_count": int(price_prewarm_summary.get("unknown_count") or 0),
        "price_reason_counts": dict(price_prewarm_summary.get("reason_counts") or {}),
        "valuation_summary_path": str(
            valuation_summary.get("summary_path") or run_dir / "valuation_summary.json"
        ),
        "intangible_economics_path": str(intangible_path),
        "value_gates_path": str(gates.get("value_gates_path") or run_dir / "value_gates.json"),
        "value_gate_counts": dict((gates.get("summary") or {}).get("counts") or {}),
        "zone_distribution": _zone_distribution(ranked),
        "ranked_candidates": ranked,
    }
    _json_write(run_dir / "tier2_summary.json", summary)
    return summary


def _run_tier3(
    *,
    run_id: str,
    tier3_tickers: list[str],
    as_of_date: str,
    cfg: AppConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or get_config()
    if not tier3_tickers:
        return {
            "run_id": run_id,
            "ticker_count": 0,
            "tier3_tickers": [],
            "dossier_status": "SKIPPED",
        }

    dossier_summary = run_dossier_for_peer_set(
        tickers=tier3_tickers,
        as_of_date=as_of_date,
        years_back=_DEFAULT_TIER3_YEARS_BACK,
        run_id=run_id,
        workers=1,
        resume=False,
        min_annual_filings=3,
    )
    diff_paths: list[str] = []
    for ticker in tier3_tickers:
        try:
            diff_paths.append(
                str(
                    build_filing_diff_for_ticker(
                        ticker=ticker, run_id=run_id, years_back=_DEFAULT_TIER3_YEARS_BACK
                    )
                )
            )
        except InvalidFinancialInputError:
            raise
        except Exception as exc:
            logger.warning("tier3 filing diff failed", extra={"ticker": ticker, "error": str(exc)})
    pattern_report = scan_peer_set(run_id=run_id, tickers=tier3_tickers)
    variant_paths: list[str] = []
    for ticker in tier3_tickers:
        try:
            report = build_variant_perceptions(
                ticker=ticker, as_of_date=as_of_date, run_id=run_id, cfg=cfg
            )
            variant_paths.append(
                str((cfg.outputs_dir / "variant_perceptions" / f"{ticker}_{as_of_date}.json"))
            )
            _ = report
        except Exception as exc:
            logger.warning(
                "tier3 variant synthesis failed", extra={"ticker": ticker, "error": str(exc)}
            )
    summary = {
        "run_id": run_id,
        "ticker_count": len(tier3_tickers),
        "tier3_tickers": list(tier3_tickers),
        "dossier_summary_path": str((cfg.dossiers_dir / run_id / "dossier_summary.json")),
        "dossier_status": str(dossier_summary.get("status") or UNKNOWN),
        "diff_paths": diff_paths,
        "pattern_report_path": str(cfg.outputs_dir / "patterns" / f"{run_id}_pattern_scan.json"),
        "patterns_with_signal": list(pattern_report.patterns_with_signal),
        "variant_paths": variant_paths,
    }
    _json_write(cfg.sectors_dir / run_id / "tier3_summary.json", summary)
    return summary


def _latest_annual_filing_snapshots(
    rows: list[dict[str, Any]], *, cfg: AppConfig | None = None
) -> dict[str, dict[str, str]]:
    cfg = cfg or get_config()
    snapshots: dict[str, dict[str, str]] = {}
    http = HttpClient(cfg)
    for row in rows:
        ticker = str(row.get("ticker") or "").upper()
        cik = str(row.get("cik") or "")
        if not ticker or not cik:
            continue
        submissions = load_company_submissions(cik, http=http, cfg=cfg)
        snapshots[ticker] = _latest_annual_filing_info(submissions)
    return snapshots


def _price_zone_lookup(run_dir: Path, tickers: list[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for ticker in tickers:
        valuation = _load_valuation_payload(run_dir, ticker)
        detail = (
            valuation.get("pricing_zone_detail")
            if isinstance(valuation.get("pricing_zone_detail"), dict)
            else {}
        )
        out[ticker.upper()] = {
            "pricing_zone": str(valuation.get("pricing_zone") or UNKNOWN),
            "current_price": (valuation.get("input_snapshot") or {}).get("current_price", UNKNOWN),
            "epv_adjusted": detail.get("epv_adjusted_per_share", UNKNOWN),
            "dcf_base": detail.get("dcf_base_per_share", UNKNOWN),
            "implied_return_base": valuation.get("implied_return_base", UNKNOWN),
        }
    return out


def deep_scan_sector(
    *,
    sector: str,
    as_of_date: str,
    tier1_limit: int = 300,
    tier2_limit: int = 30,
    tier3_limit: int = 10,
    run_id: str | None = None,
    continuous: bool = False,
    interval_seconds: int | None = None,
    cfg: AppConfig | None = None,
    max_continuous_cycles: int | None = None,
) -> dict[str, Any]:
    cfg = cfg or get_config()
    init_db(cfg)
    run_id = run_id or _default_run_id("deep_scan", sector)
    run_dir = _load_sector_run_dir(run_id, cfg=cfg)
    interval = int(
        interval_seconds
        if interval_seconds is not None
        else cfg.universe_continuous_interval_seconds
    )

    discovered = discover_sector_universe(sector, cfg=cfg)
    logger.info("Tier 1 discovery: %s discovered for sector=%s", len(discovered), sector)
    tier1_candidates = discovered[: max(1, int(tier1_limit))]
    tier1_rows, tier1_passed = quick_screen_sector_universe(
        tier1_candidates, as_of_date=as_of_date, cfg=cfg
    )
    logger.info(
        "Tier 1 screen: %s screened, %s passed for sector=%s",
        len(tier1_rows),
        len(tier1_passed),
        sector,
    )
    tier2_tickers = [str(row.get("ticker") or "").upper() for row in tier1_passed]
    tier2_summary = _build_tier2_artifacts(
        sector=sector,
        run_id=run_id,
        tickers=tier2_tickers,
        as_of_date=as_of_date,
        cfg=cfg,
    )
    logger.info(
        "Tier 2 valuation: %s valued, gate_counts=%s, zone_distribution=%s",
        len(tier2_tickers),
        json.dumps(tier2_summary.get("value_gate_counts") or {}, sort_keys=True),
        json.dumps(tier2_summary.get("zone_distribution") or {}, sort_keys=True),
    )
    ranked_candidates = list(tier2_summary.get("ranked_candidates") or [])
    tier2_rank_limit = max(0, int(tier2_limit))
    tier3_limit_effective = max(0, int(tier3_limit))
    tier3_tickers, tier3_margin_of_safety = _select_tier3_candidates(
        ranked_candidates,
        tier2_rank_limit=tier2_rank_limit,
        tier3_limit=tier3_limit_effective,
    )
    alerts = _margin_of_safety_alerts(ranked_candidates[:tier2_rank_limit])
    logger.info("Tier 3 depth: running top %s tickers", len(tier3_tickers))
    tier3_summary = _run_tier3(
        run_id=run_id, tier3_tickers=tier3_tickers, as_of_date=as_of_date, cfg=cfg
    )

    summary = {
        "sector": _normalize_sector_key(sector),
        "run_id": run_id,
        "as_of_date": as_of_date,
        "tier1_limit": int(tier1_limit),
        "tier2_limit": int(tier2_limit),
        "tier3_limit": int(tier3_limit),
        "discovered_count": len(discovered),
        "tier1_evaluated_count": len(tier1_rows),
        "tier1_pass_count": len(tier1_passed),
        "tier1_pass_rate": (len(tier1_passed) / len(tier1_rows)) if tier1_rows else 0.0,
        "tier2_ticker_count": len(tier2_tickers),
        "tier2_zone_distribution": dict(tier2_summary.get("zone_distribution") or {}),
        "tier2_gate_counts": dict(tier2_summary.get("value_gate_counts") or {}),
        "tier2_anomaly_count": int(
            (tier2_summary.get("zone_distribution") or {}).get(VALUATION_ANOMALY) or 0
        ),
        "tier2_ranked_candidates": ranked_candidates[:tier2_rank_limit],
        "tier3_tickers": tier3_tickers,
        "tier3_margin_of_safety": tier3_margin_of_safety,
        "alerts": alerts,
        "artifacts": {
            "sector_run_dir": str(run_dir),
            "tier2_summary_path": str(run_dir / "tier2_summary.json"),
            "tier3_summary_path": str(run_dir / "tier3_summary.json"),
            "deep_scan_summary_path": str(run_dir / "deep_scan_summary.json"),
        },
        "continuous": {
            "enabled": bool(continuous),
            "interval_seconds": int(interval),
            "alerts": [],
            "cycles_completed": 0,
        },
    }
    _json_write(run_dir / "deep_scan_summary.json", summary)

    if not continuous:
        return summary

    tracked_tickers = [
        str(row.get("ticker") or "").upper() for row in ranked_candidates[:tier2_rank_limit]
    ]
    prior_zone_lookup = _price_zone_lookup(run_dir, tracked_tickers)
    prior_filing_lookup = _latest_annual_filing_snapshots(
        [row for row in discovered if str(row.get("ticker") or "").upper() in set(tracked_tickers)],
        cfg=cfg,
    )
    cycle_count = 0
    while True:
        if max_continuous_cycles is not None and cycle_count >= max(0, int(max_continuous_cycles)):
            break
        time.sleep(max(1, interval))
        _build_tier2_artifacts(
            sector=sector,
            run_id=run_id,
            tickers=tracked_tickers,
            as_of_date=as_of_date,
            cfg=cfg,
        )
        current_zone_lookup = _price_zone_lookup(run_dir, tracked_tickers)
        current_filing_lookup = _latest_annual_filing_snapshots(
            [
                row
                for row in discovered
                if str(row.get("ticker") or "").upper() in set(tracked_tickers)
            ],
            cfg=cfg,
        )
        alerts: list[str] = []
        for ticker in tracked_tickers:
            previous = prior_zone_lookup.get(ticker, {})
            current = current_zone_lookup.get(ticker, {})
            old_zone = str(previous.get("pricing_zone") or UNKNOWN)
            new_zone = str(current.get("pricing_zone") or UNKNOWN)
            if old_zone != new_zone:
                alerts.append(
                    f"ALERT: {ticker} moved from {old_zone} to {new_zone} "
                    f"(price {previous.get('current_price', UNKNOWN)} -> {current.get('current_price', UNKNOWN)}, "
                    f"EPV adj = {current.get('epv_adjusted', UNKNOWN)})"
                )
        for ticker in tracked_tickers:
            previous = prior_filing_lookup.get(ticker, {})
            current = current_filing_lookup.get(ticker, {})
            if str(current.get("filing_date") or "") > str(previous.get("filing_date") or ""):
                alerts.append(
                    f"ALERT: New 10-K detected for {ticker} ({current.get('filing_date')}); full re-analysis queued"
                )
                refresh_run_id = f"{run_id}__refresh_{ticker}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
                _run_tier3(
                    run_id=refresh_run_id, tier3_tickers=[ticker], as_of_date=as_of_date, cfg=cfg
                )
        prior_zone_lookup = current_zone_lookup
        prior_filing_lookup = current_filing_lookup
        cycle_count += 1
        summary["continuous"]["cycles_completed"] = cycle_count
        summary["continuous"]["alerts"].extend(alerts)
        _json_write(run_dir / "deep_scan_summary.json", summary)
        if alerts:
            for alert in alerts:
                logger.info(alert)
    return summary
