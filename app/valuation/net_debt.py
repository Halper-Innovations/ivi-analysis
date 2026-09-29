from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

from app.config import AppConfig, get_config
from app.db import utc_now_iso
from app.market.company_facts_extract import (
    OPERATING_LEASE_CURRENT_TAG,
    OPERATING_LEASE_DIRECT_TAG_PRIORITY,
    OPERATING_LEASE_NONCURRENT_TAG,
    extract_cash_equivalents_asof,
    extract_operating_lease_liability_asof,
    extract_short_term_investments_asof,
    extract_total_debt_asof,
)
from app.util.issuer_classification import (
    classify_issuer_by_sic,
    registrant_sic,
    ISSUER_CLASS_FINANCIAL,
    ISSUER_CLASS_OPERATING,
    infer_issuer_classification,
)
from app.valuation.evidenced_zero import raw_debt_zero_evidence
from app.valuation.facts import resolve_financial_facts_asof


logger = logging.getLogger(__name__)

UNKNOWN = "UNKNOWN"
STATUS_OK = "OK"
STATUS_UNKNOWN = "UNKNOWN"
CONFIDENCE_HIGH = "HIGH"
CONFIDENCE_MEDIUM = "MEDIUM"
CONFIDENCE_LOW = "LOW"

REASON_OK = "OK"
REASON_MISSING_CASH = "MISSING_CASH"
REASON_MISSING_DEBT = "MISSING_DEBT"
REASON_MISSING_BOTH = "MISSING_BOTH"
REASON_DATELINE_MISMATCH = "DATELINE_MISMATCH"
# Debt and cash more than one quarter apart are not one balance sheet: refuse. (It was one
# fiscal year, 366 days, which netted a debt figure against cash three quarters newer at
# MEDIUM confidence.) Within a quarter the pair is netted, named and capped at MEDIUM.
_MAX_DEBT_CASH_GAP_DAYS = 95
REASON_NO_FACTS = "NO_FACTS"
REASON_EXTRACT_EXCEPTION = "EXTRACT_EXCEPTION"
REASON_BANK_SPECIFIC_HANDLING = "BANK_SPECIFIC_HANDLING"
_USD_TO_MUSD = 1_000_000.0


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float))


def _to_num(value: Any) -> float | str:
    return float(value) if _is_num(value) else UNKNOWN


def _unit_is_millions(unit: Any) -> bool:
    """True when an XBRL/derived unit already denominates the value in $millions."""
    token = str(unit or "").strip().lower()
    return token in {"usdm", "usd millions", "usd_millions", "musd"}


def _to_musd(value: Any, unit: Any = None) -> float | str:
    """Convert a balance to $millions.

    FIX 2: raw XBRL ``USD`` facts are reported in whole dollars, so they are divided
    by 1e6 UNCONDITIONALLY (mirroring app.ingest.companyfacts._to_millions). The prior
    abs(value) >= 100_000 threshold mis-handled genuine micro-cap balances — e.g. a
    real $50,000 cash position (val=50000.0, unit="USD") was passed through as if it
    were already $50,000 millions. The XBRL unit is now carried through so values that
    are *already* in millions (unit="USDm", e.g. the synthetic estimated-zero fact)
    are left untouched.
    """
    if not _is_num(value):
        return UNKNOWN
    numeric = float(value)
    if _unit_is_millions(unit):
        return numeric
    return numeric / _USD_TO_MUSD


def _safe_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _dedupe_refs(refs: list[str]) -> list[str]:
    out: list[str] = []
    for ref in refs:
        token = str(ref).strip()
        if token and token not in out:
            out.append(token)
    return out


def _load_companyfacts_payload(facts_row: dict[str, Any]) -> dict[str, Any]:
    cache_path = str(facts_row.get("cache_path") or "").strip()
    if not cache_path:
        return {}
    payload = _safe_json(Path(cache_path))
    if isinstance(payload.get("companyfacts"), dict):
        return payload.get("companyfacts")  # type: ignore[return-value]
    if isinstance(payload.get("facts"), dict):
        return payload  # type: ignore[return-value]
    return {}


def _fact_summary(fact: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(fact, dict):
        return {
            "value": UNKNOWN,
            "unit": None,
            "tag": None,
            "date": None,
            "period_end": None,
            "filed_date": None,
            "confidence": CONFIDENCE_LOW,
            "estimated_zero": False,
            "derived_from": [],
            "source_refs": [],
        }
    refs = [str(ref) for ref in (fact.get("derived_from") or []) if str(ref).strip()]
    period_end = str(fact.get("fact_end_date") or "") or None
    return {
        "value": _to_musd(fact.get("value"), fact.get("unit")),
        "unit": str(fact.get("unit") or "") or None,
        "tag": str(fact.get("tag") or "") or None,
        "date": period_end,
        "period_end": period_end,
        "filed_date": str(fact.get("filed_date") or "") or None,
        "resolution": str(
            fact.get("resolution") or ("RESOLVED" if _is_num(fact.get("value")) else "UNAVAILABLE")
        ),
        "confidence": str(
            fact.get("confidence")
            or (CONFIDENCE_LOW if fact.get("estimated_zero") else CONFIDENCE_HIGH)
        ),
        "estimated_zero": bool(fact.get("estimated_zero")),
        "derived_from": refs,
        "source_refs": refs,
    }


def _lease_fact_for_balance_sheet(
    companyfacts: dict[str, Any], *, period_end: str, as_of_date: str
) -> dict[str, Any] | None:
    """The operating lease for one balance-sheet date, as known at ``as_of_date``.

    Asking the extractor "as of" the balance-sheet date would also move the
    information cutoff back to that date and drop the very filing that reported
    it (a September quarter is filed in November). Only the period is narrowed
    here; the filing cutoff stays the requested as-of date.
    """
    facts = companyfacts.get("facts") if isinstance(companyfacts.get("facts"), dict) else {}
    pruned: dict[str, dict[str, Any]] = {}
    lease_tags = [
        *OPERATING_LEASE_DIRECT_TAG_PRIORITY,
        OPERATING_LEASE_CURRENT_TAG,
        OPERATING_LEASE_NONCURRENT_TAG,
    ]
    for taxonomy, tag in lease_tags:
        taxonomy_node = facts.get(taxonomy) if isinstance(facts, dict) else None
        tag_node = taxonomy_node.get(tag) if isinstance(taxonomy_node, dict) else None
        units = tag_node.get("units") if isinstance(tag_node, dict) else None
        if not isinstance(units, dict):
            continue
        kept_units = {
            unit: [
                row
                for row in rows
                if isinstance(row, dict) and str(row.get("end") or "") <= period_end
            ]
            for unit, rows in units.items()
            if isinstance(rows, list)
        }
        pruned.setdefault(taxonomy, {})[tag] = {"units": kept_units}
    return extract_operating_lease_liability_asof({"facts": pruned}, as_of_date)


def _sic_for_cik(cik: str, *, cfg: AppConfig) -> tuple[str | None, str]:
    """The SEC-filed SIC code for a CIK, and why it is missing when it is.

    Returns ``(sic, reason)``. The reason is what makes the SIC override
    auditable: the lookup used to answer ``None`` for five different situations
    and the classifier silently fell back to the substring rule, so a run could
    have the override switched on and still be classified the old way with
    nothing in the payload saying so.
    """
    token = str(cik or "").strip()
    if not token:
        return None, "NO_CIK"
    # One lookup for the whole codebase: the shared registrant query in
    # app/util/issuer_classification.py (same reason codes), which fetches a missing
    # SIC once from the SEC when the network is enabled.
    return registrant_sic(cik=token, cfg=cfg)


def _infer_companyfacts_issuer_classification(
    companyfacts: dict[str, Any], *, cfg: AppConfig | None = None
) -> tuple[str, str]:
    """Classify the issuer, and report WHICH rule answered.

    Returns ``(classification, source)``. ``source`` is "sic" when the SIC
    override decided, "substring" when the override is off, and
    "substring:<reason>" when the override was ON but no usable SIC was found —
    the case that used to be invisible.
    """
    # SIC-first, on by default (VOE_ISSUER_CLASSIFICATION_BY_SIC, set it to "false" to
    # opt out). The substring classifier below reads tag names and calls 31 of 76
    # large US companies financial (71 of the top 200 liquid US names), which
    # refuses their net-debt bridge and blacks out every valuation downstream. It
    # remains the fallback when no usable SIC code is on file, and the payload's
    # issuer_classification_source says which rule answered.
    if cfg is not None and getattr(cfg, "issuer_classification_by_sic", False):
        sic, reason = _sic_for_cik(str(companyfacts.get("cik") or ""), cfg=cfg)
        sic_class = classify_issuer_by_sic(sic)
        if sic_class is not None:
            return sic_class, "sic"
        if reason == "OK":
            reason = "SIC_UNPARSEABLE"
        logger.warning(
            "net_debt: SIC override on but unresolved for cik=%s (%s); "
            "falling back to the substring classifier",
            str(companyfacts.get("cik") or ""),
            reason,
        )
        fallback_source = f"substring:{reason}"
    else:
        fallback_source = "substring"
    entity_name = str(companyfacts.get("entityName") or "").strip()
    facts = companyfacts.get("facts") if isinstance(companyfacts.get("facts"), dict) else {}
    line_items: list[str] = []
    raw_tag_texts: list[str] = [entity_name]
    for taxonomy in ("us-gaap", "dei"):
        node = facts.get(taxonomy) if isinstance(facts, dict) else {}
        if not isinstance(node, dict):
            continue
        tags = [str(tag) for tag in node.keys() if str(tag).strip()]
        line_items.extend(tags)
        raw_tag_texts.extend(tags)
    return (
        infer_issuer_classification(texts=raw_tag_texts, line_items=line_items),
        fallback_source,
    )


def resolve_net_debt_proxy(
    ticker: str,
    as_of_date: str,
    *,
    run_id: str | None = None,
    facts_row: dict[str, Any] | None = None,
    cfg: AppConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or get_config()
    ticker_norm = str(ticker or "").strip().upper()
    base_facts_row = (
        facts_row
        if isinstance(facts_row, dict)
        else resolve_financial_facts_asof(
            ticker=ticker_norm,
            as_of_date=as_of_date,
            run_id=run_id,
            refresh=False,
            cfg=cfg,
        )
    )
    base_refs = [str(ref) for ref in (base_facts_row.get("derived_from") or []) if str(ref).strip()]
    base_refs = _dedupe_refs(base_refs + [f"facts_coverage.rows[{ticker_norm}]"])
    companyfacts = _load_companyfacts_payload(base_facts_row)

    if not companyfacts:
        return {
            "ticker": ticker_norm,
            "as_of_date": as_of_date,
            "status": STATUS_UNKNOWN,
            "reason_code": REASON_NO_FACTS,
            "total_debt": _fact_summary(None),
            "cash_equivalents": _fact_summary(None),
            "operating_lease_liability": _fact_summary(None),
            "net_debt_proxy": UNKNOWN,
            "net_debt_resolution": "UNAVAILABLE",
            "net_debt_confidence": CONFIDENCE_LOW,
            "issuer_classification": ISSUER_CLASS_OPERATING,
            "issuer_classification_source": "unclassified",
            "derived_from": base_refs,
            "generated_at": utc_now_iso(),
        }

    try:
        issuer_classification, issuer_classification_source = (
            _infer_companyfacts_issuer_classification(companyfacts, cfg=cfg)
        )
        debt_fact = extract_total_debt_asof(companyfacts, as_of_date)
        cash_fact = extract_cash_equivalents_asof(companyfacts, as_of_date)
        lease_fact = extract_operating_lease_liability_asof(companyfacts, as_of_date)
    except Exception:
        return {
            "ticker": ticker_norm,
            "as_of_date": as_of_date,
            "status": STATUS_UNKNOWN,
            "reason_code": REASON_EXTRACT_EXCEPTION,
            "total_debt": _fact_summary(None),
            "cash_equivalents": _fact_summary(None),
            "operating_lease_liability": _fact_summary(None),
            "net_debt_proxy": UNKNOWN,
            "net_debt_resolution": "UNAVAILABLE",
            "net_debt_confidence": CONFIDENCE_LOW,
            "issuer_classification": ISSUER_CLASS_OPERATING,
            "issuer_classification_source": "unclassified",
            "derived_from": base_refs,
            "generated_at": utc_now_iso(),
        }

    # A debt fact older than the cash's balance sheet by more than the allowed gap is not
    # this balance sheet's debt (JPMorgan's freshest undimensioned total debt is from 2014
    # beside 2026 cash): it is dropped, never shown as the total debt, and the result is a
    # named DATELINE_MISMATCH. It never becomes an evidenced zero either: a company that
    # reported debt and then stopped tagging it has not shown it has none.
    stale_debt_flag: str | None = None
    if isinstance(debt_fact, dict) and isinstance(cash_fact, dict):
        stale_end = str(debt_fact.get("fact_end_date") or "")[:10]
        fresh_end = str(cash_fact.get("fact_end_date") or "")[:10]
        try:
            stale_gap = (date.fromisoformat(fresh_end) - date.fromisoformat(stale_end)).days
        except ValueError:
            stale_gap = None
        if stale_gap is not None and stale_gap > _MAX_DEBT_CASH_GAP_DAYS:
            stale_debt_flag = f"STALE_DEBT_DROPPED:debt={stale_end}:cash={fresh_end}"
            debt_fact = None

    # Debt-only evidenced zero: a filer with no debt fact
    # at all has zero debt only when the filing that dates its cash reports no
    # debt concept and its named non-debt liability lines add up to its total
    # liabilities. Missing cash is never evidenced this way.
    debt_evidence: dict[str, Any] | None = None
    if (
        debt_fact is None
        and stale_debt_flag is None
        and isinstance(cash_fact, dict)
        and _is_num(cash_fact.get("value"))
    ):
        cash_end = str(cash_fact.get("fact_end_date") or "")
        try:
            debt_evidence = (
                raw_debt_zero_evidence(companyfacts, period_end=cash_end, as_of_date=as_of_date)
                if cash_end
                else None
            )
        except Exception:
            debt_evidence = None
        if debt_evidence is not None:
            basis = debt_evidence["basis_raw_record"]
            debt_fact = {
                "value": 0.0,
                "unit": "USD",
                "fact_end_date": cash_end,
                "filed_date": str(basis.get("filed_date") or ""),
                "taxonomy": "derived",
                "tag": "EVIDENCED_ZERO_DEBT",
                "resolution": "EVIDENCED_ZERO",
                "derived_from": [
                    f"companyfacts.{basis['taxonomy']}.{basis['concept']}"
                    f"[end_date={cash_end},unit=USD,filed={basis['filed_date']},"
                    f"accn={basis['accession']}]",
                    "derived:total_debt=0(no_debt_concept;complete_liabilities)",
                ],
            }

    debt_summary = _fact_summary(debt_fact)
    cash_summary = _fact_summary(cash_fact)
    # Short-term investments count as cash: read at
    # the cash's own balance-sheet date, never double counted with a combined
    # cash-and-short-term-investments line, recorded as their own component.
    try:
        sti_fact = extract_short_term_investments_asof(
            companyfacts, as_of_date, cash_fact=cash_fact
        )
    except Exception:
        sti_fact = None
    sti_summary: dict[str, Any] | None = None
    if sti_fact is not None:
        sti_summary = {
            **_fact_summary(sti_fact),
            "tags": list(sti_fact.get("tags") or []),
            "derivation": sti_fact.get("derivation"),
        }
    sti_value = float(sti_summary["value"]) if sti_summary is not None else 0.0
    # The three operands are each resolved as "the freshest fact at or before the
    # as-of date", independently, so the lease could come off a different balance
    # sheet than the debt and cash it is added to. Ask for the lease as of the
    # DEBT's own balance-sheet date first; keep the freshest one when no lease was
    # filed for that date, and say so rather than adding it silently.
    lease_dateline_flag: str | None = None
    debt_period_end = str(debt_summary.get("period_end") or "")
    lease_period_end = str((_fact_summary(lease_fact)).get("period_end") or "")
    if debt_period_end and lease_period_end and lease_period_end != debt_period_end:
        try:
            aligned_lease = _lease_fact_for_balance_sheet(
                companyfacts, period_end=debt_period_end, as_of_date=as_of_date
            )
        except Exception:
            aligned_lease = None
        aligned_end = str((_fact_summary(aligned_lease)).get("period_end") or "")
        if aligned_lease is not None and aligned_end == debt_period_end:
            lease_fact = aligned_lease
        else:
            lease_dateline_flag = (
                f"LEASE_DATELINE_MISMATCH:lease={lease_period_end}:debt={debt_period_end}"
            )
    lease_summary = _fact_summary(lease_fact)
    debt_value = debt_summary["value"]
    cash_value = cash_summary["value"]
    # FIX 1: include operating lease liability (ASC 842) so the as-of proxy uses the
    # SAME net-debt definition as the inline valuation_writer path (LEASE_ADJUSTED).
    # An unreadable lease contributes nothing, but it is NOT a filed zero: the
    # lease-inclusive proxy is then really lease-exclusive and says so
    # (LEASE_LIABILITY_UNKNOWN below), instead of quietly reporting a leverage basis
    # it did not measure. Whether such a proxy may keep HIGH confidence is a
    # separate contract question; only the silence is closed here.
    lease_readable = _is_num(lease_summary["value"])
    lease_value = lease_summary["value"] if lease_readable else 0.0
    debt_missing = not _is_num(debt_value)
    cash_missing = not _is_num(cash_value)
    net_debt_confidence = (
        CONFIDENCE_HIGH if not debt_missing and not cash_missing else CONFIDENCE_LOW
    )
    # Debt and cash must come off the same balance sheet. A different date is named
    # (DATELINE_MISMATCH:cash=..:debt=..) and caps confidence below HIGH; more than
    # _MAX_DEBT_CASH_GAP_DAYS apart is refused outright rather than netted.
    dateline_flag: str | None = None
    dateline_refused = False
    debt_end = str(debt_summary.get("period_end") or "")
    cash_end_str = str(cash_summary.get("period_end") or "")
    if not debt_missing and not cash_missing and debt_end and cash_end_str and debt_end != cash_end_str:
        dateline_flag = f"DATELINE_MISMATCH:cash={cash_end_str}:debt={debt_end}"
        try:
            gap_days = abs((date.fromisoformat(debt_end[:10]) - date.fromisoformat(cash_end_str[:10])).days)
        except ValueError:
            gap_days = None
        # An unparseable date cannot be shown to be within the gap: refuse.
        dateline_refused = gap_days is None or gap_days > _MAX_DEBT_CASH_GAP_DAYS
        if net_debt_confidence == CONFIDENCE_HIGH:
            net_debt_confidence = CONFIDENCE_MEDIUM

    if issuer_classification == ISSUER_CLASS_FINANCIAL:
        derived = _dedupe_refs(
            base_refs
            + debt_summary["derived_from"]
            + cash_summary["derived_from"]
            + lease_summary["derived_from"]
        )
        return {
            "ticker": ticker_norm,
            "as_of_date": as_of_date,
            "status": STATUS_UNKNOWN,
            "reason_code": REASON_BANK_SPECIFIC_HANDLING,
            "issuer_classification": issuer_classification,
            "issuer_classification_source": issuer_classification_source,
            "total_debt": debt_summary,
            "cash_equivalents": cash_summary,
            "operating_lease_liability": lease_summary,
            "net_debt_proxy": UNKNOWN,
            "net_debt_resolution": "UNAVAILABLE",
            "net_debt_confidence": net_debt_confidence,
            "net_debt_flags": [stale_debt_flag] if stale_debt_flag else [],
            "derived_from": derived,
            "generated_at": utc_now_iso(),
        }

    net_debt_flags: list[str] = []
    # net_debt_proxy stays lease-INCLUSIVE (leverage/solvency basis);
    # net_debt_proxy_lease_exclusive is the valuation-bridge basis, because the
    # DCF/EPV flow bases are already rent-burdened under ASC 842 and charging
    # the lease again as a debt stock would double-count it.
    net_debt_proxy_lease_exclusive: float | str = UNKNOWN
    if stale_debt_flag is not None:
        status = STATUS_UNKNOWN
        reason_code = REASON_DATELINE_MISMATCH
        net_debt_proxy = UNKNOWN
        resolution = "UNAVAILABLE"
        net_debt_confidence = CONFIDENCE_LOW
        net_debt_flags.append(stale_debt_flag)
    elif dateline_refused:
        status = STATUS_UNKNOWN
        reason_code = REASON_DATELINE_MISMATCH
        net_debt_proxy = UNKNOWN
        resolution = "UNAVAILABLE"
        net_debt_confidence = CONFIDENCE_LOW
        net_debt_flags.append(dateline_flag or REASON_DATELINE_MISMATCH)
    elif not debt_missing and not cash_missing:
        status = STATUS_OK
        reason_code = REASON_OK
        net_debt_proxy: float | str = (
            float(debt_value) + float(lease_value) - float(cash_value) - sti_value
        )
        net_debt_proxy_lease_exclusive = float(debt_value) - float(cash_value) - sti_value
        resolution = "DERIVED"
        if sti_summary is not None:
            net_debt_flags.append("SHORT_TERM_INVESTMENTS_INCLUDED")
        if dateline_flag:
            net_debt_flags.append(dateline_flag)
        if not lease_readable:
            net_debt_flags.append("LEASE_LIABILITY_UNKNOWN")
            # The lease-inclusive proxy is really lease-exclusive: not a measured
            # basis, so not HIGH confidence.
            if net_debt_confidence == CONFIDENCE_HIGH:
                net_debt_confidence = CONFIDENCE_MEDIUM
        elif lease_value > 0:
            net_debt_flags.append("LEASE_ADJUSTED")
        if lease_dateline_flag:
            net_debt_flags.append(lease_dateline_flag)
        if debt_evidence is not None:
            net_debt_flags.append("DEBT_EVIDENCED_ZERO")
    elif debt_missing and cash_missing:
        status = STATUS_UNKNOWN
        reason_code = REASON_MISSING_BOTH
        net_debt_proxy = UNKNOWN
        resolution = "UNAVAILABLE"
    elif debt_missing:
        status = STATUS_UNKNOWN
        reason_code = REASON_MISSING_DEBT
        net_debt_proxy = UNKNOWN
        resolution = "UNAVAILABLE"
    else:
        # A provenance-backed semantic zero proves only the debt operand.
        # Cash remains a separate required input; omitting it would turn
        # "unknown cash" into zero and fabricate enterprise value.
        status = STATUS_UNKNOWN
        reason_code = REASON_MISSING_CASH
        net_debt_proxy = UNKNOWN
        resolution = "UNAVAILABLE"

    derived = _dedupe_refs(
        base_refs
        + debt_summary["derived_from"]
        + cash_summary["derived_from"]
        + (sti_summary["derived_from"] if sti_summary is not None and status == STATUS_OK else [])
        + lease_summary["derived_from"]
    )
    return {
        "ticker": ticker_norm,
        "as_of_date": as_of_date,
        "status": status,
        "reason_code": reason_code,
        "issuer_classification": issuer_classification,
        "issuer_classification_source": issuer_classification_source,
        "total_debt": debt_summary,
        "cash_equivalents": cash_summary,
        "operating_lease_liability": lease_summary,
        **(
            {"short_term_investments": sti_summary}
            if sti_summary is not None and status == STATUS_OK
            else {}
        ),
        "net_debt_proxy": _to_num(net_debt_proxy),
        "net_debt_proxy_lease_exclusive": _to_num(net_debt_proxy_lease_exclusive),
        "net_debt_resolution": resolution,
        "net_debt_confidence": net_debt_confidence,
        "net_debt_flags": net_debt_flags,
        **({"total_debt_evidence": debt_evidence} if debt_evidence is not None else {}),
        "derived_from": derived,
        "generated_at": utc_now_iso(),
    }


def _unique_tickers(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        token = str(item or "").strip().upper()
        if token and token not in seen:
            seen.add(token)
            out.append(token)
    return sorted(out)


def write_net_debt_coverage_for_run(
    *,
    run_id: str,
    as_of_date: str,
    tickers: list[str],
    output_path: Path,
    facts_rows_by_ticker: dict[str, dict[str, Any]] | None = None,
    resolved_by_ticker: dict[str, dict[str, Any]] | None = None,
    cfg: AppConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or get_config()
    rows: list[dict[str, Any]] = []
    for ticker in _unique_tickers(tickers):
        resolved = (resolved_by_ticker or {}).get(ticker)
        if not isinstance(resolved, dict):
            resolved = resolve_net_debt_proxy(
                ticker=ticker,
                as_of_date=as_of_date,
                run_id=run_id,
                facts_row=(facts_rows_by_ticker or {}).get(ticker),
                cfg=cfg,
            )
        rows.append(
            {
                "ticker": ticker,
                "as_of_date": as_of_date,
                "status": str(resolved.get("status") or STATUS_UNKNOWN),
                "reason_code": str(resolved.get("reason_code") or REASON_EXTRACT_EXCEPTION),
                "total_debt_value": resolved.get("total_debt", {}).get("value", UNKNOWN)
                if isinstance(resolved.get("total_debt"), dict)
                else UNKNOWN,
                "cash_equivalents_value": resolved.get("cash_equivalents", {}).get("value", UNKNOWN)
                if isinstance(resolved.get("cash_equivalents"), dict)
                else UNKNOWN,
                "net_debt_proxy_value": resolved.get("net_debt_proxy", UNKNOWN),
                "net_debt_resolution": str(resolved.get("net_debt_resolution") or "UNAVAILABLE"),
                # A row that carries no confidence of its own has no usable facts
                # behind it; it must not read as HIGH by default.
                "net_debt_confidence": str(resolved.get("net_debt_confidence") or CONFIDENCE_LOW),
                "issuer_classification": str(
                    resolved.get("issuer_classification") or ISSUER_CLASS_OPERATING
                ),
                "tags_used": {
                    "debt_tag": resolved.get("total_debt", {}).get("tag")
                    if isinstance(resolved.get("total_debt"), dict)
                    else None,
                    "cash_tag": resolved.get("cash_equivalents", {}).get("tag")
                    if isinstance(resolved.get("cash_equivalents"), dict)
                    else None,
                },
                "fact_dates_used": {
                    "debt_date": resolved.get("total_debt", {}).get("date")
                    if isinstance(resolved.get("total_debt"), dict)
                    else None,
                    "cash_date": resolved.get("cash_equivalents", {}).get("date")
                    if isinstance(resolved.get("cash_equivalents"), dict)
                    else None,
                },
                "derived_from": [
                    str(ref) for ref in (resolved.get("derived_from") or []) if str(ref).strip()
                ],
            }
        )

    rows = sorted(rows, key=lambda row: str(row.get("ticker") or ""))
    status_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status") or STATUS_UNKNOWN).upper()
        reason = str(row.get("reason_code") or REASON_EXTRACT_EXCEPTION).upper()
        status_counts[status] = status_counts.get(status, 0) + 1
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    payload = {
        "run_id": run_id,
        "as_of_date": as_of_date,
        "ticker_count": len(rows),
        "status_counts": dict(sorted(status_counts.items(), key=lambda kv: kv[0])),
        "reason_counts": dict(sorted(reason_counts.items(), key=lambda kv: kv[0])),
        "entries": rows,
        "generated_at": utc_now_iso(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["net_debt_coverage_path"] = str(output_path)
    return payload
