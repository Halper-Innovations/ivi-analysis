"""Cheapness-explanation pass: known reasons a name may be cheap, at queue time.

Before a name renders at/below target in any queue or brief, this pass
attaches a "known reasons it may be cheap" block built from:

a. Deterministic evidence — the trailing-12mo EDGAR filing index with 8-K
   item codes; Legal Proceedings and Commitments & Contingencies sections
   from the latest 10-K/10-Q; and an XBRL mezzanine check (redeemable
   noncontrolling interest / redeemable preferred carrying amounts vs market
   cap, MEZZANINE_OBLIGATION above 25% of cap).
b. A cheap-LLM summary — at most 3 bullets naming candidate explanations,
   each citing filings by accession number, or NO_KNOWN_EVENT when the
   filing record is clean. Budget-capped, cached by a fingerprint of the
   trailing accession set, re-run only when a new filing lands.

Deterministic findings outrank the LLM: a fired flag or an open
EVENT_PENDING event forces the verdict to KNOWN_EVENTS, and bullets whose
citations are not in the filing index are dropped.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from datetime import date, datetime, timedelta, timezone
from typing import Any

from app.autonomous.financial_integrity import (
    FinancialIntegrityScope,
    InvalidFinancialInputError,
    require_unchanged_financial_integrity_scope,
)
from app.autonomous.sector_runtime import _synthesize_provider_json_with_meta
from app.autonomous.v1_financial_context import (
    BoundV1FinancialScope,
    bind_v1_financial_scope,
    build_canonical_v1_financial_context,
    financial_input_scenario,
)
from app.events import store
from app.events.submissions_adapter import SecSubmissionsAdapter
from app.llm.providers.retry_guard import llm_cost_budget

MEZZANINE_THRESHOLD = 0.25
# A mezzanine concept whose latest instant is older than this is treated as
# extinguished (de-SPAC'd temporary equity, redeemed preferred), not a live
# claim — companyfacts keeps last-ever values forever. One filing cycle +
# lag + buffer, matching the cap chain's stale-shares window.
MEZZANINE_MAX_AGE_DAYS = 400
FLAG_MEZZANINE = "MEZZANINE_OBLIGATION"
NO_KNOWN_EVENT = "NO_KNOWN_EVENT"
KNOWN_EVENTS = "KNOWN_EVENTS"
LLM_UNAVAILABLE = "LLM_UNAVAILABLE"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CHEAPNESS_PUBLICATION_SCHEMA = "cheapness_publication_v1"
_CHEAPNESS_LLM_USAGE_SCHEMA = "cheapness_llm_usage_v1"
_CHEAPNESS_ROW_FIELDS = (
    "ticker",
    "cik",
    "filings_fingerprint",
    "financial_scope_fingerprint",
    "financial_scope_publication_fingerprint",
    "publication_source_sha256",
    "paid_attempt_id",
    "as_of",
    "flags_json",
    "deterministic_json",
    "llm_verdict",
    "llm_bullets_json",
    "llm_model",
    "llm_usage_json",
    "llm_cost_usd",
    "llm_budget_usd",
)


class CheapnessPaidAttemptAmbiguousError(RuntimeError):
    """A prior physical cheapness call cannot safely be repeated."""


class CheapnessCostIntegrityError(RuntimeError):
    """Provider-reported cost exceeded the conservative paid reservation."""


# Carrying-amount concepts for redeemable instruments parked between
# liabilities and equity. The umbrella concept (temporary equity including
# the NCI portion) is preferred when present; otherwise the parent temporary
# equity and redeemable-NCI concepts are summed (they are disjoint).
MEZZANINE_UMBRELLA_CONCEPTS = (
    "TemporaryEquityCarryingAmountIncludingPortionAttributableToNoncontrollingInterests",
)
MEZZANINE_COMPONENT_CONCEPTS = (
    "TemporaryEquityCarryingAmountAttributableToParent",
    "RedeemableNoncontrollingInterestEquityCarryingAmount",
    "RedeemableNoncontrollingInterestEquityCommonCarryingAmount",
    "RedeemableNoncontrollingInterestEquityPreferredCarryingAmount",
)

_SECTION_EXCERPT_CHARS = 3500

CHEAPNESS_SUMMARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": [NO_KNOWN_EVENT, KNOWN_EVENTS]},
        "bullets": {
            "type": "array",
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "accessions": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["text", "accessions"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["verdict", "bullets"],
    "additionalProperties": False,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _require_cheapness_scope_unchanged(
    authorized_scope: BoundV1FinancialScope,
    current_scope: BoundV1FinancialScope,
) -> str:
    current_result = current_scope.require()
    expected = str(authorized_scope.expected_scope_fingerprint or "")
    observed = str(current_scope.expected_scope_fingerprint or "")
    if expected and observed == expected:
        return observed
    current = FinancialIntegrityScope(
        context=getattr(current_scope, "context", "cheapness_publication"),
        run_as_of_date=getattr(current_scope, "run_as_of_date", ""),
        packets=current_scope.packets,
        scenarios=getattr(current_scope, "scenarios", ()),
    )
    result = require_unchanged_financial_integrity_scope(
        current,
        expected_scope_fingerprint=expected,
    )
    return str(getattr(result, "scope_fingerprint", current_result.scope_fingerprint))


def _cheapness_publication_source_payload(
    *,
    ticker: str,
    as_of: str,
    scope_fingerprint: str,
    filings_fingerprint_value: str,
    provider_prompt: str,
) -> dict[str, Any]:
    return {
        "schema": _CHEAPNESS_PUBLICATION_SCHEMA,
        "ticker": ticker.upper(),
        "as_of": as_of,
        "financial_scope_fingerprint": scope_fingerprint,
        "filings_fingerprint": filings_fingerprint_value,
        "provider_prompt": provider_prompt,
    }


def _cheapness_publication_source_sha256(
    *,
    ticker: str,
    as_of: str,
    scope_fingerprint: str,
    filings_fingerprint_value: str,
    provider_prompt: str,
) -> str:
    return _canonical_sha256(
        _cheapness_publication_source_payload(
            ticker=ticker,
            as_of=as_of,
            scope_fingerprint=scope_fingerprint,
            filings_fingerprint_value=filings_fingerprint_value,
            provider_prompt=provider_prompt,
        )
    )


def _cheapness_publication_row_payload(row: Any) -> dict[str, Any] | None:
    payload: dict[str, Any] = {}
    for field in _CHEAPNESS_ROW_FIELDS:
        try:
            value = row[field]
        except (KeyError, IndexError):
            return None
        if field in {
            "flags_json",
            "deterministic_json",
            "llm_bullets_json",
            "llm_usage_json",
        }:
            try:
                value = json.loads(value)
                _canonical_json(value)
            except (TypeError, ValueError, json.JSONDecodeError):
                return None
        payload[field] = value
    return payload


def _cheapness_publication_row_sha256(row: Any) -> str | None:
    payload = _cheapness_publication_row_payload(row)
    return _canonical_sha256(payload) if payload is not None else None


def _cheapness_row_is_authorized(conn: sqlite3.Connection, row: Any) -> bool:
    try:
        paid = str(row["financial_scope_fingerprint"] or "").strip().lower()
        publication = str(row["financial_scope_publication_fingerprint"] or "").strip().lower()
        source_sha = str(row["publication_source_sha256"] or "").strip().lower()
        row_sha = str(row["publication_row_sha256"] or "").strip().lower()
    except (KeyError, IndexError):
        return False
    expected_row_sha = _cheapness_publication_row_sha256(row)
    row_is_self_consistent = (
        _SHA256_RE.fullmatch(paid) is not None
        and publication == paid
        and _SHA256_RE.fullmatch(source_sha) is not None
        and _SHA256_RE.fullmatch(row_sha) is not None
        and expected_row_sha is not None
        and row_sha == expected_row_sha
    )
    if not row_is_self_consistent:
        return False
    try:
        authorization = conn.execute(
            """
            SELECT publication_row_sha256, financial_scope_fingerprint,
                   publication_source_sha256, publication_evidence_json
            FROM cheapness_publication_authorizations
            WHERE ticker = ?
              AND filings_fingerprint = ?
            ORDER BY authorization_id DESC
            LIMIT 1
            """,
            (
                str(row["ticker"]).strip().upper(),
                str(row["filings_fingerprint"]),
            ),
        ).fetchone()
    except sqlite3.OperationalError:
        return False
    if authorization is None:
        return False
    if (
        str(authorization["publication_row_sha256"] or "").strip().lower() != row_sha
        or str(authorization["financial_scope_fingerprint"] or "").strip().lower() != paid
        or str(authorization["publication_source_sha256"] or "").strip().lower() != source_sha
    ):
        return False
    try:
        evidence = json.loads(authorization["publication_evidence_json"])
    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
        return False
    if not isinstance(evidence, dict) or set(evidence) != {
        "schema",
        "ticker",
        "as_of",
        "financial_scope_fingerprint",
        "filings_fingerprint",
        "provider_prompt",
    }:
        return False
    return (
        evidence.get("schema") == _CHEAPNESS_PUBLICATION_SCHEMA
        and evidence.get("ticker") == str(row["ticker"]).strip().upper()
        and evidence.get("as_of") == row["as_of"]
        and evidence.get("financial_scope_fingerprint") == paid
        and evidence.get("filings_fingerprint") == row["filings_fingerprint"]
        and isinstance(evidence.get("provider_prompt"), str)
        and _canonical_sha256(evidence) == source_sha
    )


def filings_fingerprint(filing_index: list[dict[str, str]]) -> str:
    accessions = sorted(f["accession"] for f in filing_index)
    return hashlib.sha256("|".join(accessions).encode("utf-8")).hexdigest()


def cik_for_ticker(conn: sqlite3.Connection, ticker: str) -> str | None:
    ticker = ticker.upper()
    try:
        row = conn.execute(
            "SELECT cik FROM sec_registrants WHERE primary_ticker = ?", (ticker,)
        ).fetchone()
        if row is not None:
            return str(row["cik"]).zfill(10)
    except sqlite3.OperationalError:
        pass
    try:
        from app.universe.ticker_cik_map import load_ticker_cik_map

        cik = load_ticker_cik_map().get(ticker)
        return str(cik).zfill(10) if cik else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Deterministic leg 1: trailing-12mo filing index with item codes
# ---------------------------------------------------------------------------


def trailing_filing_index(
    adapter: SecSubmissionsAdapter, cik: str, *, as_of: str, days: int = 365
) -> list[dict[str, str]] | None:
    start = (date.fromisoformat(as_of) - timedelta(days=days)).isoformat()
    return adapter.filings_window(cik, start=start, end=as_of)


# ---------------------------------------------------------------------------
# Deterministic leg 2: Legal Proceedings + Commitments & Contingencies
# ---------------------------------------------------------------------------


def _cheapness_patterns(form: str) -> dict:
    from app.dossier.sections import (
        COMMITMENTS_CONTINGENCIES_PATTERNS,
        LEGAL_PROCEEDINGS_PATTERNS,
        LEGAL_PROCEEDINGS_PATTERNS_10Q,
        SECTION_PATTERNS,
        SECTION_PATTERNS_10Q,
    )

    if form.startswith("10-Q"):
        base = dict(SECTION_PATTERNS_10Q)
        base["legal_proceedings"] = LEGAL_PROCEEDINGS_PATTERNS_10Q
    else:
        base = dict(SECTION_PATTERNS)
        base["legal_proceedings"] = LEGAL_PROCEEDINGS_PATTERNS
    base["commitments_contingencies"] = COMMITMENTS_CONTINGENCIES_PATTERNS
    return base


def _primary_doc_text(client, cik: str, filing: dict[str, str]) -> str | None:
    from app.dossier.filing_cache import filing_cache_path

    cik_plain = str(int(cik))
    accession = filing["accession"]
    accession_nodash = accession.replace("-", "")
    candidates = [
        filing_cache_path(cik=cik_plain, accession=accession),
        filing_cache_path(cik=cik_plain, accession=accession_nodash),
    ]
    for path in candidates:
        if path.exists() and path.is_file():
            return path.read_text(encoding="utf-8", errors="ignore")
    doc_name = filing.get("primary_document") or "document.html"
    url = f"https://www.sec.gov/Archives/edgar/data/{cik_plain}/{accession_nodash}/{doc_name}"
    try:
        payload = client.download_bytes(url)
    except Exception:
        return None
    target = candidates[1]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return payload.decode("utf-8", errors="ignore")


def extract_risk_sections(
    client, cik: str, filing_index: list[dict[str, str]]
) -> dict[str, dict[str, str]]:
    """legal_proceedings / commitments_contingencies from the latest 10-K/10-Q."""
    from app.dossier.sections import section_by_label, segment_sections
    from app.util.html_strip import strip_html

    out: dict[str, dict[str, str]] = {}
    latest: dict[str, dict[str, str]] = {}
    for filing in filing_index:  # newest first
        form = filing["form"]
        if form.startswith("10-K") and "10-K" not in latest:
            latest["10-K"] = filing
        elif form.startswith("10-Q") and "10-Q" not in latest:
            latest["10-Q"] = filing
    for form, filing in latest.items():
        text = _primary_doc_text(client, cik, filing)
        if not text:
            continue
        spans = segment_sections(text, _cheapness_patterns(form))
        for label in ("legal_proceedings", "commitments_contingencies"):
            span = section_by_label(spans, label)
            if span is None or label in out:
                continue
            excerpt = strip_html(span.text)[:_SECTION_EXCERPT_CHARS]
            out[label] = {
                "source_form": form,
                "accession": filing["accession"],
                "filing_date": filing["filing_date"],
                "excerpt": excerpt,
            }
    return out


# ---------------------------------------------------------------------------
# Deterministic leg 3: XBRL mezzanine check (the BOOM class)
# ---------------------------------------------------------------------------


def _companyfacts_source_url(
    raw_companyfacts: dict[str, Any],
    explicit_source_url: str | None,
) -> str | None:
    explicit = str(explicit_source_url or "").strip()
    if explicit:
        return explicit
    embedded = str(raw_companyfacts.get("source_url") or "").strip()
    if embedded:
        return embedded
    raw_cik = str(raw_companyfacts.get("cik") or "").strip()
    if not raw_cik.isdigit() or len(raw_cik) > 10:
        return None
    return f"https://data.sec.gov/api/xbrl/companyfacts/CIK{raw_cik.zfill(10)}.json"


def _latest_instant(
    concept_payload: dict[str, Any],
    *,
    as_of_date: date,
    source_url: str | None,
) -> tuple[dict[str, Any] | None, tuple[str, ...]]:
    """Select one visible, fully sourced raw-USD instant fact.

    A malformed fact that could have been visible fails the whole concept
    closed. Facts whose valid period or filing date is after ``as_of_date`` are
    simply not yet visible and may not displace an older visible fact.
    """
    units = concept_payload.get("units")
    if not isinstance(units, dict) or "USD" not in units:
        return None, ("MISSING_LITERAL_USD_UNIT",)
    series = units.get("USD")
    if not isinstance(series, list):
        return None, ("MALFORMED_USD_SERIES",)

    candidates: list[dict[str, Any]] = []
    invalid_reasons: set[str] = set()
    for point in series:
        if not isinstance(point, dict):
            invalid_reasons.add("MALFORMED_FACT")
            continue
        end_raw = str(point.get("end") or "").strip()
        try:
            period_end = date.fromisoformat(end_raw)
        except ValueError:
            invalid_reasons.add("MISSING_OR_INVALID_PERIOD_END")
            continue
        if period_end > as_of_date:
            continue

        filed_raw = str(point.get("filed") or "").strip()
        try:
            filed_date = date.fromisoformat(filed_raw)
        except ValueError:
            invalid_reasons.add("MISSING_OR_INVALID_FILED_DATE")
            continue
        if filed_date > as_of_date:
            continue
        if period_end > filed_date:
            invalid_reasons.add("PERIOD_END_AFTER_FILED_DATE")
            continue

        accession = str(point.get("accn") or "").strip()
        value = point.get("val")
        if not accession:
            invalid_reasons.add("MISSING_ACCESSION")
            continue
        if not source_url:
            invalid_reasons.add("MISSING_SOURCE_URL")
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            invalid_reasons.add("MISSING_OR_NONFINITE_VALUE")
            continue
        candidates.append(
            {
                "reported_value": float(value),
                "reported_unit": "USD",
                "period_end": period_end.isoformat(),
                "filed_date": filed_date.isoformat(),
                "accession": accession,
                "source_url": source_url,
            }
        )

    if invalid_reasons:
        return None, tuple(sorted(invalid_reasons))
    if not candidates:
        return None, ()
    latest = max(
        candidates,
        key=lambda candidate: (
            candidate["period_end"],
            candidate["filed_date"],
            candidate["accession"],
        ),
    )
    return latest, ()


def mezzanine_check(
    raw_companyfacts: dict | None,
    market_cap_mm: float | None,
    *,
    as_of: str | None = None,
    source_url: str | None = None,
) -> dict[str, Any]:
    """Redeemable NCI / redeemable preferred carrying amounts vs market cap.

    Only concepts still being reported count: a concept whose latest instant
    is older than MEZZANINE_MAX_AGE_DAYS is an extinguished instrument
    (recorded under stale_components, never summed) — companyfacts keeps a
    de-SPAC's 2018 temporary equity forever.
    """
    result: dict[str, Any] = {
        "flag": None,
        "mezzanine_total_mm": None,
        "market_cap_mm": market_cap_mm,
        "ratio_to_cap": None,
        "components": [],
        "stale_components": [],
        "invalid_components": [],
        "source_references": [],
        "unit": "USD_millions",
        "reason_codes": [],
        "status": "OK",
    }
    if not raw_companyfacts:
        result["status"] = "NO_COMPANYFACTS"
        return result
    effective_as_of = as_of or datetime.now(timezone.utc).date().isoformat()
    try:
        as_of_date = date.fromisoformat(effective_as_of)
    except ValueError:
        result["status"] = "NEEDS_DATA"
        result["reason_codes"] = ["INVALID_AS_OF_DATE"]
        return result
    facts = raw_companyfacts.get("facts")
    gaap = facts.get("us-gaap") if isinstance(facts, dict) else None
    gaap = gaap if isinstance(gaap, dict) else {}
    exact_source_url = _companyfacts_source_url(raw_companyfacts, source_url)
    cutoff = (as_of_date - timedelta(days=MEZZANINE_MAX_AGE_DAYS)).isoformat()

    components: list[dict[str, Any]] = []
    stale_components: list[dict[str, Any]] = []
    invalid_components: list[dict[str, Any]] = []

    def _collect(concepts: tuple[str, ...]) -> float | None:
        total = None
        for concept in concepts:
            payload = gaap.get(concept)
            if not payload:
                continue
            if not isinstance(payload, dict):
                invalid_components.append(
                    {"concept": concept, "reason_codes": ["MALFORMED_CONCEPT"]}
                )
                continue
            latest, invalid_reasons = _latest_instant(
                payload,
                as_of_date=as_of_date,
                source_url=exact_source_url,
            )
            if invalid_reasons:
                invalid_components.append(
                    {
                        "concept": concept,
                        "reason_codes": list(invalid_reasons),
                    }
                )
                continue
            if latest is None:
                continue
            value = float(latest["reported_value"])
            end = str(latest["period_end"])
            accession = str(latest["accession"])
            source_reference = f"{latest['source_url']}#us-gaap:{concept}:accession={accession}"
            entry = {
                "concept": concept,
                "reported_value": value,
                "reported_unit": "USD",
                "value_mm": round(value / 1e6, 3),
                "unit": "USD_millions",
                "as_of": end,
                "period_end": end,
                "filed_date": latest["filed_date"],
                "accession": accession,
                "source_url": latest["source_url"],
                "source_reference": source_reference,
            }
            if end < cutoff:
                stale_components.append(entry)
                continue
            components.append(entry)
            total = (total or 0.0) + value
        return total

    umbrella = _collect(MEZZANINE_UMBRELLA_CONCEPTS)
    component_sum = None
    if umbrella is None:
        component_sum = _collect(MEZZANINE_COMPONENT_CONCEPTS)
    mezzanine = umbrella if umbrella is not None else component_sum
    result["components"] = components
    result["stale_components"] = stale_components
    result["invalid_components"] = invalid_components
    if invalid_components:
        result["components"] = []
        result["stale_components"] = []
        result["status"] = "NEEDS_DATA"
        result["reason_codes"] = sorted(
            {reason for component in invalid_components for reason in component["reason_codes"]}
        )
        return result
    if mezzanine is None:
        result["status"] = "NO_MEZZANINE_CONCEPTS"
        return result
    result["source_references"] = [component["source_reference"] for component in components]
    mezzanine_mm = mezzanine / 1e6
    result["mezzanine_total_mm"] = round(mezzanine_mm, 3)
    if not market_cap_mm or market_cap_mm <= 0:
        result["status"] = "NO_MARKET_CAP"
        return result
    ratio = mezzanine_mm / float(market_cap_mm)
    result["ratio_to_cap"] = round(ratio, 4)
    if ratio > MEZZANINE_THRESHOLD:
        result["flag"] = FLAG_MEZZANINE
    return result


def load_raw_companyfacts(cik: str) -> dict | None:
    from app.market.company_facts_provider import fetch_company_facts

    try:
        payload = fetch_company_facts(cik)
    except Exception:
        return None
    facts = payload.get("companyfacts")
    return facts if isinstance(facts, dict) else None


# ---------------------------------------------------------------------------
# Cheap-LLM summary (analyst lane: every claim cites a filing)
# ---------------------------------------------------------------------------


def _build_prompt(
    ticker: str,
    *,
    as_of: str,
    filing_index: list[dict[str, str]],
    sections: dict[str, dict[str, str]],
    mezzanine: dict[str, Any],
    open_events: list[dict[str, Any]],
) -> str:
    index_lines = [
        f"- {f['filing_date']} {f['form']} accession {f['accession']}"
        + (f" items [{f['items']}]" if f.get("items") else "")
        for f in filing_index[:60]
    ]
    section_blocks = []
    for label, payload in sections.items():
        section_blocks.append(
            f"### {label} (from {payload['source_form']} {payload['accession']}, "
            f"filed {payload['filing_date']})\n{payload['excerpt']}"
        )
    event_lines = [
        f"- OPEN EVENT {e['event_type']} since {e['detection_date']} "
        f"(anchor accession {e['anchor_accession']})"
        for e in open_events
    ]
    if mezzanine.get("status") == "OK":
        mezzanine_prompt = mezzanine
    else:
        # Rejected concepts are audit metadata, not evidence for prose. Keep the
        # complete rejected provenance in deterministic_json/cache scope, but
        # expose only the fail-closed status to the analyst lane.
        mezzanine_prompt = {
            "flag": None,
            "reason_codes": list(mezzanine.get("reason_codes") or []),
            "status": mezzanine.get("status"),
        }
    mezz_line = json.dumps(mezzanine_prompt, sort_keys=True)
    return (
        f"You are the analyst lane of a value-investing watchlist platform. "
        f"Ticker {ticker}, as of {as_of}. The name screens cheap; your job is to "
        "name the KNOWN reasons it may be cheap — disclosed events and "
        "obligations a buyer must reckon with — strictly from the SEC filing "
        "record below. This is not a buy/sell call.\n\n"
        "Rules:\n"
        "- At most 3 bullets, most material first.\n"
        "- Every bullet MUST cite at least one accession number from the "
        "filing index below. No claims without a filing behind them.\n"
        "- If the record shows no candidate explanation, return verdict "
        f"{NO_KNOWN_EVENT} with zero bullets. Do not invent reasons.\n\n"
        "- A deterministic mezzanine status other than OK is unavailable data, "
        "not evidence of an obligation. Never infer overhang prose from rejected "
        "or missing facts.\n\n"
        f"## Trailing-12mo filing index\n"
        + "\n".join(index_lines)
        + "\n\n"
        + (("## Open corporate events\n" + "\n".join(event_lines) + "\n\n") if event_lines else "")
        + f"## Mezzanine check (deterministic)\n{mezz_line}\n\n"
        + ("\n\n".join(section_blocks) if section_blocks else "(no risk sections extracted)")
    )


def _rebuild_cheapness_publication_state(
    *,
    ticker: str,
    cik: str,
    as_of: str,
    conn: sqlite3.Connection,
    client: Any,
    adapter: SecSubmissionsAdapter,
    raw_companyfacts_snapshot: dict[str, Any] | None,
    reload_companyfacts: bool,
    financial_integrity_scope: BoundV1FinancialScope | None,
) -> dict[str, Any]:
    filing_index = trailing_filing_index(adapter, cik, as_of=as_of)
    current_filing_index = filing_index if filing_index is not None else []
    if financial_integrity_scope is None:
        from app.config import get_config

        context = build_canonical_v1_financial_context(
            tickers=[ticker],
            as_of_date=as_of,
            db_path=get_config().db_path,
        )
        packet = context.packets[ticker]
    else:
        financial_integrity_scope.require()
        packet = financial_integrity_scope.packets[0]
    packet_payload = dict(packet) if isinstance(packet, dict) else dict(vars(packet))
    market_cap_mm = packet_payload.get("market_cap_mm")
    raw_companyfacts = (
        load_raw_companyfacts(cik) if reload_companyfacts else raw_companyfacts_snapshot
    )
    mezzanine = mezzanine_check(
        raw_companyfacts,
        market_cap_mm,
        as_of=as_of,
        source_url=f"https://data.sec.gov/api/xbrl/companyfacts/CIK{str(cik).zfill(10)}.json",
    )
    sections = extract_risk_sections(client, cik, current_filing_index)
    open_events = [
        dict(row) for row in store.open_queue_protection_events(conn, ciks=[cik]).get(cik, [])
    ]
    prompt = _build_prompt(
        ticker,
        as_of=as_of,
        filing_index=current_filing_index,
        sections=sections,
        mezzanine=mezzanine,
        open_events=open_events,
    )
    scenario = financial_input_scenario(
        packet,
        financial_inputs={
            "market_cap_mm": market_cap_mm,
            "market_cap_unit": packet_payload.get("market_cap_unit"),
            "mezzanine": mezzanine,
            "provider_prompt": prompt,
        },
    )
    scope = bind_v1_financial_scope(
        context=f"cheapness:{ticker}:{as_of}",
        run_as_of_date=as_of,
        packets=(packet,),
        scenarios=(scenario,),
    )
    flags: list[str] = []
    if mezzanine["flag"]:
        flags.append(mezzanine["flag"])
    flags.extend(sorted({store.event_pending_flag(event["event_type"]) for event in open_events}))
    return {
        "packet": packet,
        "packet_payload": packet_payload,
        "market_cap_mm": market_cap_mm,
        "filing_index": current_filing_index,
        "filings_fingerprint": filings_fingerprint(current_filing_index),
        "sections": sections,
        "mezzanine": mezzanine,
        "open_events": open_events,
        "prompt": prompt,
        "scenario": scenario,
        "scope": scope,
        "flags": flags,
    }


def _cheapness_attempt_request(
    provider: Any,
    *,
    ticker: str,
    cik: str,
    as_of: str,
    filings_fingerprint_value: str,
    scope_fingerprint: str,
    prompt: str,
) -> dict[str, Any]:
    from app.autonomous import sector_runtime as sector_runtime_module

    provider_name = sector_runtime_module._provider_name(provider)
    provider_kwargs: dict[str, Any] = {
        "prompt": prompt,
        "schema": CHEAPNESS_SUMMARY_SCHEMA,
        "schema_name": "cheapness_summary_v1",
        "max_output_tokens": 1000,
    }
    if provider_name == "openai":
        provider_kwargs["allow_output_token_retry"] = False
    request_payload = {
        "schema": "cheapness_paid_attempt_request_v1",
        "ticker": ticker,
        "cik": cik,
        "as_of": as_of,
        "filings_fingerprint": filings_fingerprint_value,
        "financial_scope_fingerprint": scope_fingerprint,
        "provider": provider_name,
        "model": sector_runtime_module._provider_model_for_estimate(provider),
        "provider_kwargs": provider_kwargs,
    }
    request_sha256 = hashlib.sha256(_canonical_json(request_payload).encode("utf-8")).hexdigest()
    attempt_id = hashlib.sha256(
        _canonical_json(
            {
                "schema": "cheapness_paid_attempt_identity_v1",
                "ticker": ticker,
                "as_of": as_of,
                "filings_fingerprint": filings_fingerprint_value,
                "financial_scope_fingerprint": scope_fingerprint,
                "request_sha256": request_sha256,
            }
        ).encode("utf-8")
    ).hexdigest()
    return {
        "attempt_id": attempt_id,
        "request_sha256": request_sha256,
        "provider": provider_name,
        "model": request_payload["model"],
        "provider_kwargs": provider_kwargs,
        "estimated_cost_usd": float(
            sector_runtime_module._estimated_llm_call_cost_usd(
                provider,
                provider_kwargs,
            )
        ),
    }


def _reserve_cheapness_paid_attempt(
    conn: sqlite3.Connection,
    *,
    attempt_request: dict[str, Any],
    ticker: str,
    cik: str,
    as_of: str,
    filings_fingerprint_value: str,
    scope_fingerprint: str,
) -> tuple[dict[str, Any] | None, str]:
    # A provider call cannot be crash-durable inside a caller-owned uncommitted
    # transaction. Commit deterministic setup first, then reserve in its own
    # immediate transaction before any physical egress.
    if conn.in_transaction:
        conn.commit()
    conn.execute("BEGIN IMMEDIATE")
    try:
        existing = conn.execute(
            "SELECT * FROM cheapness_paid_attempts WHERE attempt_id = ?",
            (str(attempt_request["attempt_id"]),),
        ).fetchone()
        if existing is not None:
            status = str(existing["status"])
            if status == "RESERVED":
                raise CheapnessPaidAttemptAmbiguousError(
                    f"Cheapness {ticker}/{as_of} has a RESERVED paid attempt; "
                    "retry is blocked because the physical-call outcome is ambiguous"
                )
            if status == "REJECTED":
                raise CheapnessPaidAttemptAmbiguousError(
                    f"Cheapness {ticker}/{as_of} has a rejected paid response for "
                    "these exact inputs; retry is blocked to prevent duplicate spend"
                )
            raw_result = existing["result_json"]
            if raw_result is None:
                raise CheapnessPaidAttemptAmbiguousError(
                    f"Cheapness {ticker}/{as_of} paid attempt lacks a reusable result"
                )
            conn.commit()
            parsed = json.loads(str(raw_result))
            if not isinstance(parsed, dict):
                raise CheapnessPaidAttemptAmbiguousError(
                    f"Cheapness {ticker}/{as_of} paid attempt result is invalid"
                )
            return dict(parsed), status
        conn.execute(
            """
            INSERT INTO cheapness_paid_attempts (
                attempt_id, ticker, cik, as_of, filings_fingerprint,
                financial_scope_fingerprint, request_sha256, provider, model,
                status, estimated_cost_usd, reserved_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'RESERVED', ?, ?)
            """,
            (
                str(attempt_request["attempt_id"]),
                ticker,
                cik,
                as_of,
                filings_fingerprint_value,
                scope_fingerprint,
                str(attempt_request["request_sha256"]),
                str(attempt_request["provider"]),
                str(attempt_request["model"]),
                max(0.0, float(attempt_request["estimated_cost_usd"])),
                _now(),
            ),
        )
        conn.commit()
        return None, "RESERVED"
    except BaseException:
        conn.rollback()
        raise


def _complete_cheapness_paid_attempt(
    conn: sqlite3.Connection,
    *,
    attempt_id: str,
    llm: dict[str, Any],
) -> None:
    if conn.in_transaction:
        conn.commit()
    conn.execute("BEGIN IMMEDIATE")
    overrun: tuple[float, float] | None = None
    try:
        row = conn.execute(
            """
            SELECT estimated_cost_usd
            FROM cheapness_paid_attempts
            WHERE attempt_id = ? AND status = 'RESERVED'
            """,
            (attempt_id,),
        ).fetchone()
        if row is None:
            raise CheapnessPaidAttemptAmbiguousError(
                "Cheapness paid-attempt reservation changed before completion"
            )
        estimated_cost_usd = max(0.0, float(row["estimated_cost_usd"]))
        accounted_cost_usd = max(0.0, float(llm.get("cost_usd") or 0.0))
        if accounted_cost_usd > estimated_cost_usd + 1e-12:
            overrun = (estimated_cost_usd, accounted_cost_usd)
        conn.execute(
            """
            UPDATE cheapness_paid_attempts
            SET status = ?, outcome = ?, accounted_cost_usd = ?,
                result_json = ?, error = ?, completed_at = ?
            WHERE attempt_id = ? AND status = 'RESERVED'
            """,
            (
                "REJECTED" if overrun is not None else "RETURNED",
                "ERROR" if llm.get("error") else "SUCCESS",
                accounted_cost_usd,
                _canonical_json(llm),
                (
                    "provider-reported cost exceeded conservative reservation: "
                    f"${accounted_cost_usd:.6f} actual > "
                    f"${estimated_cost_usd:.6f} reserved"
                    if overrun is not None
                    else (str(llm.get("error"))[:500] if llm.get("error") else None)
                ),
                _now(),
                attempt_id,
            ),
        )
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    if overrun is not None:
        estimated_cost_usd, accounted_cost_usd = overrun
        raise CheapnessCostIntegrityError(
            "Cheapness provider-reported cost exceeded its conservative "
            f"reservation: ${accounted_cost_usd:.6f} actual > "
            f"${estimated_cost_usd:.6f} reserved"
        )


def _reject_cheapness_paid_attempt(
    conn: sqlite3.Connection,
    *,
    attempt_id: str,
    error: BaseException,
) -> None:
    if conn.in_transaction:
        conn.commit()
    conn.execute("BEGIN IMMEDIATE")
    try:
        cursor = conn.execute(
            """
            UPDATE cheapness_paid_attempts
            SET status = 'REJECTED', error = ?, completed_at = COALESCE(completed_at, ?)
            WHERE attempt_id = ? AND status = 'RETURNED'
            """,
            (str(error)[:500], _now(), attempt_id),
        )
        if cursor.rowcount != 1:
            existing = conn.execute(
                "SELECT status FROM cheapness_paid_attempts WHERE attempt_id = ?",
                (attempt_id,),
            ).fetchone()
            if existing is not None and str(existing["status"]) == "PUBLISHED":
                conn.commit()
                return
            raise CheapnessPaidAttemptAmbiguousError(
                "Cheapness paid attempt was not RETURNED at rejection"
            )
        conn.commit()
    except BaseException:
        conn.rollback()
        raise


def _run_llm_summary(
    provider,
    *,
    ticker: str,
    as_of: str,
    filing_index: list[dict[str, str]],
    sections: dict[str, dict[str, str]],
    mezzanine: dict[str, Any],
    open_events: list[dict[str, Any]],
    max_cost_usd: float,
    financial_integrity_scope: BoundV1FinancialScope | None,
    financial_scenarios: tuple[Any, ...] | None,
) -> dict[str, Any]:
    prompt = _build_prompt(
        ticker,
        as_of=as_of,
        filing_index=filing_index,
        sections=sections,
        mezzanine=mezzanine,
        open_events=open_events,
    )
    budget_usd = max(0.0, float(max_cost_usd))
    cost_context = None
    provider_meta: dict[str, Any] = {}
    strict_provider_kwargs: dict[str, Any] = {}
    from app.autonomous import sector_runtime as sector_runtime_module

    if sector_runtime_module._provider_name(provider) == "openai":
        strict_provider_kwargs["allow_output_token_retry"] = False
    try:
        with llm_cost_budget(
            budget_usd,
            strict_first_call=True,
        ) as cost_context:
            if financial_integrity_scope is not None:
                financial_integrity_scope.require(scenarios=financial_scenarios)
                integrity_scope: Any = financial_integrity_scope
            else:
                integrity_scope = FinancialIntegrityScope(
                    context=f"cheapness:{ticker}:{as_of}",
                    run_as_of_date=as_of,
                )
            payload, provider_meta = _synthesize_provider_json_with_meta(
                provider,
                prompt=prompt,
                schema=CHEAPNESS_SUMMARY_SCHEMA,
                schema_name="cheapness_summary_v1",
                max_output_tokens=1000,
                integrity_scope=integrity_scope,
                **strict_provider_kwargs,
            )
    except InvalidFinancialInputError:
        raise
    except Exception as exc:
        summary = cost_context.summary() if cost_context is not None else {}
        usage = {
            "schema": _CHEAPNESS_LLM_USAGE_SCHEMA,
            "max_cost_usd": budget_usd,
            "call_count": int(summary.get("call_count") or 0),
            "cumulative_cost_usd": float(summary.get("cumulative_cost_usd") or 0.0),
            "reserved_cost_usd": float(summary.get("reserved_cost_usd") or 0.0),
            "events": list(summary.get("events") or []),
        }
        return {
            "verdict": LLM_UNAVAILABLE,
            "bullets": [],
            "model": None,
            "error": str(exc)[:300],
            "usage": usage,
            "cost_usd": usage["cumulative_cost_usd"],
            "budget_usd": budget_usd,
        }

    summary = cost_context.summary() if cost_context is not None else {}
    usage = {
        "schema": _CHEAPNESS_LLM_USAGE_SCHEMA,
        "max_cost_usd": budget_usd,
        "call_count": int(summary.get("call_count") or 0),
        "cumulative_cost_usd": float(summary.get("cumulative_cost_usd") or 0.0),
        "reserved_cost_usd": float(summary.get("reserved_cost_usd") or 0.0),
        "events": list(summary.get("events") or []),
    }
    model = str(provider_meta.get("model") or "") or None

    known_accessions = {f["accession"] for f in filing_index}
    known_accessions.update(e["anchor_accession"] for e in open_events)
    bullets = []
    dropped = 0
    for bullet in payload.get("bullets", [])[:3]:
        cited = [a for a in bullet.get("accessions", []) if a in known_accessions]
        if not cited:
            dropped += 1
            continue
        bullets.append({"text": str(bullet.get("text", "")).strip(), "accessions": cited})
    verdict = payload.get("verdict")
    if verdict not in {NO_KNOWN_EVENT, KNOWN_EVENTS}:
        verdict = KNOWN_EVENTS if bullets else NO_KNOWN_EVENT
    return {
        "verdict": verdict,
        "bullets": bullets,
        "model": model,
        "dropped_uncited": dropped,
        "usage": usage,
        "cost_usd": usage["cumulative_cost_usd"],
        "budget_usd": budget_usd,
    }


# ---------------------------------------------------------------------------
# Orchestration + persistence
# ---------------------------------------------------------------------------


def _publish_cheapness_report(
    *,
    ticker: str,
    cik: str,
    as_of: str,
    conn: sqlite3.Connection,
    client: Any,
    adapter: SecSubmissionsAdapter,
    raw_companyfacts_snapshot: dict[str, Any] | None,
    reload_companyfacts: bool,
    financial_integrity_scope: BoundV1FinancialScope | None,
    call_scope: BoundV1FinancialScope,
    llm: dict[str, Any],
    paid_attempt_id: str | None,
) -> dict[str, Any]:
    """Rebuild and persist publication state under one SQLite write boundary.

    ``BEGIN IMMEDIATE`` closes the last mutable-input window when this helper
    owns the transaction.  If the caller already has a transaction, a
    savepoint provides failure isolation while the caller's existing write
    lock (or SQLite's failed deferred-transaction upgrade) keeps publication
    fail-closed.
    """

    paid_llm_result = dict(llm)
    llm = dict(llm)
    owns_transaction = not conn.in_transaction
    savepoint = "cheapness_publication"
    if owns_transaction:
        conn.execute("BEGIN IMMEDIATE")
    else:
        conn.execute(f"SAVEPOINT {savepoint}")

    try:
        publication_state = _rebuild_cheapness_publication_state(
            ticker=ticker,
            cik=cik,
            as_of=as_of,
            conn=conn,
            client=client,
            adapter=adapter,
            raw_companyfacts_snapshot=raw_companyfacts_snapshot,
            reload_companyfacts=reload_companyfacts,
            financial_integrity_scope=financial_integrity_scope,
        )
        publication_scope_fingerprint = _require_cheapness_scope_unchanged(
            call_scope,
            publication_state["scope"],
        )
        filing_index = publication_state["filing_index"]
        fingerprint = publication_state["filings_fingerprint"]
        sections = publication_state["sections"]
        mezzanine = publication_state["mezzanine"]
        open_events = publication_state["open_events"]
        flags = publication_state["flags"]
        provider_prompt = publication_state["prompt"]

        # Deterministic truth outranks the LLM: open events or a fired flag mean
        # the record is NOT clean.
        if flags and llm["verdict"] == NO_KNOWN_EVENT:
            llm["verdict"] = KNOWN_EVENTS

        deterministic = {
            "filing_index": filing_index,
            "sections": {
                label: {k: v for k, v in payload.items() if k != "excerpt"}
                | {"excerpt_chars": len(payload["excerpt"])}
                for label, payload in sections.items()
            },
            "mezzanine": mezzanine,
            "open_events": [
                {
                    "event_type": event["event_type"],
                    "detection_date": event["detection_date"],
                    "anchor_accession": event["anchor_accession"],
                }
                for event in open_events
            ],
            "llm_error": llm.get("error"),
            "llm_dropped_uncited": llm.get("dropped_uncited", 0),
        }
        now = _now()
        publication_evidence = _cheapness_publication_source_payload(
            ticker=ticker,
            as_of=as_of,
            scope_fingerprint=publication_scope_fingerprint,
            filings_fingerprint_value=fingerprint,
            provider_prompt=provider_prompt,
        )
        publication_source_sha256 = _cheapness_publication_source_sha256(
            ticker=ticker,
            as_of=as_of,
            scope_fingerprint=publication_scope_fingerprint,
            filings_fingerprint_value=fingerprint,
            provider_prompt=provider_prompt,
        )
        publication_row = {
            "ticker": ticker,
            "cik": cik,
            "filings_fingerprint": fingerprint,
            "financial_scope_fingerprint": call_scope.expected_scope_fingerprint,
            "financial_scope_publication_fingerprint": publication_scope_fingerprint,
            "publication_source_sha256": publication_source_sha256,
            "paid_attempt_id": paid_attempt_id,
            "as_of": as_of,
            "flags_json": _canonical_json(flags),
            "deterministic_json": _canonical_json(deterministic),
            "llm_verdict": llm["verdict"],
            "llm_bullets_json": _canonical_json(llm["bullets"]),
            "llm_model": llm.get("model"),
            "llm_usage_json": _canonical_json(llm["usage"]),
            "llm_cost_usd": float(llm["cost_usd"]),
            "llm_budget_usd": float(llm["budget_usd"]),
        }
        publication_row_sha256 = _cheapness_publication_row_sha256(publication_row)
        if publication_row_sha256 is None:
            raise ValueError("cheapness publication row is not canonical")
        conn.execute(
            """
            INSERT INTO cheapness_reports(
                ticker, cik, filings_fingerprint, financial_scope_fingerprint,
                financial_scope_publication_fingerprint,
                publication_source_sha256, publication_row_sha256,
                paid_attempt_id, as_of, flags_json,
                deterministic_json, llm_verdict, llm_bullets_json, llm_model,
                llm_usage_json, llm_cost_usd, llm_budget_usd,
                created_at, updated_at)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker, filings_fingerprint) DO UPDATE SET
                financial_scope_fingerprint=excluded.financial_scope_fingerprint,
                financial_scope_publication_fingerprint=excluded.financial_scope_publication_fingerprint,
                publication_source_sha256=excluded.publication_source_sha256,
                publication_row_sha256=excluded.publication_row_sha256,
                paid_attempt_id=excluded.paid_attempt_id,
                as_of=excluded.as_of,
                flags_json=excluded.flags_json,
                deterministic_json=excluded.deterministic_json,
                llm_verdict=excluded.llm_verdict,
                llm_bullets_json=excluded.llm_bullets_json,
                llm_model=excluded.llm_model,
                llm_usage_json=excluded.llm_usage_json,
                llm_cost_usd=excluded.llm_cost_usd,
                llm_budget_usd=excluded.llm_budget_usd,
                updated_at=excluded.updated_at
            """,
            (
                ticker,
                cik,
                fingerprint,
                call_scope.expected_scope_fingerprint,
                publication_scope_fingerprint,
                publication_source_sha256,
                publication_row_sha256,
                paid_attempt_id,
                as_of,
                publication_row["flags_json"],
                publication_row["deterministic_json"],
                llm["verdict"],
                publication_row["llm_bullets_json"],
                llm.get("model"),
                publication_row["llm_usage_json"],
                publication_row["llm_cost_usd"],
                publication_row["llm_budget_usd"],
                now,
                now,
            ),
        )
        row = conn.execute(
            "SELECT * FROM cheapness_reports WHERE ticker = ? AND filings_fingerprint = ?",
            (ticker, fingerprint),
        ).fetchone()
        if row is None or _cheapness_publication_row_sha256(row) != publication_row_sha256:
            raise ValueError("cheapness publication row changed before authorization")
        conn.execute(
            """
            INSERT OR IGNORE INTO cheapness_publication_authorizations (
                ticker, filings_fingerprint, publication_row_sha256,
                financial_scope_fingerprint, publication_source_sha256,
                publication_evidence_json, authorized_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticker,
                fingerprint,
                publication_row_sha256,
                publication_scope_fingerprint,
                publication_source_sha256,
                _canonical_json(publication_evidence),
                now,
            ),
        )
        if not _cheapness_row_is_authorized(conn, row):
            raise ValueError("cheapness publication authorization did not bind the exact row")
        if paid_attempt_id is not None:
            attempt = conn.execute(
                """
                SELECT status, outcome, result_json, accounted_cost_usd
                FROM cheapness_paid_attempts
                WHERE attempt_id = ?
                """,
                (paid_attempt_id,),
            ).fetchone()
            if attempt is None or str(attempt["status"]) != "RETURNED":
                raise CheapnessPaidAttemptAmbiguousError(
                    "Cheapness publication requires the exact RETURNED paid attempt"
                )
            if _canonical_json(paid_llm_result) != str(attempt["result_json"]):
                raise CheapnessPaidAttemptAmbiguousError(
                    "Cheapness publication result does not match its paid attempt"
                )
            if abs(float(attempt["accounted_cost_usd"]) - float(llm["cost_usd"])) > 1e-12:
                raise CheapnessPaidAttemptAmbiguousError(
                    "Cheapness publication cost does not match its paid attempt"
                )
            attempt_outcome = str(attempt["outcome"] or "")
            if attempt_outcome == "ERROR" and (
                paid_llm_result.get("verdict") != LLM_UNAVAILABLE
                or not paid_llm_result.get("error")
            ):
                raise CheapnessPaidAttemptAmbiguousError(
                    "Cheapness ERROR attempt may publish only its explicit "
                    "LLM_UNAVAILABLE degraded result"
                )
            if attempt_outcome == "SUCCESS" and paid_llm_result.get("error"):
                raise CheapnessPaidAttemptAmbiguousError(
                    "Cheapness SUCCESS attempt cannot publish an error result"
                )
            cursor = conn.execute(
                """
                UPDATE cheapness_paid_attempts
                SET status = 'PUBLISHED', published_at = ?,
                    publication_row_sha256 = ?
                WHERE attempt_id = ? AND status = 'RETURNED'
                """,
                (now, publication_row_sha256, paid_attempt_id),
            )
            if cursor.rowcount != 1:
                raise CheapnessPaidAttemptAmbiguousError(
                    "Cheapness paid attempt changed before publication"
                )
        elif int(llm.get("usage", {}).get("call_count") or 0) != 0:
            raise CheapnessPaidAttemptAmbiguousError(
                "Cheapness publication with physical calls requires a paid attempt"
            )
    except BaseException:
        if owns_transaction:
            conn.rollback()
        else:
            conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
            conn.execute(f"RELEASE SAVEPOINT {savepoint}")
        raise
    else:
        if owns_transaction:
            conn.commit()
        else:
            conn.execute(f"RELEASE SAVEPOINT {savepoint}")

    report = _row_to_report(row)
    report["from_cache"] = False
    return report


def build_cheapness_report(
    ticker: str,
    *,
    as_of: str,
    conn: sqlite3.Connection,
    client=None,
    adapter: SecSubmissionsAdapter | None = None,
    provider=None,
    market_cap_mm: float | None = None,
    raw_companyfacts: dict | None = None,
    force: bool = False,
    llm_budget_usd: float = 0.05,
    financial_integrity_scope: BoundV1FinancialScope | None = None,
) -> dict[str, Any]:
    ticker = ticker.upper()
    reload_companyfacts = raw_companyfacts is None
    if client is None and adapter is None:
        from app.ingest.sec_client import SecClient

        client = SecClient()
    if adapter is None:
        adapter = SecSubmissionsAdapter(client)
    if client is None:
        client = adapter.client

    cik = cik_for_ticker(conn, ticker)
    if cik is None:
        return {"ticker": ticker, "status": "NO_CIK"}

    filing_index = trailing_filing_index(adapter, cik, as_of=as_of)
    if filing_index is None:
        return {"ticker": ticker, "cik": cik, "status": "SUBMISSIONS_FETCH_FAILED"}
    fingerprint = filings_fingerprint(filing_index)

    if financial_integrity_scope is None:
        from app.config import get_config

        financial_context = build_canonical_v1_financial_context(
            tickers=[ticker],
            as_of_date=as_of,
            db_path=get_config().db_path,
        )
        canonical_packet = financial_context.packets[ticker]
    else:
        financial_integrity_scope.require()
        canonical_packet = financial_integrity_scope.packets[0]
    canonical_packet_payload = (
        dict(canonical_packet)
        if isinstance(canonical_packet, dict)
        else dict(vars(canonical_packet))
    )
    market_cap_mm = canonical_packet_payload.get("market_cap_mm")

    if raw_companyfacts is None:
        raw_companyfacts = load_raw_companyfacts(cik)
    mezzanine = mezzanine_check(
        raw_companyfacts,
        market_cap_mm,
        as_of=as_of,
        source_url=(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{str(cik).zfill(10)}.json"),
    )

    sections = extract_risk_sections(client, cik, filing_index)

    open_events = [
        dict(row) for row in store.open_queue_protection_events(conn, ciks=[cik]).get(cik, [])
    ]

    flags = []
    if mezzanine["flag"]:
        flags.append(mezzanine["flag"])
    flags.extend(sorted({store.event_pending_flag(e["event_type"]) for e in open_events}))

    if provider is None:
        from app.llm.providers import get_llm_provider

        provider = get_llm_provider()
    provider_prompt = _build_prompt(
        ticker,
        as_of=as_of,
        filing_index=filing_index,
        sections=sections,
        mezzanine=mezzanine,
        open_events=open_events,
    )
    financial_scenario = financial_input_scenario(
        canonical_packet,
        financial_inputs={
            "market_cap_mm": market_cap_mm,
            "market_cap_unit": canonical_packet_payload.get("market_cap_unit"),
            "mezzanine": mezzanine,
            "provider_prompt": provider_prompt,
        },
    )
    call_scope = bind_v1_financial_scope(
        context=f"cheapness:{ticker}:{as_of}",
        run_as_of_date=as_of,
        packets=(canonical_packet,),
        scenarios=(financial_scenario,),
    )
    scope_fingerprint = call_scope.expected_scope_fingerprint

    if not force:
        cached = conn.execute(
            """
            SELECT *
            FROM cheapness_reports
            WHERE ticker = ?
              AND filings_fingerprint = ?
              AND financial_scope_fingerprint = ?
            """,
            (ticker, fingerprint, scope_fingerprint),
        ).fetchone()
        if cached is not None and _cheapness_row_is_authorized(conn, cached):
            report = _row_to_report(cached)
            report["from_cache"] = True
            return report

    attempt_request = _cheapness_attempt_request(
        provider,
        ticker=ticker,
        cik=cik,
        as_of=as_of,
        filings_fingerprint_value=fingerprint,
        scope_fingerprint=scope_fingerprint,
        prompt=provider_prompt,
    )
    paid_attempt_id: str | None = None
    llm: dict[str, Any]
    if float(attempt_request["estimated_cost_usd"]) <= max(0.0, float(llm_budget_usd)) + 1e-12:
        paid_attempt_id = str(attempt_request["attempt_id"])
        reusable_llm, paid_attempt_status = _reserve_cheapness_paid_attempt(
            conn,
            attempt_request=attempt_request,
            ticker=ticker,
            cik=cik,
            as_of=as_of,
            filings_fingerprint_value=fingerprint,
            scope_fingerprint=scope_fingerprint,
        )
        if paid_attempt_status == "PUBLISHED":
            published = conn.execute(
                """
                SELECT *
                FROM cheapness_reports
                WHERE paid_attempt_id = ?
                  AND financial_scope_fingerprint = ?
                """,
                (paid_attempt_id, scope_fingerprint),
            ).fetchone()
            if published is None or not _cheapness_row_is_authorized(conn, published):
                raise CheapnessPaidAttemptAmbiguousError(
                    "Published cheapness paid attempt lacks its authorized result"
                )
            report = _row_to_report(published)
            report["from_cache"] = True
            return report
        if reusable_llm is None:
            llm = _run_llm_summary(
                provider,
                ticker=ticker,
                as_of=as_of,
                filing_index=filing_index,
                sections=sections,
                mezzanine=mezzanine,
                open_events=open_events,
                max_cost_usd=llm_budget_usd,
                financial_integrity_scope=call_scope,
                financial_scenarios=(financial_scenario,),
            )
            _complete_cheapness_paid_attempt(
                conn,
                attempt_id=paid_attempt_id,
                llm=llm,
            )
        else:
            llm = reusable_llm
    else:
        # Let the shared strict budget context produce the normal deterministic
        # zero-call degraded result. No paid-attempt row is created because no
        # physical egress was authorized.
        llm = _run_llm_summary(
            provider,
            ticker=ticker,
            as_of=as_of,
            filing_index=filing_index,
            sections=sections,
            mezzanine=mezzanine,
            open_events=open_events,
            max_cost_usd=llm_budget_usd,
            financial_integrity_scope=call_scope,
            financial_scenarios=(financial_scenario,),
        )

    try:
        post_response_state = _rebuild_cheapness_publication_state(
            ticker=ticker,
            cik=cik,
            as_of=as_of,
            conn=conn,
            client=client,
            adapter=adapter,
            raw_companyfacts_snapshot=raw_companyfacts,
            reload_companyfacts=reload_companyfacts,
            financial_integrity_scope=financial_integrity_scope,
        )
        _require_cheapness_scope_unchanged(
            call_scope,
            post_response_state["scope"],
        )
    except BaseException as exc:
        if paid_attempt_id is not None:
            _reject_cheapness_paid_attempt(
                conn,
                attempt_id=paid_attempt_id,
                error=exc,
            )
        raise

    # Re-read mutable publication inputs and persist under one SQLite write
    # boundary. A paid response may not authorize an event, filing, quote, or
    # packet that changed while the response was being normalized.
    return _publish_cheapness_report(
        ticker=ticker,
        cik=cik,
        as_of=as_of,
        conn=conn,
        client=client,
        adapter=adapter,
        raw_companyfacts_snapshot=raw_companyfacts,
        reload_companyfacts=reload_companyfacts,
        financial_integrity_scope=financial_integrity_scope,
        call_scope=call_scope,
        llm=llm,
        paid_attempt_id=paid_attempt_id,
    )


def _row_to_report(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "ticker": row["ticker"],
        "cik": row["cik"],
        "status": "OK",
        "as_of": row["as_of"],
        "filings_fingerprint": row["filings_fingerprint"],
        "financial_scope_fingerprint": row["financial_scope_fingerprint"],
        "financial_scope_publication_fingerprint": (row["financial_scope_publication_fingerprint"]),
        "publication_source_sha256": row["publication_source_sha256"],
        "publication_row_sha256": row["publication_row_sha256"],
        "paid_attempt_id": row["paid_attempt_id"],
        "flags": json.loads(row["flags_json"] or "[]"),
        "deterministic": json.loads(row["deterministic_json"] or "{}"),
        "llm_verdict": row["llm_verdict"],
        "llm_bullets": json.loads(row["llm_bullets_json"] or "[]"),
        "llm_model": row["llm_model"],
        "llm_usage": json.loads(row["llm_usage_json"] or "{}"),
        "llm_cost_usd": float(row["llm_cost_usd"] or 0.0),
        "llm_budget_usd": float(row["llm_budget_usd"] or 0.0),
        "updated_at": row["updated_at"],
    }


def latest_cheapness_by_ticker(
    conn: sqlite3.Connection, tickers: list[str] | None = None
) -> dict[str, dict[str, Any]]:
    try:
        sql = (
            "SELECT cr.* FROM cheapness_reports cr "
            "JOIN (SELECT ticker, MAX(id) AS latest_id FROM cheapness_reports "
            "GROUP BY ticker) l "
            "ON l.latest_id = cr.id"
        )
        params: list[str] = []
        if tickers is not None:
            marks = ",".join("?" for _ in tickers)
            sql += f" WHERE cr.ticker IN ({marks})"
            params = [t.upper() for t in tickers]
        rows = conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError:
        return {}
    return {
        row["ticker"]: _row_to_report(row)
        for row in rows
        if _cheapness_row_is_authorized(conn, row)
    }


def cheapness_headline(report: dict[str, Any] | None) -> str:
    """One-cell summary for queue/digest tables."""
    if not report or report.get("status") != "OK":
        return "n/a"
    flags = report.get("flags") or []
    verdict = report.get("llm_verdict")
    n_bullets = len(report.get("llm_bullets") or [])
    parts = list(flags)
    if verdict == NO_KNOWN_EVENT and not flags:
        return NO_KNOWN_EVENT
    if verdict == LLM_UNAVAILABLE:
        parts.append("LLM_UNAVAILABLE")
    elif n_bullets:
        parts.append(f"{n_bullets} candidate reason{'s' if n_bullets != 1 else ''}")
    return "; ".join(parts) if parts else "n/a"


def render_cheapness_block(report: dict[str, Any] | None) -> list[str]:
    """Markdown lines for `ivi watchlist show` and briefs."""
    if not report or report.get("status") != "OK":
        return ["No cheapness report on record (run `ivi events cheapness <ticker>`)."]
    lines = [
        f"- Verdict: {report['llm_verdict'] or 'n/a'} (as of {report['as_of']})",
    ]
    usage = report.get("llm_usage") or {}
    lines.append(
        "- LLM usage: "
        f"{int(usage.get('call_count') or 0)} provider call(s), "
        f"${float(report.get('llm_cost_usd') or 0.0):.4f} charged / "
        f"${float(report.get('llm_budget_usd') or 0.0):.4f} hard cap"
    )
    flags = report.get("flags") or []
    if flags:
        lines.append(f"- Flags: {', '.join(flags)}")
    mezz = (report.get("deterministic") or {}).get("mezzanine") or {}
    if mezz.get("mezzanine_total_mm") is not None:
        ratio = mezz.get("ratio_to_cap")
        ratio_str = f"{ratio * 100:.0f}% of market cap" if ratio is not None else "ratio n/a"
        lines.append(
            f"- Mezzanine obligations: ${mezz['mezzanine_total_mm']:,.1f}M carrying amount "
            f"({ratio_str})"
        )
        for comp in mezz.get("components", []):
            lines.append(
                f"  - {comp['concept']}: ${comp['value_mm']:,.1f}M as of {comp['as_of']} "
                f"(accession {comp['accession']})"
            )
    for event in (report.get("deterministic") or {}).get("open_events", []):
        lines.append(
            f"- Open event: {event['event_type']} since {event['detection_date']} "
            f"(anchor {event['anchor_accession']})"
        )
    for bullet in report.get("llm_bullets") or []:
        accessions = ", ".join(bullet.get("accessions", []))
        lines.append(f"- {bullet['text']} [{accessions}]")
    if report["llm_verdict"] == NO_KNOWN_EVENT and not flags:
        lines.append("- Filing record clean over the trailing 12 months.")
    return lines
