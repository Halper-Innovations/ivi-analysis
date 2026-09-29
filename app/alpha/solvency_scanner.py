"""Assess solvency and going-concern risk from financials + filing text.

Combines deterministic financial checks (negative equity, current ratio,
cash runway) with keyword search of the filing for distress language
(going concern, valuation allowance, no assurance of financing, debt maturities).

No LLM calls — purely deterministic + keyword matching.
"""

from __future__ import annotations

import logging
import math
import os
import re
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from app.alpha.schemas import GoingConcernAssertion, SolvencyAssessment
from app.db import connect, get_db
from app.util.financial_data_access import (
    ANNUAL_COMPANYFACTS_PERIOD_TYPES,
    companyfacts_rows,
    issuer_companyfacts_rows,
)
from app.valuation.lineage import latest_decision_eligible_valuation_row

logger = logging.getLogger(__name__)

# Keyword patterns for filing text search
_GOING_CONCERN_PATTERNS = [
    r"\bgoing\s+concern\b",
    r"\bsubstantial\s+doubt\s+about\s+(?:our|the\s+company(?:['’]s)?)\s+ability\s+to\s+continue\b",
    r"\braise(?:s|d)?\s+substantial\s+doubt\b",
    r"\bability\s+to\s+(?:continue|operate)\s+as\s+a\s+going\s+concern\b",
]
_NEGATED_GOING_CONCERN_PATTERNS = [
    r"substantial\s+doubt.*does\s+not\s+exist",
    r"substantial\s+doubt.*did\s+not\s+exist",
    r"substantial\s+doubt.*not\s+exist",
    r"d(?:o|oes|id)\s+not\s+raise\s+substantial\s+doubt",
    r"no\s+substantial\s+doubt",
    # Other ways a filing says the doubt is ABSENT. Each names what is negated right
    # after the negation, so "has not identified financing, and there is substantial
    # doubt" (an assertion) is not swept in: ("no conditions or events raise
    # substantial doubt", "does not include an explanatory paragraph regarding our
    # ability to continue as a going concern", "did not express substantial doubt",
    # "not aware of any conditions that raise substantial doubt", "do not believe there
    # is substantial doubt", "absence of substantial doubt").
    r"\bno\s+(?:conditions?|events?|explanatory\s+paragraph)\b[^.;]{0,200}"
    r"(?:substantial\s+doubt|going\s+concern)",
    r"\bnot\s+(?:currently\s+|yet\s+)?(?:aware\s+of|identif\w+|conclud\w+|determin\w+|find|found"
    r"|express\w*|includ\w+|contain\w*|issu\w+|report\w*)\s+(?:that\s+)?"
    r"(?:there\s+(?:is|are|was|were)\s+)?(?:any\s+|an\s+|a\s+)?"
    r"(?:(?:material\s+|such\s+)?(?:conditions?|events?|substantial\s+doubt|explanatory\s+paragraph)"
    r"|going\s+concern)",
    r"\bnot\s+(?:believe|think)\s+(?:that\s+)?(?:there\s+(?:is|are)\s+)?(?:any\s+)?"
    r"(?:conditions?|events?|substantial\s+doubt)",
    r"\bthere\s+(?:is|are|was|were)\s+not\s+(?:any\s+)?substantial\s+doubt",
    r"\b(?:absence\s+of|free\s+(?:of|from)|without)\b[^.;]{0,80}"
    r"(?:substantial\s+doubt|explanatory\s+paragraph)",
]
# ASC 205-40 process language: management "evaluates whether there are conditions and
# events ... that raise substantial doubt". It describes the test, not a result, so it
# is accounting policy unless the same sentence also states the conclusion outright.
_EVALUATION_FRAMING_GOING_CONCERN_PATTERNS = [
    r"\b(?:evaluat\w*|assess\w*|consider\w*|determin\w*|analy[sz]\w*)\s+(?:each\s+\w+\s+)?"
    r"(?:whether|if)\b[^.;]{0,300}(?:substantial\s+doubt|going\s+concern)",
    r"\bwhether\s+(?:there\s+(?:are|is)\s+)?(?:any\s+)?(?:conditions?|events?)\b[^.;]{0,300}"
    r"(?:substantial\s+doubt|going\s+concern)",
]
_HYPOTHETICAL_GOING_CONCERN_PATTERNS = [
    r"\bif\b.{0,120}(?:going\s+concern|substantial\s+doubt)",
    r"\b(?:could|may|might|would)\b.{0,220}(?:going\s+concern|substantial\s+doubt)",
    r"(?:going\s+concern|substantial\s+doubt).{0,80}\b(?:could|may|might|would)\b",
]
# Present-indicative doubt assertions.  A modal elsewhere in the sentence must
# not suppress these: in "plans may not be implemented, and substantial doubt
# exists" the modal governs the plans, not the doubt.  Only modal-governed
# doubt clauses ("could raise substantial doubt") stay hypothetical, so the
# plural form carries lookbehinds for the modals and the "whether/if/when"
# evaluation framings stay out of the exists/there-is forms.
_ASSERTED_GOING_CONCERN_PATTERNS = [
    r"(?<!if )(?<!when )(?<!whether )\bsubstantial\s+doubt\s+(?:currently\s+)?exists\b",
    r"(?<!whether )\bthere\s+(?:is|remains)\s+substantial\s+doubt\b",
    r"\braises\s+substantial\s+doubt\b",
    r"(?<!may )(?<!might )(?<!could )(?<!would )(?<!will )(?<!can )(?<!shall )"
    r"(?<!should )(?<!not )(?<!to )\braise\s+substantial\s+doubt\b",
]
_NO_ASSURANCE_PATTERNS = [
    r"no\s+assurance(?:\s+can\s+be\s+given)?(?:[^.;]{0,120})(?:raise|obtain|secure|procure|access|arrange)(?:[^.;]{0,120})(?:financ|capital|fund|liquidity|credit)",
    r"cannot\s+(?:guarantee|assure)(?:[^.;]{0,120})(?:additional|sufficient|needed|necessary)(?:[^.;]{0,80})(?:financ|capital|fund|liquidity|credit)",
    r"may\s+not\s+be\s+able\s+to\s+(?:raise|obtain|secure|access|arrange)(?:[^.;]{0,120})(?:financ|capital|fund|liquidity|credit|refinancing)",
]
_VALUATION_ALLOWANCE_PATTERNS = [
    r"full\s+valuation\s+allowance",
    r"recorded\s+a\s+(?:full\s+)?valuation\s+allowance\s+against\s+(?:our|the|all)",
    r"more.likely.than.not\s+that\s+(?:such\s+)?deferred\s+tax\s+assets\s+w(?:ill|ould)\s+not\s+be\s+realized",
]
_DEBT_MATURITY_PATTERNS = [
    r"(?:repayment|maturity|maturities|due)\s+(?:date|during)\s+.*(?:20\d{2})",
    r"(?:convertible\s+note|credit\s+facility|term\s+loan).*(?:due|matur|repay).*(?:20\d{2})",
]

_BLOCKABLE_GOING_CONCERN_SUBJECTS = frozenset(
    {"REGISTRANT", "CONSOLIDATED_GROUP", "CONSOLIDATED_SUBSIDIARY"}
)
_CONSOLIDATED_GROUP_PATTERNS = [
    r"\b(?:the\s+)?company\s+and\s+(?:its\s+)?(?:consolidated\s+)?subsidiar(?:y|ies)\b",
    r"\bwe\s+and\s+our\s+(?:consolidated\s+)?subsidiar(?:y|ies)\b",
]
_CONSOLIDATED_SUBSIDIARY_PATTERNS = [
    r"\b(?:our|the\s+company['’]s|a|the)\s+(?:wholly[-\s]+owned\s+)?consolidated\s+subsidiar(?:y|ies)\b",
    r"\b(?:our|the\s+company['’]s)\s+wholly[-\s]+owned\s+subsidiar(?:y|ies)\b",
]
_REGISTRANT_SUBJECT_PATTERNS = [
    r"\bour\s+ability\s+to\s+(?:continue|operate)\b",
    r"\b(?:the\s+)?company['’]s\s+ability\s+to\s+(?:continue|operate)\b",
    r"\bability\s+of\s+the\s+company\s+to\s+(?:continue|operate)\b",
    r"\bmanagement\s+(?:has|have)\s+(?:concluded|determined)\b",
    r"\bwe\s+(?:cannot|could\s+not|may\s+not|might\s+not|will\s+not|are\s+unable\s+to)\b[^.;]{0,160}\b(?:continue|operate)\b",
]
# Non-possessive registrant forms ("its ability", "the entity's ability") are
# how the AS 2415 auditor paragraph names the registrant, but the same pronoun
# also appears in third-party sentences ("Bakkt … monitoring its ability"), so
# these only attribute after every third-party/consolidated check has failed.
_REGISTRANT_FALLBACK_SUBJECT_PATTERNS = [
    r"\bits\s+ability\s+to\s+(?:continue|operate)\b",
    r"\bthe\s+entity['’]s\s+ability\s+to\s+(?:continue|operate)\b",
]
_THIRD_PARTY_SUBJECT_PATTERNS: tuple[tuple[str, list[str]], ...] = (
    (
        "INVESTEE",
        [
            r"\binvestee(?:['’]s|s)?\b",
            r"\bequity\s+method\s+investment\b",
            r"\bcarrying\s+(?:amount|value)\s+of\s+(?:the|our)\s+investment\b",
            r"\binvestment\s+in\s+[a-z0-9&.\-]+\b",
        ],
    ),
    (
        "PARTNER",
        [
            r"\bpartners?\b",
            r"\bjoint\s+ventures?\b",
            r"\bchannel\s+sales\s+relationships?\b",
            r"\bplatform\s+partnerships?\b",
        ],
    ),
    (
        "COUNTERPARTY",
        [
            r"\bcounterpart(?:y|ies)(?:['’]s)?\b",
            r"\bpurchasers?(?:['’]s)?\b",
            r"\bbuyers?(?:['’]s)?\b",
            r"\bsellers?(?:['’]s)?\b",
            r"\bcustomers?(?:['’]s)?\b",
            r"\bsuppliers?(?:['’]s)?\b",
            r"\bvendors?(?:['’]s)?\b",
        ],
    ),
)
_ACCOUNTING_POLICY_PATTERNS = [
    r"\b(?:investee|investment)\b[^.;]{0,220}\b(?:impair|carrying\s+(?:amount|value)|recoverable|financial\s+indicators?)\b",
    r"\b(?:impair|carrying\s+(?:amount|value)|recoverable|financial\s+indicators?)\b[^.;]{0,220}\b(?:investee|investment)\b",
    r"\bsignificant\s+doubt\s+about\s+an?\s+investee['’]s\s+ability\b",
    r"\binvestee['’]s\s+ability\s+to\s+(?:continue|operate)\s+as\s+a\s+going\s+concern\b",
]


@dataclass(frozen=True)
class _FilingEvidence:
    text: str
    accession: str | None = None
    form_type: str | None = None
    filing_date: str | None = None
    issuer_cik: str | None = None
    source_url: str | None = None
    content_revision: str | None = None


def _db_context(db_path: str | Path | None):
    """Open the requested DB without falling back to the configured DB path."""

    return get_db() if db_path is None else closing(connect(db_path))


def _load_latest_annual(
    ticker: str,
    *,
    as_of_date: str | None = None,
    require_filed_asof: bool = False,
    issuer_cik: str | None = None,
    aliases: Sequence[str] = (),
    db_path: str | Path | None = None,
) -> dict[str, float]:
    try:
        with _db_context(db_path) as conn:
            kwargs = {
                "period_types": ANNUAL_COMPANYFACTS_PERIOD_TYPES,
                "as_of_date": as_of_date,
                "value_not_null": True,
                "require_filed_asof": require_filed_asof,
                "order_by": "fiscal_year DESC",
            }
            if issuer_cik is not None or aliases:
                _scope, rows = issuer_companyfacts_rows(
                    conn,
                    ticker,
                    columns=("line_item", "value"),
                    issuer_cik=issuer_cik,
                    aliases=aliases,
                    **kwargs,
                )
            else:
                rows = companyfacts_rows(
                    conn,
                    ticker,
                    columns=("line_item", "value"),
                    **kwargs,
                )
    except Exception:
        return {}
    out: dict[str, float] = {}
    for r in rows:
        li = str(r["line_item"])
        if li not in out:  # keep most recent year only
            out[li] = float(r["value"])
    return out


def _load_market_cap(
    ticker: str,
    *,
    as_of_date: str | None = None,
    db_path: str | Path | None = None,
) -> float | None:
    """Best-effort market cap from the newest exact-source scorecard."""
    try:
        with _db_context(db_path) as conn:
            row = latest_decision_eligible_valuation_row(
                conn,
                ticker=ticker,
                method="scorecard",
                as_of_date=as_of_date,
            )
    except Exception:
        return None
    if not row or not row["outputs_json"]:
        return None
    try:
        import json as _json

        sc = _json.loads(row["outputs_json"])
    except Exception:
        return None
    pzd = sc.get("pricing_zone_detail") or {}
    mc = pzd.get("market_cap")
    if isinstance(mc, (int, float)) and mc > 0:
        return float(mc)
    return None


def _is_net_cash_fortress(
    cash: float,
    total_debt: float | None,
    market_cap: float | None,
) -> bool:
    """A company is a 'net-cash fortress' when it has meaningfully more cash
    than debt AND that net-cash cushion is either sizeable in absolute dollars
    or material relative to market cap.

    This catches HRMY-style companies ($753M cash, $164M debt, ~$1.7B mcap)
    that were being flagged CRITICAL because of going-concern regex matches
    on innocent 10-K boilerplate.

    Two tests fire independently:
    1. Relative: net_cash / market_cap >= 20% (only when mcap is known)
    2. Absolute: net_cash >= $100M AND (no debt OR net_cash >= 2x debt)

    Either triggers the fortress override.
    """
    if (
        cash is None
        or total_debt is None
        or not math.isfinite(float(cash))
        or not math.isfinite(float(total_debt))
        or cash <= 0
    ):
        return False
    debt = float(total_debt)
    net_cash = cash - debt
    if net_cash <= 0:
        return False

    # Relative test — catches mid/small caps where 20%+ of mcap is in net cash
    if market_cap is not None and market_cap > 0:
        if (net_cash / market_cap) >= 0.20:
            return True

    # Absolute test — catches large net-cash balance sheets regardless of mcap
    # (e.g. a $30B mcap company with $1.4B net cash and zero debt is still
    # clearly not a solvency risk even though net cash is only ~5% of mcap)
    if net_cash >= 100.0 and (debt == 0 or net_cash >= 2 * debt):
        return True

    return False


def _load_filing_evidence(
    ticker: str,
    *,
    as_of_date: str | None = None,
    require_filed_asof: bool = False,
    allow_network_materialization: bool = True,
    issuer_cik: str | None = None,
    aliases: Sequence[str] = (),
    db_path: str | Path | None = None,
) -> _FilingEvidence | None:
    """Load the latest annual filing text together with immutable identity."""
    try:
        from app.alpha.filing_risk_scan import _find_latest_annual_path, _strip_html

        # A strict filed-as-of read must have an actual cutoff and may only use
        # already-materialized evidence.  Ignore a permissive compatibility
        # argument rather than allowing strict callers to fetch after the fact.
        if require_filed_asof and not as_of_date:
            return None
        materialization_allowed = bool(allow_network_materialization) and not require_filed_asof
        bounded_lookup = bool(
            as_of_date is not None
            or require_filed_asof
            or issuer_cik is not None
            or aliases
            or db_path is not None
        )
        issuer_bound_lookup = bool(issuer_cik is not None or aliases or db_path is not None)

        if bounded_lookup:
            from app.alpha.filing_risk_scan import _latest_readable_annual_document

            with _db_context(db_path) as conn:
                document, _warnings = _latest_readable_annual_document(
                    ticker,
                    as_of_date=as_of_date,
                    issuer_cik=issuer_cik,
                    aliases=tuple(aliases),
                    issuer_aware=issuer_bound_lookup,
                    allow_network_materialization=materialization_allowed,
                    connection=conn,
                )
        else:
            from app.alpha.filing_risk_scan import _latest_readable_annual_document

            document, _warnings = _latest_readable_annual_document(
                ticker,
                allow_network_materialization=materialization_allowed,
            )
        if document is not None:
            return _FilingEvidence(
                text=_strip_html(document.html),
                accession=str(document.accession or "") or None,
                form_type=str(document.form_type or "") or None,
                filing_date=str(document.filing_date or "") or None,
                issuer_cik=str(document.cik or "") or None,
                source_url=str(document.primary_doc_url or "") or None,
                content_revision=str(document.content_revision or "") or None,
            )

        # Preserve the path-only fallback solely for the undated, ticker-only
        # legacy lane.  It cannot represent a cutoff or issuer identity, so
        # using it after a bounded lookup would silently discard those
        # constraints and could read a future or different-issuer filing.
        if bounded_lookup:
            return None
        path, form_type = _find_latest_annual_path(ticker)
        if not path or not os.path.exists(path):
            return None
        with open(path, "r", errors="ignore") as fh:
            html = fh.read()
        return _FilingEvidence(
            text=_strip_html(html),
            form_type=str(form_type or "") or None,
        )
    except Exception:
        return None


def _load_filing_text(
    ticker: str,
    *,
    as_of_date: str | None = None,
    require_filed_asof: bool = False,
    allow_network_materialization: bool = True,
    issuer_cik: str | None = None,
    aliases: Sequence[str] = (),
    db_path: str | Path | None = None,
) -> str | None:
    """Compatibility wrapper returning only the latest annual filing text."""

    evidence = _load_filing_evidence(
        ticker,
        as_of_date=as_of_date,
        require_filed_asof=require_filed_asof,
        allow_network_materialization=allow_network_materialization,
        issuer_cik=issuer_cik,
        aliases=aliases,
        db_path=db_path,
    )
    return evidence.text if evidence is not None else None


def _search_patterns(text: str, patterns: list[str]) -> bool:
    lower = text.lower()
    for pat in patterns:
        if re.search(pat, lower):
            return True
    return False


def _match_windows(text: str, patterns: list[str], window: int = 160) -> list[str]:
    """Extract normalized windows around regex matches for context checks."""
    lower = re.sub(r"\s+", " ", text.lower())
    windows: list[str] = []
    for pat in patterns:
        for match in re.finditer(pat, lower):
            start = max(0, match.start() - window)
            end = min(len(lower), match.end() + window)
            windows.append(lower[start:end])
    return windows


def _assertion_excerpt(text: str, start: int, end: int, *, max_chars: int = 640) -> str:
    """Return the containing sentence without borrowing a neighboring subject."""

    left = max(text.rfind(mark, 0, start) for mark in (".", "?", "!")) + 1
    right_candidates = [
        position for mark in (".", "?", "!") if (position := text.find(mark, end)) >= 0
    ]
    right = min(right_candidates) + 1 if right_candidates else len(text)
    excerpt = text[left:right].strip()
    if len(excerpt) <= max_chars:
        return excerpt
    local_start = max(left, start - (max_chars // 2))
    local_end = min(right, local_start + max_chars)
    if local_end - local_start < max_chars:
        local_start = max(left, local_end - max_chars)
    return text[local_start:local_end].strip()


def _filing_section(text: str, match_start: int) -> str:
    """Infer a stable coarse filing section from the nearest preceding heading."""

    prefix = text[max(0, match_start - 6000) : match_start].lower()
    markers = {
        "AUDITOR_REPORT": (
            "report of independent registered public accounting firm",
            "report of independent auditors",
        ),
        "FINANCIAL_STATEMENTS_NOTES": (
            "notes to consolidated financial statements",
            "notes to financial statements",
        ),
        "ITEM_1A_RISK_FACTORS": ("item 1a risk factors", "item 1a. risk factors"),
        "ITEM_7_MD&A": (
            "item 7 management's discussion",
            "item 7 management’s discussion",
            "liquidity and capital resources",
        ),
    }
    latest = (-1, "ANNUAL_FILING_OTHER")
    for section, candidates in markers.items():
        position = max(prefix.rfind(candidate) for candidate in candidates)
        if position > latest[0]:
            latest = (position, section)
    return latest[1]


def _assertion_mode(excerpt: str) -> str:
    lower = excerpt.lower()
    if _search_patterns(lower, _NEGATED_GOING_CONCERN_PATTERNS):
        return "NEGATED"
    if _search_patterns(
        lower,
        [
            r"\bsubstantial\s+doubt\b[^.;]{0,180}\b(?:alleviated|resolved|removed)\b",
            r"\b(?:alleviated|resolved|removed)\b[^.;]{0,180}\bsubstantial\s+doubt\b",
            r"\bno\s+longer\b[^.;]{0,160}\b(?:going\s+concern|substantial\s+doubt)\b",
        ],
    ):
        return "HISTORICAL_RESOLVED"
    if _search_patterns(lower, _ACCOUNTING_POLICY_PATTERNS):
        return "ACCOUNTING_POLICY"
    if _search_patterns(lower, _EVALUATION_FRAMING_GOING_CONCERN_PATTERNS) and not _search_patterns(
        lower, _ASSERTED_GOING_CONCERN_PATTERNS[:2]
    ):
        return "ACCOUNTING_POLICY"
    if _search_patterns(lower, _ASSERTED_GOING_CONCERN_PATTERNS):
        return "AFFIRMATIVE_CURRENT"
    if _search_patterns(lower, _HYPOTHETICAL_GOING_CONCERN_PATTERNS):
        return "HYPOTHETICAL"
    return "AFFIRMATIVE_CURRENT"


def _assertion_subject(excerpt: str, ticker: str) -> tuple[str, str | None]:
    lower = excerpt.lower()
    for pattern in _REGISTRANT_SUBJECT_PATTERNS:
        if match := re.search(pattern, lower):
            return "REGISTRANT", str(ticker or "").upper() or match.group(0)

    # Accounting policies commonly say "we assess an investee's ability".
    # The speaker is the registrant, but the going-concern subject is still the
    # investee, so third-party nouns precede broader group/header attribution.
    for subject, patterns in _THIRD_PARTY_SUBJECT_PATTERNS:
        for pattern in patterns:
            if match := re.search(pattern, lower):
                return subject, match.group(0)
    named_third_party = re.search(
        r"\b([a-z][a-z0-9&.\-]{2,})\s+(?:has\s+disclosed|is\s+monitoring|"
        r"has\s+reported)[^.;]{0,180}\b(?:going\s+concern|substantial\s+doubt)\b",
        lower,
    )
    if named_third_party and named_third_party.group(1) not in {
        "company",
        "issuer",
        "management",
        "registrant",
    }:
        return "THIRD_PARTY", named_third_party.group(1)

    for pattern in _CONSOLIDATED_GROUP_PATTERNS:
        if match := re.search(pattern, lower):
            return "CONSOLIDATED_GROUP", match.group(0)
    for pattern in _CONSOLIDATED_SUBSIDIARY_PATTERNS:
        if match := re.search(pattern, lower):
            return "CONSOLIDATED_SUBSIDIARY", match.group(0)
    for pattern in _REGISTRANT_FALLBACK_SUBJECT_PATTERNS:
        if match := re.search(pattern, lower):
            return "REGISTRANT", str(ticker or "").upper() or match.group(0)
    return "UNATTRIBUTED", None


def detect_going_concern_assertions(
    text: str,
    *,
    ticker: str,
    accession: str | None = None,
    form_type: str | None = None,
    filing_date: str | None = None,
    issuer_cik: str | None = None,
    source_url: str | None = None,
    content_revision: str | None = None,
    section: str | None = None,
    corroborating_distress: Sequence[str] = (),
) -> list[GoingConcernAssertion]:
    """Extract subject-attributed going-concern assertions from filing text.

    Matches about third parties, accounting policy, hypotheticals, negations,
    and unattributed language remain visible evidence but are never blockable.
    Only current affirmative assertions about the registrant or its
    consolidated group/subsidiary can drive ``going_concern_language=True``.
    """

    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if not normalized:
        return []
    matches = sorted(
        (
            match
            for pattern in _GOING_CONCERN_PATTERNS
            for match in re.finditer(pattern, normalized, flags=re.IGNORECASE)
        ),
        key=lambda item: (item.start(), item.end()),
    )
    distress = tuple(dict.fromkeys(str(item) for item in corroborating_distress if str(item)))
    assertions: list[GoingConcernAssertion] = []
    seen_excerpts: set[str] = set()
    for match in matches:
        excerpt = _assertion_excerpt(normalized, match.start(), match.end())
        excerpt_key = excerpt.lower()
        if excerpt_key in seen_excerpts:
            continue
        seen_excerpts.add(excerpt_key)
        assertion_mode = _assertion_mode(excerpt)
        subject, subject_detail = _assertion_subject(excerpt, ticker)
        blockable = bool(
            assertion_mode == "AFFIRMATIVE_CURRENT" and subject in _BLOCKABLE_GOING_CONCERN_SUBJECTS
        )
        assertions.append(
            GoingConcernAssertion(
                subject=subject,
                subject_detail=subject_detail,
                assertion_mode=assertion_mode,
                blockable=blockable,
                accession=str(accession or "") or None,
                form_type=str(form_type or "") or None,
                filing_date=str(filing_date or "") or None,
                section=section or _filing_section(normalized, match.start()),
                excerpt=excerpt,
                corroborating_distress=distress,
                issuer_cik=str(issuer_cik or "") or None,
                source_url=str(source_url or "") or None,
                content_revision=str(content_revision or "") or None,
            )
        )
    return assertions


def going_concern_asserted(solvency: object) -> bool:
    """The one rule for "this solvency record supports a going-concern block".

    True only when the record reports going-concern language AND carries at least one
    stored assertion that is blockable (a current affirmative statement about the
    registrant or its consolidated group) with a non-empty filed excerpt. A bare
    ``going_concern_language: true`` — an old cached payload, a tool summary that dropped
    the assertions — is a text flag, not a filed assertion, and blocks nothing.

    Accepts a ``SolvencyAssessment``, or the dict form the packets and tool payloads
    carry (assertions as dicts from ``GoingConcernAssertion.to_dict``).
    """

    def _get(source: object, key: str) -> object:
        if isinstance(source, dict):
            return source.get(key)
        return getattr(source, key, None)

    if _get(solvency, "going_concern_language") is not True:
        return False
    assertions = _get(solvency, "going_concern_assertions")
    if not isinstance(assertions, (list, tuple)):
        return False
    return any(
        _get(item, "blockable") is True and str(_get(item, "excerpt") or "").strip()
        for item in assertions
    )


def _detect_going_concern_language(text: str) -> bool:
    """Compatibility boolean derived from the attributed assertion contract."""

    return any(
        assertion.blockable for assertion in detect_going_concern_assertions(text, ticker="")
    )


def assess_solvency(
    ticker: str,
    *,
    as_of_date: str | None = None,
    require_filed_asof: bool = False,
    issuer_cik: str | None = None,
    aliases: Sequence[str] = (),
    db_path: str | Path | None = None,
) -> SolvencyAssessment:
    """Assess solvency risk from financials and filing text."""
    upper = ticker.upper()
    facts = _load_latest_annual(
        upper,
        as_of_date=as_of_date,
        require_filed_asof=require_filed_asof,
        issuer_cik=issuer_cik,
        aliases=aliases,
        db_path=db_path,
    )

    # Preserve the legacy ticker-only fast path: historically a missing facts
    # row returned UNKNOWN without filing materialization.  Explicit v2/PIT
    # callers still inspect the filing because issuer-bound evidence can be
    # available even when normalized facts are incomplete.
    if (
        not facts
        and as_of_date is None
        and not require_filed_asof
        and issuer_cik is None
        and not aliases
        and db_path is None
    ):
        return SolvencyAssessment(
            solvency_risk="UNKNOWN",
            details="No financial data available.",
        )

    facts_available = bool(facts)

    # Keep the historical net-cash override, but defer applying it until after
    # filing attribution.  The override was a workaround for boilerplate false
    # positives; it must not erase an affirmative registrant/consolidated-
    # subsidiary disclosure now that those can be distinguished reliably.
    cash_val = facts.get("cash")
    total_debt_val = facts.get("total_debt")
    equity_val = facts.get("equity")
    mcap = _load_market_cap(upper, as_of_date=as_of_date, db_path=db_path)
    net_cash_fortress = _is_net_cash_fortress(cash_val, total_debt_val, mcap) and not (
        equity_val is not None and equity_val < 0
    )
    fortress_details: str | None = None
    if net_cash_fortress:
        net_cash = cash_val - total_debt_val
        mc_pct = f" ({net_cash / mcap:.0%} of market cap)" if mcap else ""
        fortress_details = (
            f"Net-cash fortress: ${cash_val:,.0f}M cash vs. "
            f"${total_debt_val:,.0f}M debt = ${net_cash:,.0f}M net cash"
            f"{mc_pct}. Fortress balance sheet overrides non-blockable text-based flags."
        )

    signals: list[str] = []

    # --- Financial checks ---
    equity = facts.get("equity")
    negative_equity = equity is not None and equity < 0
    if negative_equity:
        signals.append("NEGATIVE_EQUITY")

    ca = facts.get("current_assets")
    cl = facts.get("current_liabilities")
    current_ratio = (ca / cl) if ca and cl and cl > 0 else None
    current_ratio_distressed = current_ratio is not None and current_ratio < 1.0
    critical_liquidity = current_ratio_distressed and (current_ratio or 1) < 0.7
    if current_ratio_distressed:
        signals.append("CURRENT_RATIO_BELOW_1")

    cash = facts.get("cash")
    cfo = facts.get("cfo")
    cash_runway_quarters = None
    if cfo is not None and cfo < 0 and cash is not None and cash > 0:
        cash_runway_quarters = (cash / abs(cfo)) * 4  # annualize to quarters
        if cash_runway_quarters < 8:
            signals.append("LOW_CASH_RUNWAY")

    # --- Filing text checks ---
    filing_evidence = _load_filing_evidence(
        upper,
        as_of_date=as_of_date,
        require_filed_asof=require_filed_asof,
        allow_network_materialization=not require_filed_asof,
        issuer_cik=issuer_cik,
        aliases=aliases,
        db_path=db_path,
    )
    filing_text = filing_evidence.text if filing_evidence is not None else None
    going_concern = False
    no_assurance = False
    valuation_allowance = False
    debt_due = False
    going_concern_assertions: list[GoingConcernAssertion] = []

    if filing_text:
        no_assurance = _search_patterns(filing_text, _NO_ASSURANCE_PATTERNS)
        valuation_allowance = _search_patterns(filing_text, _VALUATION_ALLOWANCE_PATTERNS)
        debt_due = _search_patterns(filing_text, _DEBT_MATURITY_PATTERNS)

        corroborating_distress: list[str] = []
        if negative_equity:
            corroborating_distress.append("NEGATIVE_EQUITY")
        if critical_liquidity:
            corroborating_distress.append("CRITICAL_LIQUIDITY_RATIO_BELOW_0_7")
        if "LOW_CASH_RUNWAY" in signals:
            corroborating_distress.append("LOW_CASH_RUNWAY")
        if no_assurance:
            corroborating_distress.append("NO_ASSURANCE_FINANCING")
        if debt_due:
            corroborating_distress.append("DEBT_DUE_WITHIN_12MO")

        going_concern_assertions = detect_going_concern_assertions(
            filing_text,
            ticker=upper,
            accession=filing_evidence.accession if filing_evidence else None,
            form_type=filing_evidence.form_type if filing_evidence else None,
            filing_date=filing_evidence.filing_date if filing_evidence else None,
            issuer_cik=filing_evidence.issuer_cik if filing_evidence else None,
            source_url=filing_evidence.source_url if filing_evidence else None,
            content_revision=(filing_evidence.content_revision if filing_evidence else None),
            corroborating_distress=corroborating_distress,
        )
        going_concern = any(item.blockable for item in going_concern_assertions)
        if going_concern:
            signals.append("GOING_CONCERN_LANGUAGE")
        if no_assurance:
            signals.append("NO_ASSURANCE_FINANCING")
        if valuation_allowance:
            signals.append("FULL_VALUATION_ALLOWANCE")
        if debt_due:
            signals.append("DEBT_DUE_WITHIN_12MO")

    if net_cash_fortress and not going_concern:
        return SolvencyAssessment(
            solvency_risk="LOW",
            going_concern_assertions=going_concern_assertions,
            details=fortress_details or "Net-cash fortress.",
        )

    # --- Classification ---
    corroborating_distress = sum(
        [
            negative_equity,
            critical_liquidity,
            no_assurance,
            "LOW_CASH_RUNWAY" in signals,
        ]
    )
    elevated_count = len(signals)

    if corroborating_distress >= 2 or (going_concern and corroborating_distress >= 1):
        risk = "CRITICAL"
    elif going_concern or elevated_count >= 1:
        risk = "ELEVATED"
    elif not facts_available:
        risk = "UNKNOWN"
    else:
        risk = "LOW"

    details_parts = []
    if negative_equity:
        details_parts.append(f"Negative equity (${equity:.1f}M)")
    if current_ratio is not None and current_ratio_distressed:
        details_parts.append(f"Current ratio {current_ratio:.2f}")
    if cash_runway_quarters is not None:
        details_parts.append(f"~{cash_runway_quarters:.0f} quarters cash runway")
    if going_concern:
        details_parts.append("Going concern language in filing")
    if no_assurance:
        details_parts.append("'No assurance' about financing in filing")
    if valuation_allowance:
        details_parts.append("Full valuation allowance on deferred tax assets")
    if debt_due:
        details_parts.append("Debt maturities within 12 months")

    return SolvencyAssessment(
        solvency_risk=risk,
        negative_equity=negative_equity,
        current_ratio=current_ratio,
        current_ratio_distressed=current_ratio_distressed,
        debt_due_within_12mo=debt_due,
        going_concern_language=going_concern,
        valuation_allowance_full=valuation_allowance,
        no_assurance_financing=no_assurance,
        cash_runway_quarters=cash_runway_quarters,
        going_concern_assertions=going_concern_assertions,
        signals=signals,
        details=(
            "; ".join(details_parts)
            if details_parts
            else "No financial data available."
            if not facts_available
            else "No distress signals detected."
        ),
    )
