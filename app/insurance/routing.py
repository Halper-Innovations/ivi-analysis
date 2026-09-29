"""Deterministic security and insurer routing.

The routing layer is intentionally conservative. It does not try to prove an
insurance valuation is good; it decides which valuation model is allowed to
touch the security and records explicit blocker codes when the model fit is
ambiguous.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from app.config import AppConfig
from app.insurance.sources import (
    cached_submission_profile,
    latest_cached_filing_text,
    latest_company_profile,
    latest_sector,
)


SECURITY_COMMON = "common"
SECURITY_PREFERRED = "preferred"
SECURITY_DEPOSITARY = "depositary_preferred"
SECURITY_DEBT = "debt"
SECURITY_UNIT = "unit"
SECURITY_WARRANT = "warrant"
SECURITY_RIGHT = "right"
SECURITY_UNKNOWN = "SECURITY_TYPE_UNKNOWN"

ISSUER_INSURANCE_UNDERWRITER = "insurance_underwriter"
ISSUER_INSURANCE_SERVICES = "insurance_services"
ISSUER_FINANCIAL_SERVICES = "financial_services"
ISSUER_NON_INSURER = "non_insurer"
ISSUER_UNKNOWN = "ISSUER_TYPE_UNKNOWN"

MODEL_ROUTED = "ROUTED"
MODEL_NOT_APPLICABLE = "NOT_APPLICABLE"
MODEL_BLOCKED = "MODEL_BLOCKED"


@dataclass(frozen=True)
class SecurityRoutingResult:
    ticker: str
    security_type: str
    issuer_type: str
    insurance_subtype: str | None
    accounting_regime: str
    model_status: str
    reason_codes: list[str] = field(default_factory=list)
    model_fit_warnings: list[str] = field(default_factory=list)
    confidence: str = "LOW"
    derived_from: list[str] = field(default_factory=list)
    company_name: str | None = None
    sector: str | None = None
    filing_evidence: dict[str, Any] = field(default_factory=dict)
    issuer_name: str | None = None
    issuer_primary_ticker: str | None = None
    issuer_listed_tickers: list[str] = field(default_factory=list)
    security_identity_status: str = "UNVERIFIED"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_PREFERRED_RE = re.compile(
    r"\b(preferred|preference|depositary share|depositary shares|non-cumulative|cumulative redeemable|series [a-z])\b",
    re.IGNORECASE,
)
_DEBT_RE = re.compile(
    r"\b(senior notes?|subordinated notes?|debentures?|notes due)\b", re.IGNORECASE
)
_UNIT_RE = re.compile(r"\b(units?|unit)\b", re.IGNORECASE)
_WARRANT_RE = re.compile(r"\b(warrants?|warrant)\b", re.IGNORECASE)
_RIGHT_RE = re.compile(r"\b(rights?|subscription rights?)\b", re.IGNORECASE)
_COMMON_RE = re.compile(
    r"\b(common stock|common shares|ordinary shares|class [abc] common)\b", re.IGNORECASE
)

_INSURANCE_RE = re.compile(
    r"\b(insurance|insurer|policyholder|premiums?|underwriting|claims?|loss reserves?|annuit|reinsurance|ceded|assumed)\b",
    re.IGNORECASE,
)
_STRONG_UNDERWRITER_RE = re.compile(
    r"\b(writes?|underwrites?|issues?|sells?)\s+(?:\w+\s+){0,5}(insurance|policies|policy|annuities?|reinsurance)\b"
    r"|\b(gross|net)\s+(written|earned)\s+premiums?\b"
    r"|\bpremiums?\s+(written|earned|ceded|assumed)\b"
    r"|\bcombined ratio\b|\bloss ratio\b|\bloss reserves?\b|\bpolicyholder\b"
    r"|\bclaims and claim adjustment\b|\bloss adjustment expenses?\b",
    re.IGNORECASE,
)
_LIFE_RE = re.compile(
    r"\b(life insurance|annuit|market risk benefits?|deferred acquisition costs?|dac|long-duration|policyholder account)\b",
    re.IGNORECASE,
)
_PC_RE = re.compile(
    r"\b(property and casualty|p&c|commercial lines|personal lines|catastrophe|combined ratio|loss ratio|loss reserves?|claims and claim adjustment|loss adjustment expenses?)\b",
    re.IGNORECASE,
)
_REINSURANCE_RE = re.compile(
    r"\b(reinsurance|reinsurer|retrocession|assumed premiums?|ceded premiums?)\b", re.IGNORECASE
)
_BROKER_RE = re.compile(
    r"\b(brokerage|broker|agency commissions?|advisory fees?|risk consulting|benefits consulting)\b",
    re.IGNORECASE,
)
_TITLE_MORTGAGE_RE = re.compile(
    r"\b(title insurance|mortgage insurance|crop insurance|credit insurance)\b", re.IGNORECASE
)


def _ticker_suffix_type(ticker: str) -> str | None:
    upper = ticker.upper()
    if upper.endswith(("-WT", ".WT", "-WS", ".WS", "WS")):
        return SECURITY_WARRANT
    if upper.endswith(("-RT", ".RT", "RT")):
        return SECURITY_RIGHT
    if upper.endswith(("-UN", ".UN", "U")) and len(upper) <= 6:
        return SECURITY_UNIT
    if re.search(r"[-.]P[A-Z0-9]?$", upper):
        return SECURITY_PREFERRED
    return None


def _classify_security_type(ticker: str, name: str, text: str) -> tuple[str, list[str], list[str]]:
    derived_from: list[str] = []
    reason_codes: list[str] = []

    if _DEBT_RE.search(name):
        return SECURITY_DEBT, ["company_name:debt_security"], ["SECURITY_IS_DEBT"]
    if _PREFERRED_RE.search(name):
        derived_from.append("company_name:preferred_terms")
        if "depositary share" in name.lower():
            return SECURITY_DEPOSITARY, derived_from, ["SECURITY_IS_DEPOSITARY_PREFERRED"]
        return SECURITY_PREFERRED, derived_from, ["SECURITY_IS_PREFERRED"]
    if _WARRANT_RE.search(name):
        return SECURITY_WARRANT, ["company_name:warrant"], ["SECURITY_IS_WARRANT"]
    if _RIGHT_RE.search(name):
        return SECURITY_RIGHT, ["company_name:right"], ["SECURITY_IS_RIGHT"]
    if _UNIT_RE.search(name) and not _INSURANCE_RE.search(name):
        return SECURITY_UNIT, ["company_name:unit"], ["SECURITY_IS_UNIT"]

    suffix_type = _ticker_suffix_type(ticker)
    if suffix_type:
        return suffix_type, ["ticker_suffix:security_hint"], [f"SECURITY_IS_{suffix_type.upper()}"]

    if name.strip() or text.strip():
        return SECURITY_COMMON, derived_from + ["default:operating_common"], reason_codes
    return SECURITY_UNKNOWN, derived_from, ["SECURITY_TYPE_UNKNOWN"]


def _apply_security_identity_policy(
    *,
    ticker: str,
    security_type: str,
    name: str,
    submission_profile: dict[str, Any],
    security_sources: list[str] | None = None,
) -> tuple[str, str, list[str], list[str], list[str]]:
    """Conservatively verify common equity identity for multi-security issuers."""
    if security_type == SECURITY_UNKNOWN:
        return security_type, "UNVERIFIED_UNKNOWN_SECURITY", [], [], []
    if security_type != SECURITY_COMMON:
        # A ticker-suffix hint is weak evidence: a real operating common can
        # simply end in U/W/R (KEQU-class false positive). When the hint is
        # the ONLY source (name and filing text said nothing) and the SEC
        # submissions profile shows this IS the issuer's primary or only
        # listed ticker, the hint is wrong — a genuine SPAC unit/warrant
        # trades alongside the issuer's base ticker, not as its primary.
        if "ticker_suffix:security_hint" in (security_sources or []):
            listed_tickers = [
                str(item).upper()
                for item in (submission_profile.get("issuer_listed_tickers") or [])
                if str(item).strip()
            ]
            primary = str(submission_profile.get("issuer_primary_ticker") or "").upper() or None
            if (
                listed_tickers
                and ticker in listed_tickers
                and (len(listed_tickers) == 1 or ticker == primary)
            ):
                return (
                    SECURITY_COMMON,
                    "VERIFIED_COMMON_PRIMARY_TICKER_SUFFIX_OVERRIDE",
                    [],
                    ["submission_cache:primary_ticker_overrides_suffix_hint"],
                    [],
                )
        return security_type, "EXPLICIT_NON_COMMON", [], [], []

    listed_tickers = [
        str(item).upper()
        for item in (submission_profile.get("issuer_listed_tickers") or [])
        if str(item).strip()
    ]
    primary = str(submission_profile.get("issuer_primary_ticker") or "").upper() or None
    if _COMMON_RE.search(name):
        return security_type, "VERIFIED_COMMON_BY_NAME", [], ["company_name:common_equity"], []
    if listed_tickers:
        if ticker not in listed_tickers:
            return (
                SECURITY_UNKNOWN,
                "UNVERIFIED_NOT_LISTED_FOR_CIK",
                ["SECURITY_TYPE_UNKNOWN", "SECURITY_IDENTITY_UNVERIFIED"],
                ["submission_cache:ticker_not_listed"],
                [],
            )
        if len(listed_tickers) == 1 or ticker == primary:
            return (
                security_type,
                "VERIFIED_COMMON_PRIMARY_TICKER",
                [],
                ["submission_cache:primary_or_only_ticker"],
                [],
            )
        return (
            SECURITY_UNKNOWN,
            "UNVERIFIED_MULTI_SECURITY_NON_PRIMARY",
            [
                "SECURITY_TYPE_UNKNOWN",
                "MULTI_SECURITY_CIK_NON_PRIMARY_TICKER",
                "SECURITY_IDENTITY_UNVERIFIED",
            ],
            ["submission_cache:non_primary_listed_ticker"],
            [],
        )
    return (
        SECURITY_UNKNOWN,
        "UNVERIFIED_NO_LISTING_PROFILE",
        ["SECURITY_TYPE_UNKNOWN", "SECURITY_IDENTITY_UNVERIFIED"],
        ["submission_cache:missing"],
        [],
    )


def _classify_issuer_and_subtype(
    name: str, sector: str | None, text: str
) -> tuple[str, str | None, list[str], list[str], str]:
    sector_lower = str(sector or "").strip().lower()
    text_haystack = f"{name} {text[:350_000]}"
    haystack = f"{name} {sector or ''} {text[:350_000]}"
    derived_from: list[str] = []
    warnings: list[str] = []

    sector_is_insurance = sector_lower == "insurance"
    has_strong_underwriter_evidence = bool(_STRONG_UNDERWRITER_RE.search(text_haystack))
    has_insurance = sector_is_insurance or has_strong_underwriter_evidence
    has_broker = bool(_BROKER_RE.search(haystack))
    has_pc = bool(_PC_RE.search(haystack))
    has_reinsurance = bool(_REINSURANCE_RE.search(haystack))
    has_underwriting = bool(_LIFE_RE.search(haystack) or has_pc or has_reinsurance)
    explicit_reinsurer = bool(
        re.search(
            r"\breinsurer\b|\breinsurance\s+(?:company|group|ltd|corp|corporation)\b",
            name,
            re.IGNORECASE,
        )
    )

    if has_broker and not has_underwriting:
        return (
            ISSUER_INSURANCE_SERVICES,
            "broker_services",
            ["BROKER_SERVICES_NOT_UNDERWRITER"],
            derived_from + ["text:broker_services"],
            "MEDIUM",
        )
    if has_insurance:
        if sector_is_insurance:
            derived_from.append("sector:insurance")
        if has_strong_underwriter_evidence:
            derived_from.append("text:insurance_underwriting_evidence")
        if _TITLE_MORTGAGE_RE.search(haystack):
            return (
                ISSUER_INSURANCE_UNDERWRITER,
                "title_mortgage_specialty",
                [],
                derived_from + ["text:specialty_insurance"],
                "HIGH",
            )
        if _LIFE_RE.search(haystack):
            return (
                ISSUER_INSURANCE_UNDERWRITER,
                "life_annuity",
                [],
                derived_from + ["text:life_annuity"],
                "HIGH",
            )
        if has_reinsurance and explicit_reinsurer:
            return (
                ISSUER_INSURANCE_UNDERWRITER,
                "reinsurer",
                [],
                derived_from + ["text:reinsurance"],
                "HIGH",
            )
        if has_pc:
            return (
                ISSUER_INSURANCE_UNDERWRITER,
                "pc_insurer",
                [],
                derived_from + ["text:pc_insurance"],
                "HIGH",
            )
        if has_reinsurance:
            return (
                ISSUER_INSURANCE_UNDERWRITER,
                "reinsurer",
                [],
                derived_from + ["text:reinsurance"],
                "HIGH",
            )
        warnings.append("INSURANCE_SUBTYPE_UNCLEAR")
        return ISSUER_INSURANCE_UNDERWRITER, "insurer_unknown", warnings, derived_from, "MEDIUM"

    if re.search(
        r"\b(bank|broker-dealer|asset management|investment adviser|exchange)\b",
        haystack,
        re.IGNORECASE,
    ):
        return ISSUER_FINANCIAL_SERVICES, None, [], ["text:financial_services"], "MEDIUM"
    if name.strip() or sector:
        return ISSUER_NON_INSURER, None, [], ["default:non_insurer"], "HIGH"
    return ISSUER_UNKNOWN, None, ["ISSUER_TYPE_UNKNOWN"], [], "LOW"


def _accounting_regime(text: str, subtype: str | None) -> tuple[str, list[str], list[str]]:
    lower = text[:220_000].lower()
    derived_from: list[str] = []
    warnings: list[str] = []
    if "ifrs 17" in lower or "contractual service margin" in lower or re.search(r"\bcsm\b", lower):
        return "IFRS_17", ["filing_text:ifrs17"], warnings
    if (
        "ldti" in lower
        or "long-duration targeted improvements" in lower
        or "market risk benefits" in lower
        or (subtype == "life_annuity" and "deferred acquisition costs" in lower)
    ):
        return "US_GAAP_LDTI", ["filing_text:ldti"], warnings
    if subtype == "life_annuity":
        warnings.append("ACCOUNTING_REGIME_UNKNOWN_FOR_LIFE")
        return "ACCOUNTING_REGIME_UNKNOWN", derived_from, warnings
    return "US_GAAP", derived_from + ["default:us_gaap_or_not_disclosed"], warnings


def route_security(
    ticker: str,
    *,
    as_of_date: str | None = None,
    pipeline_version: str = "v1",
    issuer_cik: str | None = None,
    aliases: Sequence[str] = (),
    db_path: str | Path | None = None,
    cfg: AppConfig | None = None,
) -> SecurityRoutingResult:
    """Classify security type, issuer type, insurance subtype, and accounting regime."""
    upper = ticker.upper()
    normalized_pipeline = str(pipeline_version or "v1").strip().lower()
    if normalized_pipeline not in {"v1", "v2"}:
        raise ValueError("pipeline_version must be v1 or v2")
    is_v2 = normalized_pipeline == "v2"
    if is_v2 and not as_of_date:
        raise ValueError("v2 insurance routing requires as_of_date")

    if is_v2:
        profile = latest_company_profile(
            upper,
            issuer_cik=issuer_cik,
            aliases=aliases,
            db_path=db_path,
            cfg=cfg,
        )
        sector = latest_sector(
            upper,
            as_of_date=as_of_date,
            aliases=aliases,
            db_path=db_path,
            cfg=cfg,
        )
        filing_text, filing_meta = latest_cached_filing_text(
            upper,
            as_of_date=as_of_date,
            issuer_cik=issuer_cik,
            aliases=aliases,
            db_path=db_path,
            cfg=cfg,
        )
    else:
        profile = latest_company_profile(upper)
        sector = latest_sector(upper)
        filing_text, filing_meta = latest_cached_filing_text(upper, as_of_date=as_of_date)
    cik = issuer_cik or profile.get("cik") or filing_meta.get("cik")
    if not cik and not is_v2:
        # Census-discovered names have no companies row or cached filing yet;
        # the SEC ticker registry still resolves them so the submissions
        # identity profile (listed tickers, issuer name) stays available.
        try:
            from app.ingest.cik_registry import resolve as _resolve_cik

            cik = _resolve_cik(upper)
        except Exception:  # noqa: BLE001 - identity stays unverified without a CIK
            cik = None
    submission_profile = (
        cached_submission_profile(cik, cfg=cfg) if is_v2 else cached_submission_profile(cik)
    )
    issuer_name = str(submission_profile.get("issuer_name") or "")
    name = str(profile.get("name") or issuer_name or "")

    security_type, security_sources, security_reasons = _classify_security_type(
        upper, name, filing_text
    )
    security_type, identity_status, identity_reasons, identity_sources, identity_warnings = (
        _apply_security_identity_policy(
            ticker=upper,
            security_type=security_type,
            name=name,
            submission_profile=submission_profile,
            security_sources=security_sources,
        )
    )
    submission_primary = str(submission_profile.get("issuer_primary_ticker") or "").strip().upper()
    if (
        is_v2
        and security_type == SECURITY_COMMON
        and submission_primary
        and upper != submission_primary
    ):
        # A company-name "common stock" label proves neither class nor ADR
        # conversion terms. Issuer-wide book value per share cannot be paired
        # with a secondary-class/ADR quote until authoritative ratio evidence
        # is available from the identity stage.
        security_type = SECURITY_UNKNOWN
        identity_status = "UNVERIFIED_NON_PRIMARY_SECURITY_RATIO"
        identity_reasons = list(
            dict.fromkeys(
                [
                    *identity_reasons,
                    "SECURITY_TYPE_UNKNOWN",
                    "NON_PRIMARY_SECURITY_RATIO_UNRESOLVED",
                    "SECURITY_IDENTITY_UNVERIFIED",
                ]
            )
        )
        identity_sources = list(
            dict.fromkeys([*identity_sources, "submission_cache:non_primary_ticker"])
        )
    if identity_status == "VERIFIED_COMMON_PRIMARY_TICKER_SUFFIX_OVERRIDE":
        # The suffix hint was refuted by the issuer's own listing profile;
        # drop the stale SECURITY_IS_* reason so downstream surfaces do not
        # carry a contradicted code.
        security_reasons = []
    issuer_type, subtype, issuer_notes, issuer_sources, issuer_confidence = (
        _classify_issuer_and_subtype(name, sector, filing_text)
    )
    accounting_regime, accounting_sources, accounting_warnings = _accounting_regime(
        filing_text, subtype
    )

    reason_codes = list(
        dict.fromkeys(
            security_reasons + identity_reasons + [note for note in issuer_notes if note.isupper()]
        )
    )
    warnings = (
        [note for note in issuer_notes if not note.isupper()]
        + identity_warnings
        + accounting_warnings
    )
    derived_from = list(
        dict.fromkeys(security_sources + identity_sources + issuer_sources + accounting_sources)
    )

    if security_type == SECURITY_UNKNOWN:
        model_status = MODEL_BLOCKED
    elif security_type != SECURITY_COMMON:
        model_status = MODEL_ROUTED
    elif issuer_type == ISSUER_INSURANCE_UNDERWRITER:
        model_status = MODEL_ROUTED
    else:
        model_status = MODEL_NOT_APPLICABLE

    confidence = "LOW"
    if security_type != SECURITY_UNKNOWN and issuer_confidence == "HIGH":
        confidence = "HIGH"
    elif security_type != SECURITY_UNKNOWN and issuer_confidence in ("HIGH", "MEDIUM"):
        confidence = "MEDIUM"

    return SecurityRoutingResult(
        ticker=upper,
        security_type=security_type,
        issuer_type=issuer_type,
        insurance_subtype=subtype,
        accounting_regime=accounting_regime,
        model_status=model_status,
        reason_codes=reason_codes,
        model_fit_warnings=list(dict.fromkeys(warnings)),
        confidence=confidence,
        derived_from=derived_from,
        company_name=name or None,
        sector=sector,
        filing_evidence=filing_meta,
        issuer_name=issuer_name or None,
        issuer_primary_ticker=submission_profile.get("issuer_primary_ticker"),
        issuer_listed_tickers=[
            str(item).upper() for item in (submission_profile.get("issuer_listed_tickers") or [])
        ],
        security_identity_status=identity_status,
    )
