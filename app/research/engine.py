from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import get_config
from app.db import get_db, get_state, utc_now_iso
from app.logging import get_logger
from app.research.adapters import AdapterContext, ResearchAdapter, build_adapter_chain
from app.research.research_quality import score_research_packet
from app.research.schemas import (
    ClaimTrace,
    EvidenceGap,
    EvidenceItem,
    EvidenceLinkedEntry,
    KeyQuestion,
    ResearchPacket,
    ResearchPlanStep,
    validate_research_packet,
)
from app.research.signals import compute_research_signals, persist_research_signals
from app.score.ranker import score_ticker
from app.util.hashing import sha256_file, sha256_text
from app.util.issuer_classification import ISSUER_CLASS_FINANCIAL, resolve_issuer_classification


logger = get_logger(__name__)


def _issuer_classification_from_packet(packet: dict[str, Any]) -> str:
    fundamentals = packet.get("fundamentals", {}) if isinstance(packet, dict) else {}
    classification = str((fundamentals or {}).get("issuer_classification") or "").strip().lower()
    if classification:
        return classification
    financials = packet.get("financials", []) if isinstance(packet, dict) else []
    line_items = [str(row.get("line_item") or "") for row in financials if isinstance(row, dict)]
    texts = [
        str(((row.get("citation") or {}).get("snippet")) or "")
        for row in financials
        if isinstance(row, dict)
    ]
    # SIC-first (VOE_ISSUER_CLASSIFICATION_BY_SIC); the tag-name substring rule only
    # answers when no SIC code is on file for this ticker.
    return resolve_issuer_classification(
        ticker=packet.get("ticker") if isinstance(packet, dict) else None,
        texts=texts,
        line_items=line_items,
    )[0]


def _latest_packet_row(conn, ticker: str, as_of_date: str | None = None):
    if as_of_date:
        row = conn.execute(
            """
            SELECT as_of_date, packet_path
            FROM evidence_packets
            WHERE ticker = ? AND as_of_date = ?
            LIMIT 1
            """,
            (ticker, as_of_date),
        ).fetchone()
        if row:
            return row
        # A run asked "as of" a date is served the newest packet AT OR BEFORE
        # that date, never one published after it. Without the bound a
        # backdated run silently became a present-day run — no refusal, no
        # warning — and anything that walks backwards through time to test how
        # the system would have behaved was reading tomorrow's evidence.
        return conn.execute(
            """
            SELECT as_of_date, packet_path
            FROM evidence_packets
            WHERE ticker = ? AND as_of_date <= ?
            ORDER BY as_of_date DESC
            LIMIT 1
            """,
            (ticker, as_of_date),
        ).fetchone()
    return conn.execute(
        """
        SELECT as_of_date, packet_path
        FROM evidence_packets
        WHERE ticker = ?
        ORDER BY as_of_date DESC
        LIMIT 1
        """,
        (ticker,),
    ).fetchone()


def _company_metadata(conn, ticker: str) -> dict[str, str]:
    row = conn.execute(
        """
        SELECT name, ir_rss_url, homepage_url
        FROM companies
        WHERE ticker = ?
        LIMIT 1
        """,
        (ticker,),
    ).fetchone()
    if not row:
        return {"name": "", "ir_rss_url": "", "homepage_url": ""}
    return {
        "name": row["name"] or "",
        "ir_rss_url": row["ir_rss_url"] or "",
        "homepage_url": row["homepage_url"] or "",
    }


def _pick_supporting(items: list[EvidenceItem], n: int = 2) -> list[EvidenceItem]:
    return items[:n] if len(items) >= n else items[:]


def _evidence_haystack(item: EvidenceItem) -> str:
    """Concatenated, lowercased searchable text for relevance scoring."""
    parts: list[str] = [item.source_title or "", item.excerpt_text or ""]
    for citation in item.citations:
        parts.append(citation.snippet or "")
        parts.append(citation.section_label or "")
    return " ".join(parts).lower()


def _relevance_score(
    item: EvidenceItem,
    keywords: list[str],
    preferred_source_types: frozenset[str],
) -> int:
    """Score an evidence item against bucket keywords + source-type preference.

    Each keyword that appears (case-insensitive substring) in the item's
    title/excerpt/citation text contributes one point; a preferred source_type
    contributes one additional point. Higher is more relevant.
    """
    haystack = _evidence_haystack(item)
    score = sum(1 for kw in keywords if kw.lower() in haystack)
    if preferred_source_types and item.source_type in preferred_source_types:
        score += 1
    return score


def _select_relevant(
    items: list[EvidenceItem],
    keywords: list[str],
    *,
    n: int = 2,
    exclude_ids: set[str] | frozenset[str] = frozenset(),
    preferred_source_types: set[str] | frozenset[str] = frozenset(),
) -> list[EvidenceItem]:
    """Select up to ``n`` evidence items by relevance to a claim's keywords.

    Items whose ids are in ``exclude_ids`` are never returned (used to keep
    opposing buckets disjoint). Selection is by descending relevance score with
    a deterministic ascending-id tiebreaker; when nothing matches the keywords
    the result degrades to id-sorted order so the bucket is still populated with
    real evidence rather than a blind positional slice.
    """
    preferred = frozenset(preferred_source_types)
    candidates = [item for item in items if item.id not in exclude_ids]
    if not candidates:
        return []
    ranked = sorted(
        candidates,
        key=lambda item: (-_relevance_score(item, keywords, preferred), item.id),
    )
    return ranked[:n]


def _history_values(packet: dict[str, Any], key: str) -> list[float]:
    fundamentals = packet.get("fundamentals", {}) if isinstance(packet.get("fundamentals"), dict) else {}
    derived = fundamentals.get("derived_signals") if isinstance(fundamentals.get("derived_signals"), dict) else {}
    payload = derived.get(key) if isinstance(derived, dict) else None
    values = payload.get("value") if isinstance(payload, dict) else None
    if not isinstance(values, list):
        return []
    out: list[float] = []
    for item in values:
        if not isinstance(item, dict):
            continue
        value = item.get("value")
        if isinstance(value, (int, float)):
            out.append(float(value))
    return out


def _detect_base_gaps(packet: dict[str, Any]) -> list[EvidenceGap]:
    gaps: list[EvidenceGap] = []
    metrics = packet.get("fundamentals", {})
    is_financial = _issuer_classification_from_packet(packet) == ISSUER_CLASS_FINANCIAL
    required_metrics = (
        ["revenue", "deposits", "loans", "total_assets"]
        if is_financial
        else ["revenue", "operating_margin", "fcf", "net_debt"]
    )
    for metric in required_metrics:
        if metrics.get(metric) == "UNKNOWN":
            gaps.append(
                EvidenceGap(
                    gap_id=f"GAP_UNKNOWN_{metric.upper()}",
                    severity="medium",
                    summary=f"Missing or UNKNOWN metric: {metric}",
                    source_type="fundamentals",
                    recommended_action=f"Hydrate companyfacts and dossier-backed evidence for `{metric}`, rebuild the evidence packet, and rerun research.",
                )
            )
    if (
        is_financial
        and isinstance(metrics.get("loans"), (int, float))
        and metrics.get("allowance_for_credit_losses") == "UNKNOWN"
    ):
        gaps.append(
            EvidenceGap(
                gap_id="GAP_UNKNOWN_ALLOWANCE_FOR_CREDIT_LOSSES",
                severity="low",
                summary="Missing or UNKNOWN metric: allowance_for_credit_losses",
                source_type="fundamentals",
                recommended_action="Hydrate companyfacts and dossier-backed evidence for `allowance_for_credit_losses`, rebuild the evidence packet, and rerun research.",
            )
        )
    if is_financial and isinstance(metrics.get("loans"), (int, float)):
        for metric in ("provision_for_credit_losses", "net_charge_offs", "nonaccrual_loans"):
            if metrics.get(metric) in {None, "UNKNOWN"}:
                gaps.append(
                    EvidenceGap(
                        gap_id=f"GAP_UNKNOWN_{metric.upper()}",
                        severity="low",
                        summary=f"Missing or UNKNOWN metric: {metric}",
                        source_type="fundamentals",
                        recommended_action=f"Hydrate companyfacts and filing-backed evidence for `{metric}`, rebuild the evidence packet, and rerun research.",
                    )
                )

    market_price = packet.get("valuations", {}).get("reverse_dcf", {}).get("inputs", {}).get("market_price")
    if market_price == "UNKNOWN":
        gaps.append(
            EvidenceGap(
                gap_id="GAP_MARKET_PRICE_UNKNOWN",
                severity="low",
                summary="Market price unavailable for reverse DCF feasibility checks.",
                source_type="valuation",
                recommended_action="Enable a price provider or keep classification in research-only/watchlist mode.",
            )
        )
    return gaps


def _build_key_questions() -> list[KeyQuestion]:
    prompts = [
        "Is reported revenue growth durable across the next two filings?",
        "Are operating margins reverting or structurally shifting?",
        "Is operating cash flow quality consistent with reported earnings?",
        "Do debt maturities and covenants create refinancing pressure?",
        "Is dilution (SBC, ATM, convertibles) likely to impair per-share value?",
        "Is customer concentration increasing downside risk?",
        "Do non-GAAP adjustments reconcile cleanly to GAAP trends?",
    ]
    return [KeyQuestion(question_id=f"Q{i+1}", question=q) for i, q in enumerate(prompts)]


def _citations(items: list[EvidenceItem], limit: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in items:
        for citation in item.citations[:1]:
            out.append(citation.model_dump())
            if len(out) >= limit:
                return out
    return out


def _valuation_outputs(packet: dict[str, Any], method: str) -> dict[str, Any]:
    valuations = packet.get("valuations", {})
    payload = valuations.get(method, {}) if isinstance(valuations, dict) else {}
    if not isinstance(payload, dict):
        return {}
    outputs = payload.get("outputs")
    if isinstance(outputs, dict):
        return outputs
    return payload


def _dcf_base(packet: dict[str, Any]) -> Any:
    dcf = _valuation_outputs(packet, "dcf")
    if isinstance(dcf.get("base"), (int, float)):
        return dcf.get("base")
    legacy = _valuation_outputs(packet, "dcf_lite")
    per_share = legacy.get("per_share_range") if isinstance(legacy.get("per_share_range"), dict) else {}
    return per_share.get("base", "UNKNOWN")


def _secondary_valuation_base(packet: dict[str, Any]) -> tuple[str, Any, str]:
    epv = _valuation_outputs(packet, "epv")
    if isinstance(epv.get("value_per_share"), (int, float)):
        return "EPV", epv.get("value_per_share"), "valuations.epv.outputs.value_per_share"
    graham = _valuation_outputs(packet, "graham")
    if isinstance(graham.get("value_per_share"), (int, float)):
        return "Graham", graham.get("value_per_share"), "valuations.graham.outputs.value_per_share"
    legacy_mult = _valuation_outputs(packet, "multiples")
    per_share = legacy_mult.get("per_share_range") if isinstance(legacy_mult.get("per_share_range"), dict) else {}
    return "multiples", per_share.get("base", "UNKNOWN"), "valuations.multiples.outputs.per_share_range.base"


_FINDINGS_BANK_KEYWORDS = [
    "deposit", "loan", "allowance", "credit", "provision", "charge-off",
    "reserve", "funding", "liquidity", "capital",
]
_FINDINGS_KEYWORDS = [
    "valuation", "dcf", "earnings", "balance sheet", "liquidity", "net debt",
    "cash flow", "margin", "revenue",
]
_RISK_KEYWORDS = ["refinanc", "covenant", "dilution", "maturity", "default", "going concern", "risk"]
_CATALYST_KEYWORDS = ["earnings", "guidance", "filing", "10-q", "10-k", "8-k", "outlook", "results"]
_DISCONFIRMING_KEYWORDS = [
    "cash flow", "operating cash", "dilution", "stock-based", "sbc",
    "shares outstanding", "weaker", "decline", "deteriorat",
]
_FILING_SOURCE_TYPES = frozenset({"EDGAR", "sec_exhibit"})


def _build_findings(packet: dict[str, Any], evidence_items: list[EvidenceItem]) -> list[EvidenceLinkedEntry]:
    if _issuer_classification_from_packet(packet) == ISSUER_CLASS_FINANCIAL:
        support = _select_relevant(
            evidence_items, _FINDINGS_BANK_KEYWORDS, n=3,
            preferred_source_types=_FILING_SOURCE_TYPES,
        )
        ids = [item.id for item in support]
        fundamentals = packet.get("fundamentals", {}) if isinstance(packet.get("fundamentals"), dict) else {}
        deposits = fundamentals.get("deposits", "UNKNOWN")
        total_assets = fundamentals.get("total_assets", "UNKNOWN")
        loans = fundamentals.get("loans", "UNKNOWN")
        allowance = fundamentals.get("allowance_for_credit_losses", "UNKNOWN")
        provision = fundamentals.get("provision_for_credit_losses", "UNKNOWN")
        charge_offs = fundamentals.get("net_charge_offs", "UNKNOWN")
        allowance_history = _history_values(packet, "allowance_to_loans_history")
        provision_history = _history_values(packet, "provision_to_loans_history")
        charge_off_history = _history_values(packet, "net_charge_offs_to_loans_history")
        history_bits: list[str] = []
        if allowance_history:
            history_bits.append(
                f"allowance-to-loans history spans {min(allowance_history):.2%} to {max(allowance_history):.2%}"
            )
        if provision_history:
            history_bits.append(
                f"provision-to-loans history spans {min(provision_history):.2%} to {max(provision_history):.2%}"
            )
        if charge_off_history:
            history_bits.append(
                f"net charge-offs-to-loans history spans {min(charge_off_history):.2%} to {max(charge_off_history):.2%}"
            )
        history_text = f" Historical reserve coverage shows {', '.join(history_bits)}." if history_bits else ""
        return [
            EvidenceLinkedEntry(
                entry_id="F1",
                summary=(
                    f"Bank-focused evidence is led by deposits={deposits}, total_assets={total_assets}, "
                    f"loans={loans}, allowance for credit losses={allowance}, provision for credit losses={provision}, "
                    f"and net charge-offs={charge_offs}; funding mix, credit quality, reserve coverage, and "
                    f"debt/liquidity disclosures should anchor the next validation step.{history_text}"
                ),
                evidence_item_ids=ids,
                citations=_citations(support, limit=3),
                derived_from=[
                    "fundamentals.deposits",
                    "fundamentals.total_assets",
                    "fundamentals.loans",
                    "fundamentals.allowance_for_credit_losses",
                    "fundamentals.provision_for_credit_losses",
                    "fundamentals.net_charge_offs",
                    "fundamentals.derived_signals.allowance_to_loans_history",
                    "fundamentals.derived_signals.provision_to_loans_history",
                    "fundamentals.derived_signals.net_charge_offs_to_loans_history",
                ],
            )
        ]

    support = _select_relevant(
        evidence_items, _FINDINGS_KEYWORDS, n=3,
        preferred_source_types=_FILING_SOURCE_TYPES,
    )
    ids = [item.id for item in support]

    dcf_base = _dcf_base(packet)
    secondary_label, secondary_base, secondary_ref = _secondary_valuation_base(packet)

    findings = [
        EvidenceLinkedEntry(
            entry_id="F1",
            summary=(
                f"Valuation gate produced bounded ranges (DCF base={dcf_base}, {secondary_label} base={secondary_base}) "
                "from filing-derived inputs."
            ),
            evidence_item_ids=ids,
            citations=_citations(support, limit=3),
            derived_from=[
                "valuations.dcf.outputs.base",
                secondary_ref,
            ],
        ),
        EvidenceLinkedEntry(
            entry_id="F2",
            summary="Balance sheet and liquidity interpretation is tied to cited filing and IR evidence.",
            evidence_item_ids=ids,
            citations=_citations(support, limit=2),
            derived_from=["fundamentals.net_debt", "fundamentals.liquidity_stress_score"],
        ),
    ]
    return findings


def _build_risks(evidence_items: list[EvidenceItem], gaps: list[EvidenceGap]) -> list[EvidenceLinkedEntry]:
    support = _select_relevant(
        evidence_items, _RISK_KEYWORDS, n=2,
        preferred_source_types=_FILING_SOURCE_TYPES,
    )
    ids = [item.id for item in support]
    risks = [
        EvidenceLinkedEntry(
            entry_id="RISK1",
            summary="Refinancing, covenant, and dilution pathways could dominate valuation realization.",
            evidence_item_ids=ids,
            citations=_citations(support, limit=2),
        )
    ]
    if gaps:
        risks.append(
            EvidenceLinkedEntry(
                entry_id="RISK2",
                summary=f"Evidence gaps reduce confidence: {gaps[0].summary}",
                evidence_item_ids=ids,
                citations=_citations(support, limit=1),
                derived_from=["research.evidence_gaps"],
            )
        )
    return risks


def _build_catalysts(packet: dict[str, Any], evidence_items: list[EvidenceItem]) -> list[EvidenceLinkedEntry]:
    filings = packet.get("filings_used", [])
    support = _select_relevant(
        evidence_items, _CATALYST_KEYWORDS, n=2,
        preferred_source_types=_FILING_SOURCE_TYPES,
    )
    ids = [item.id for item in support]
    if filings:
        latest = filings[0]
        summary = f"Next catalyst is the filing cycle after {latest.get('form_type')} filed on {latest.get('filing_date')}."
    else:
        summary = "Next catalyst is the next SEC periodic filing and related exhibits."
    return [
        EvidenceLinkedEntry(
            entry_id="CAT1",
            summary=summary,
            evidence_item_ids=ids,
            citations=_citations(support, limit=2),
            derived_from=["filings_used"],
        )
    ]


def _build_disconfirming(
    evidence_items: list[EvidenceItem],
    exclude_ids: set[str] | frozenset[str] = frozenset(),
) -> list[EvidenceLinkedEntry]:
    # Prefer evidence disjoint from the confirming findings so an opposing claim
    # is never "supported" by the exact same citations as the claim it opposes.
    support = _select_relevant(
        evidence_items, _DISCONFIRMING_KEYWORDS, n=2,
        exclude_ids=exclude_ids,
        preferred_source_types=_FILING_SOURCE_TYPES,
    )
    if not support:
        # No disjoint evidence available (tiny pool) — fall back to any evidence
        # so the required disconfirming bucket stays populated.
        support = _select_relevant(
            evidence_items, _DISCONFIRMING_KEYWORDS, n=2,
            preferred_source_types=_FILING_SOURCE_TYPES,
        )
    ids = [item.id for item in support]
    return [
        EvidenceLinkedEntry(
            entry_id="D1",
            summary=(
                "Disconfirming check: next 10-Q shows weaker operating cash flow and rising dilution versus current packet."
            ),
            evidence_item_ids=ids,
            citations=_citations(support, limit=2),
            derived_from=["fundamentals.cfo", "fundamentals.fcf", "extracted_facts.sbc_dilution_signal"],
        )
    ]


def _assert_opposing_buckets_disjoint(
    findings: list[EvidenceLinkedEntry],
    disconfirming: list[EvidenceLinkedEntry],
    *,
    evidence_pool_size: int,
) -> None:
    """Guard: confirming findings and disconfirming evidence must not share ids.

    Disjointness is only enforceable when the evidence pool is large enough to
    split (>= 2 distinct items). With a single item, overlap is unavoidable and
    therefore tolerated.
    """
    if evidence_pool_size < 2:
        return
    finding_ids: set[str] = set()
    for entry in findings:
        finding_ids.update(entry.evidence_item_ids)
    disconfirming_ids: set[str] = set()
    for entry in disconfirming:
        disconfirming_ids.update(entry.evidence_item_ids)
    shared = finding_ids & disconfirming_ids
    if shared:
        raise ValueError(
            f"findings and disconfirming_evidence share evidence ids: {sorted(shared)}"
        )


def _build_next_actions(key_questions: list[KeyQuestion], gaps: list[EvidenceGap], *, packet: dict[str, Any]) -> list[ResearchPlanStep]:
    if _issuer_classification_from_packet(packet) == ISSUER_CLASS_FINANCIAL:
        templates = [
            (
                "Pull latest and prior two 10-Q/10-K filings and extract deposit, loan, and funding-mix bridges.",
                ["Management's Discussion and Analysis", "Balance Sheet", "Liquidity and Capital Resources"],
                ["deposits", "loans", "funding", "liquidity"],
                "If deposits weaken, loan growth deteriorates, or funding mix shifts unfavorably, weaken the balance-sheet thesis.",
                ["deposits", "loans", "total_assets"],
            ),
            (
                "Hydrate loans, allowance, provision, charge-off, and nonaccrual-credit history across recent periods.",
                ["Allowance for Credit Losses", "Loan Portfolio", "Credit Quality Indicators"],
                ["allowance", "charge-off", "nonaccrual", "provision", "loan"],
                "If reserve coverage weakens across periods, charge-offs rise, or nonaccruals deteriorate, reduce conviction in balance-sheet resilience.",
                ["loans", "allowance_for_credit_losses", "provision_for_credit_losses", "net_charge_offs", "nonaccrual_loans"],
            ),
            (
                "Parse debt exhibits for covenant definitions, maturities, refinancing paths, and liquidity buffers.",
                ["Debt Notes", "Exhibits", "Liquidity and Capital Resources"],
                ["covenant", "maturity", "refinancing", "liquidity"],
                "If covenant headroom or liquidity buffers look narrow, reject the benign funding path.",
                ["net_debt", "liquidity_stress_score"],
            ),
            (
                "Re-parse segment and income statement disclosures for segment earnings power and profit durability.",
                ["Segment Information", "Results of Operations"],
                ["segment", "pre-tax income", "net interest income", "fee income"],
                "If core segment earnings power fades without an offset, downgrade the earnings-quality thesis.",
                ["revenue", "operating_income"],
            ),
            (
                "Extract cash-flow bridge and reconciliation items from recent filings and exhibits.",
                ["Cash Flows", "Liquidity and Capital Resources", "Non-GAAP Reconciliation"],
                ["operating cash flow", "reconciliation", "non-cash", "working capital"],
                "If negative cash flow persists after removing one-off or non-cash items, reject the normalization thesis.",
                ["cfo", "fcf"],
            ),
            (
                "Extract shares outstanding trend and capital-return mechanics across recent quarters.",
                ["Stockholders' Equity Footnote", "Cover Page", "Exhibits"],
                ["shares outstanding", "share repurchase", "dividend", "stock-based compensation"],
                "If share count stops shrinking or buybacks rely on weaker liquidity, reduce per-share upside.",
                ["shares_outstanding", "share_repurchases_amount"],
            ),
            (
                "Search 8-K exhibits and investor presentations for management bridges, covenant commentary, and capital-return guidance.",
                ["Item 2.02 / 7.01 disclosures", "99.x exhibits"],
                ["investor presentation", "reconciliation", "liquidity", "capital return"],
                "If management commentary conflicts with filing-based interpretation, prioritize the disconfirming evidence.",
                ["liquidity_stress_score", "share_repurchases_amount"],
            ),
        ]
    else:
        templates = [
        (
            "Pull latest and prior two 10-Q/10-K filings and extract revenue bridge deltas.",
            ["Management's Discussion and Analysis", "Revenue Recognition Footnote"],
            ["revenue", "pricing", "demand", "volume"],
            "If revenue bridge weakens with no temporary explanation, reject growth durability.",
            ["revenue", "revenue_growth"],
        ),
        (
            "Re-parse segment and income statement disclosures for margin durability checks.",
            ["Segment Information", "Results of Operations"],
            ["gross margin", "operating margin", "segment"],
            "If core segment margins contract without offset, downgrade thesis.",
            ["gross_margin", "operating_margin"],
        ),
        (
            "Extract cash flow bridge and working-capital adjustments from recent filings.",
            ["Cash Flows", "Liquidity and Capital Resources"],
            ["operating cash flow", "working capital", "accounts receivable", "inventory"],
            "If CFO depends on one-off working capital reversals, invalidate cash-flow quality assumption.",
            ["cfo", "fcf"],
        ),
        (
            "Parse debt exhibits for covenant definitions, maturities, and refinancing paths.",
            ["Debt Notes", "Exhibits"],
            ["covenant", "minimum liquidity", "maturity", "refinancing"],
            "If covenant headroom appears narrow, reject benign balance-sheet path.",
            ["net_debt", "liquidity_stress_score"],
        ),
        (
            "Extract shares outstanding trend and dilution instruments across 4 quarters.",
            ["Cover Page", "Stockholders' Equity Footnote"],
            ["shares outstanding", "stock-based compensation", "ATM", "convertible"],
            "If share count accelerates faster than FCF growth, downgrade per-share upside.",
            ["sbc_proxy_flag", "fcf"],
        ),
        (
            "Parse customer concentration and non-GAAP reconciliation language for quality drift.",
            ["Customer Concentration", "Non-GAAP Reconciliation"],
            ["major customer", "adjusted EBITDA", "reconciliation"],
            "If concentration rises materially or adjustments expand, lower confidence.",
            ["customer_concentration_signal", "non_gaap_reconciliation_signal"],
        ),
        (
            "Validate IR/press chronology against filing claims and note any contradictions.",
            ["Item 8.01 / 2.02 disclosures", "Risk Factors"],
            ["guidance", "outlook", "capital allocation"],
            "If IR disclosures conflict with filing narrative, prioritize disconfirming interpretation.",
            ["valuation_gap", "liquidity_stress_score"],
        ),
        ]

    primary_gap = gaps[0].summary if gaps else "No material evidence gap flagged"
    actions: list[ResearchPlanStep] = []
    for idx, question in enumerate(key_questions):
        action, sections, keywords, disconfirm, metrics = templates[idx]
        actions.append(
            ResearchPlanStep(
                step_id=f"A{idx+1}",
                question_id=question.question_id,
                action=action,
                section_targets=sections,
                keywords=keywords,
                disconfirmation_check=disconfirm,
                evidence_gap=primary_gap,
                tied_metrics=metrics,
                allowed_source="EDGAR",
            )
        )

    # Append up to 2 explicit metadata remediation actions derived from evidence gaps.
    for gap in gaps[:2]:
        if "ir_rss_url" not in gap.recommended_action.lower():
            continue
        actions.append(
            ResearchPlanStep(
                step_id=f"A{len(actions)+1}",
                question_id=key_questions[0].question_id,
                action=gap.recommended_action,
                section_targets=["Universe Metadata"],
                keywords=["ir_rss_url", "allowlist"],
                disconfirmation_check="If metadata remains missing, external IR evidence stays unavailable.",
                evidence_gap=gap.summary,
                tied_metrics=["research_coverage"],
                allowed_source="EDGAR",
            )
        )
    return actions


def _build_claims(packet: dict[str, Any], evidence_items: list[EvidenceItem]) -> list[ClaimTrace]:
    claims: list[ClaimTrace] = []
    citations = _citations(_pick_supporting(evidence_items, n=2), limit=2)

    liquidity = packet.get("fundamentals", {}).get("liquidity_stress_score")
    if isinstance(liquidity, (int, float)):
        claims.append(
            ClaimTrace(
                claim_id="claim_liquidity_stress",
                label="liquidity_stress_score",
                value=float(liquidity),
                unit="score",
                citations=citations,
                derived_from=["fundamentals.liquidity_stress_score"],
            )
        )

    dcf_base = _dcf_base(packet)
    if isinstance(dcf_base, (int, float)):
        claims.append(
            ClaimTrace(
                claim_id="claim_dcf_base",
                label="dcf_per_share_base",
                value=float(dcf_base),
                unit="USD/share",
                citations=[],
                derived_from=["valuations.dcf.outputs.base"],
            )
        )

    secondary_label, secondary_base, secondary_ref = _secondary_valuation_base(packet)
    if isinstance(secondary_base, (int, float)):
        claims.append(
            ClaimTrace(
                claim_id=f"claim_{secondary_label.lower()}_base",
                label=f"{secondary_label.lower()}_per_share_base",
                value=float(secondary_base),
                unit="USD/share",
                citations=[],
                derived_from=[secondary_ref],
            )
        )
    return claims


def _dedupe_items(items: list[EvidenceItem]) -> list[EvidenceItem]:
    dedup: dict[str, EvidenceItem] = {}
    for item in items:
        dedupe_key = item.dedupe_key or sha256_text(
            f"{item.source_url}|{(item.source_published_at or '')[:10]}|{sha256_text(item.source_title or '')}"
        )
        key = f"{item.ticker}|{dedupe_key}"
        if key not in dedup:
            dedup[key] = item
    return list(dedup.values())


def _persist_evidence_items(conn, run_id: str, items: list[EvidenceItem]) -> None:
    from app.db import ensure_evidence_item_runs

    if items:
        ensure_evidence_item_runs(conn)
    for item in items:
        excerpt_hash = sha256_text(item.excerpt_text)
        content_hash = item.content_hash or excerpt_hash
        dedupe_key = item.dedupe_key or sha256_text(
            f"{item.source_url}|{(item.source_published_at or '')[:10]}|{sha256_text(item.source_title or '')}"
        )
        adapter_run_id = item.adapter_run_id or run_id
        conn.execute(
            """
            INSERT INTO evidence_items(
                evidence_id, ticker, as_of_date, run_id, adapter_run_id, source_type, source_url, source_title,
                source_published_at, retrieved_at, excerpt_text, excerpt_hash, content_hash, dedupe_key,
                citations_json, derived_from_json, item_hash, created_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(evidence_id) DO UPDATE SET
                -- as_of_date travels with run_id. Updating one without the
                -- other left the row carrying the first run's date and the
                -- second run's id, so it belonged to neither: the gating count
                -- for (ticker, as_of_date, run_id) was 0 for both runs.
                as_of_date=excluded.as_of_date,
                run_id=excluded.run_id,
                adapter_run_id=excluded.adapter_run_id,
                source_type=excluded.source_type,
                source_url=excluded.source_url,
                source_title=excluded.source_title,
                source_published_at=excluded.source_published_at,
                retrieved_at=excluded.retrieved_at,
                excerpt_text=excluded.excerpt_text,
                excerpt_hash=excluded.excerpt_hash,
                content_hash=excluded.content_hash,
                dedupe_key=excluded.dedupe_key,
                citations_json=excluded.citations_json,
                item_hash=excluded.item_hash
            """,
            (
                item.id,
                item.ticker,
                item.as_of_date,
                run_id,
                adapter_run_id,
                item.source_type,
                item.source_url,
                item.source_title,
                item.source_published_at,
                item.retrieved_at,
                item.excerpt_text,
                excerpt_hash,
                content_hash,
                dedupe_key,
                json.dumps([c.model_dump() for c in item.citations]),
                json.dumps([]),
                item.hash,
                utc_now_iso(),
            ),
        )
        conn.execute(
            """
            INSERT OR IGNORE INTO evidence_item_runs(
                evidence_id, run_id, ticker, as_of_date, created_at
            ) VALUES(?, ?, ?, ?, ?)
            """,
            (item.id, run_id, item.ticker, item.as_of_date, utc_now_iso()),
        )


def _adapter_tokens(adapter: ResearchAdapter) -> set[str]:
    name = adapter.__class__.__name__.lower()
    if "edgar" in name:
        return {"edgar"}
    if "sec" in name and "exhibit" in name:
        return {"exhibits", "earnings"}
    if "companynews" in name:
        return {"news", "homepage"}
    if "irpress" in name:
        return {"news", "ir_press"}
    return {adapter.source_type.lower()}


def _select_adapters(cfg, source_filters: set[str] | None) -> list[ResearchAdapter]:
    adapters = build_adapter_chain(cfg)
    if not source_filters:
        return adapters
    normalized = {token.strip().lower() for token in source_filters if token.strip()}
    selected: list[ResearchAdapter] = []
    for adapter in adapters:
        tokens = _adapter_tokens(adapter)
        if tokens.intersection(normalized):
            selected.append(adapter)
    return selected


def run_research_agent_for_ticker(
    ticker: str,
    as_of_date: str | None = None,
    run_id: str | None = None,
    source_filters: set[str] | None = None,
) -> Path | None:
    cfg = get_config()
    ticker = ticker.upper()
    run_id = run_id or f"research_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    with get_db() as conn:
        row = _latest_packet_row(conn, ticker, as_of_date=as_of_date)
        if not row:
            return None
        packet_path = Path(row["packet_path"])
        # The run is filed under the date it was ASKED for, not the date of the
        # evidence packet it found. Adopting the packet's date filed a
        # 2024-06-30 run under the packet's 2024-03-31, so the date the caller
        # asked about ended up with no research packet at all. With no
        # requested date (a run for today) the packet's own date still names
        # the run, which is the behaviour every live caller has.
        as_of_date = as_of_date or row["as_of_date"]
        metadata = _company_metadata(conn, ticker)

    if not packet_path.exists():
        return None

    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    context = AdapterContext(
        ticker=ticker,
        as_of_date=as_of_date,
        company_name=metadata.get("name") or None,
        packet=packet,
        run_id=run_id,
        ir_rss_url=metadata.get("ir_rss_url") or None,
        homepage_url=metadata.get("homepage_url") or None,
    )

    evidence_items: list[EvidenceItem] = []
    evidence_gaps: list[EvidenceGap] = []
    for adapter in _select_adapters(cfg, source_filters):
        try:
            result = adapter.collect(context)
            evidence_items.extend(result.evidence_items)
            evidence_gaps.extend(result.evidence_gaps)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "research_adapter_error",
                extra={"stage_name": "research", "stage_ticker": ticker, "stage_error": str(exc)},
            )
            evidence_gaps.append(
                EvidenceGap(
                    gap_id=f"GAP_ADAPTER_{adapter.__class__.__name__.upper()}",
                    severity="medium",
                    summary=f"Adapter failure: {adapter.__class__.__name__}",
                    source_type=adapter.source_type,
                    recommended_action="Inspect adapter logs and rerun research.",
                )
            )

    evidence_items = _dedupe_items(evidence_items)
    evidence_gaps.extend(_detect_base_gaps(packet))

    if not evidence_items:
        logger.info("research_skipped_no_evidence", extra={"stage_name": "research", "stage_ticker": ticker})
        return None

    key_questions = _build_key_questions()
    findings = _build_findings(packet, evidence_items)
    risks = _build_risks(evidence_items, evidence_gaps)
    catalysts = _build_catalysts(packet, evidence_items)
    finding_ids: set[str] = set()
    for entry in findings:
        finding_ids.update(entry.evidence_item_ids)
    disconfirming = _build_disconfirming(evidence_items, exclude_ids=finding_ids)
    _assert_opposing_buckets_disjoint(
        findings, disconfirming, evidence_pool_size=len(evidence_items)
    )
    next_actions = _build_next_actions(key_questions, evidence_gaps, packet=packet)
    claims = _build_claims(packet, evidence_items)
    signals_model = compute_research_signals(
        ticker=ticker,
        as_of_date=as_of_date,
        run_id=run_id,
        evidence_items=evidence_items,
    )

    packet_model = ResearchPacket(
        run_id=run_id,
        ticker=ticker,
        as_of_date=as_of_date,
        generated_at=utc_now_iso(),
        key_questions=key_questions,
        evidence_items=evidence_items,
        findings=findings,
        risks=risks,
        catalysts=catalysts,
        disconfirming_evidence=disconfirming,
        next_actions=next_actions,
        evidence_gaps=evidence_gaps,
        claims=claims,
        signals=signals_model.model_dump(mode="json"),
    )
    packet_model.quality = score_research_packet(packet_model, cfg)

    payload = packet_model.model_dump(mode="json")
    validate_research_packet(payload)

    cfg.research_dir.mkdir(parents=True, exist_ok=True)
    output_path = cfg.research_dir / f"{ticker}_{run_id}.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    packet_hash = sha256_file(output_path)

    with get_db() as conn:
        _persist_evidence_items(conn, run_id, evidence_items)
        persist_research_signals(conn, signals_model)
        conn.execute(
            """
            INSERT INTO research_packets(ticker, as_of_date, run_id, packet_path, packet_hash, created_at)
            VALUES(?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker, as_of_date, run_id) DO UPDATE SET
                packet_path=excluded.packet_path,
                packet_hash=excluded.packet_hash
            """,
            (ticker, as_of_date, run_id, str(output_path), packet_hash, utc_now_iso()),
        )

    logger.info("research_completed", extra={"stage_name": "research", "stage_ticker": ticker})
    return output_path


def _active_universe_tickers(conn) -> list[str]:
    state = get_state(conn, "active_universe") or {}
    universe_id = state.get("universe_id")
    if universe_id:
        rows = conn.execute(
            "SELECT ticker FROM universe_members WHERE universe_id = ? ORDER BY id ASC",
            (universe_id,),
        ).fetchall()
        if rows:
            return [row["ticker"] for row in rows]
    rows = conn.execute("SELECT ticker FROM companies ORDER BY ticker").fetchall()
    return [row["ticker"] for row in rows]


def _top_ranked_tickers(conn, top_n: int, as_of_date: str | None) -> list[str]:
    # Each ticker is ranked once, on its latest score at or before the as-of date.
    # Ranking every historical row let a name's stale high score outrank names with
    # better current scores, and returned the same ticker once per row.
    params: list[Any] = []
    bound = ""
    if as_of_date:
        bound = " AND s2.as_of_date <= ?"
        params.append(as_of_date)
    params.append(top_n)
    rows = conn.execute(
        f"""
        SELECT s.ticker
        FROM scores s
        WHERE s.id = (
            SELECT s2.id
            FROM scores s2
            WHERE s2.ticker = s.ticker{bound}
            ORDER BY s2.as_of_date DESC, s2.created_at DESC, s2.id DESC
            LIMIT 1
        )
        ORDER BY s.total_score DESC, s.ticker ASC
        LIMIT ?
        """,
        tuple(params),
    ).fetchall()
    return [row["ticker"] for row in rows]


def run_research_for_scope(
    as_of_date: str,
    *,
    top_n: int,
    use_active_universe: bool,
    run_id: str | None = None,
    tickers: list[str] | None = None,
    limit: int | None = None,
    source_filters: set[str] | None = None,
) -> int:
    run_id = run_id or f"research_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    tickers = [t.upper() for t in (tickers or []) if t.strip()]

    with get_db() as conn:
        if tickers:
            scope = tickers
        elif use_active_universe:
            scope = _active_universe_tickers(conn)
        else:
            scope = _top_ranked_tickers(conn, top_n=top_n, as_of_date=as_of_date)
            if not scope:
                scope = _active_universe_tickers(conn)

    if limit is not None and limit > 0:
        scope = scope[:limit]

    built = 0
    for ticker in scope:
        path = run_research_agent_for_ticker(ticker, as_of_date=as_of_date, run_id=run_id, source_filters=source_filters)
        if path:
            built += 1
    return built


def _tickers_from_recent_gaps(run_id: str | None = None) -> set[str]:
    cfg = get_config()
    tickers: set[str] = set()
    if not cfg.gaps_dir.exists():
        return tickers
    for path in cfg.gaps_dir.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if run_id and payload.get("run_id") != run_id:
            continue
        research_warnings = payload.get("research_warnings") or []
        missing_sources = payload.get("missing_research_sources") or []
        if research_warnings or missing_sources:
            ticker = str(payload.get("ticker") or "").upper()
            if ticker:
                tickers.add(ticker)
    return tickers


def _tickers_with_stale_signals(conn, as_of_date: str, run_id: str | None = None) -> set[str]:
    params: list[Any] = [as_of_date]
    sql = """
        SELECT ticker
        FROM research_signals
        WHERE as_of_date = ?
          AND (recency_days_min IS NULL OR recency_days_min > 180)
    """
    if run_id:
        sql += " AND run_id = ?"
        params.append(run_id)
    rows = conn.execute(sql, tuple(params)).fetchall()
    stale = {str(row["ticker"]).upper() for row in rows}

    if run_id:
        missing_rows = conn.execute(
            """
            SELECT c.ticker
            FROM companies c
            LEFT JOIN research_signals rs
              ON rs.ticker = c.ticker AND rs.as_of_date = ? AND rs.run_id = ?
            WHERE rs.id IS NULL
            """,
            (as_of_date, run_id),
        ).fetchall()
    else:
        missing_rows = conn.execute(
            """
            SELECT c.ticker
            FROM companies c
            LEFT JOIN research_signals rs
              ON rs.ticker = c.ticker AND rs.as_of_date = ?
            WHERE rs.id IS NULL
            """,
            (as_of_date,),
        ).fetchall()
    stale.update(str(row["ticker"]).upper() for row in missing_rows)
    return stale


def run_research_gap_closer(
    *,
    as_of_date: str,
    run_id: str | None = None,
    limit: int | None = None,
    source_filters: set[str] | None = None,
    tickers: list[str] | None = None,
) -> dict[str, Any]:
    target_run_id = run_id or f"research_gapclose_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    with get_db() as conn:
        universe_rows = conn.execute("SELECT ticker FROM companies ORDER BY ticker").fetchall()
        universe = [str(row["ticker"]).upper() for row in universe_rows]
        stale = _tickers_with_stale_signals(conn, as_of_date, run_id=run_id)
    if tickers:
        requested = {str(t).upper() for t in tickers if str(t).strip()}
        scope = sorted(requested & set(universe))
    else:
        from_gaps = _tickers_from_recent_gaps(run_id=run_id)
        scope = sorted((stale.union(from_gaps)) & set(universe))
        if not scope:
            scope = universe[:]
    if limit is not None and limit > 0:
        scope = scope[:limit]

    processed: list[str] = []
    built = 0
    for ticker in scope:
        packet = run_research_agent_for_ticker(
            ticker,
            as_of_date=as_of_date,
            run_id=target_run_id,
            source_filters=source_filters,
        )
        if not packet:
            continue
        built += 1
        processed.append(ticker)
        score_ticker(ticker, run_id=target_run_id)

    return {
        "run_id": target_run_id,
        "as_of_date": as_of_date,
        "processed_tickers": processed,
        "processed_count": len(processed),
        "research_packets_built": built,
    }


def latest_research_packet_path(conn, ticker: str, as_of_date: str | None = None) -> Path | None:
    if as_of_date:
        row = conn.execute(
            """
            SELECT packet_path
            FROM research_packets
            WHERE ticker = ? AND as_of_date = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (ticker, as_of_date),
        ).fetchone()
    else:
        row = conn.execute(
            """
            SELECT packet_path
            FROM research_packets
            WHERE ticker = ?
            ORDER BY as_of_date DESC, created_at DESC
            LIMIT 1
            """,
            (ticker,),
        ).fetchone()
    if not row:
        return None
    path = Path(row["packet_path"])
    if not path.exists():
        return None
    return path
