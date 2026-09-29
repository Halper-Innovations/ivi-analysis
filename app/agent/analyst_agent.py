from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from app.config import get_config
from app.db import get_db, utc_now_iso
from app.fundamentals.normalize import UNKNOWN
from app.logging import get_logger
from app.util.hashing import sha256_file


logger = get_logger(__name__)


class CitationRef(BaseModel):
    source_url: str
    snippet: str
    section_label: str | None = None


class Hypothesis(BaseModel):
    hypothesis_id: str
    direction: Literal["LONG", "SHORT", "BALANCE_SHEET"]
    claim: str
    valuation_anchor: str
    confidence: Literal["LOW", "MEDIUM", "HIGH"]
    citations: list[CitationRef]


class Assumption(BaseModel):
    assumption: str
    status: Literal["VERIFIED", "UNKNOWN", "CONTRADICTED"]
    critical: bool
    support: str


class ResearchPlanStep(BaseModel):
    step_id: str
    action: str
    rationale: str
    expected_artifact: str
    section_targets: list[str]
    keywords: list[str]
    disconfirmation_check: str
    metric_ties: list[str]
    allowed_source: Literal["EDGAR"] = "EDGAR"


class RedTeamFinding(BaseModel):
    thesis_under_attack: str
    strongest_counterargument: str
    most_damaging_next_datapoint: str
    citations: list[CitationRef]


class MemoOutline(BaseModel):
    title: str
    sections: list[str]
    key_claims_with_citations: list[dict[str, Any]]


class DecisionProposal(BaseModel):
    classification: Literal["LONG", "SHORT", "WATCHLIST", "ABSTAIN", "RESEARCH_ONLY"]
    horizon: str
    catalyst_window: str
    valuation_gap_summary: str
    falsifiers: list[str]
    monitoring_triggers: list[str]


class AnalystBundle(BaseModel):
    hypotheses: list[Hypothesis]
    assumptions: list[Assumption]
    research_plan: list[ResearchPlanStep]
    red_team: RedTeamFinding
    memo_outline: MemoOutline
    decision: DecisionProposal


def _latest_packet_row(conn, ticker: str) -> Any | None:
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


def _citations_from_packet(packet: dict[str, Any], limit: int = 5) -> list[CitationRef]:
    out: list[CitationRef] = []
    for item in packet.get("extracted_facts", []):
        citation = item.get("citation", {})
        if not citation.get("source_url"):
            continue
        out.append(
            CitationRef(
                source_url=citation["source_url"],
                snippet=(citation.get("snippet") or "")[:600],
                section_label=citation.get("section_label"),
            )
        )
        if len(out) >= limit:
            break
    if out:
        return out
    for item in packet.get("financials", []):
        citation = item.get("citation", {})
        if not citation.get("source_url"):
            continue
        out.append(
            CitationRef(
                source_url=citation["source_url"],
                snippet=(citation.get("snippet") or "")[:600],
                section_label=citation.get("section_label"),
            )
        )
        if len(out) >= limit:
            break
    return out


def _build_hypotheses(packet: dict[str, Any]) -> list[Hypothesis]:
    valuations = packet.get("valuations", {})
    fundamentals = packet.get("fundamentals", {})
    citations = _citations_from_packet(packet, limit=6)
    shared = citations[:2] if citations else []

    dcf = valuations.get("dcf", {}) if isinstance(valuations, dict) else {}
    dcf_outputs = dcf.get("outputs", {}) if isinstance(dcf, dict) else {}
    if not isinstance(dcf_outputs, dict):
        dcf_outputs = dcf if isinstance(dcf, dict) else {}
    reverse_dcf = valuations.get("reverse_dcf", {}).get("outputs", {}) if isinstance(valuations, dict) else {}
    epv = valuations.get("epv", {}) if isinstance(valuations, dict) else {}
    epv_outputs = epv.get("outputs", {}) if isinstance(epv, dict) else {}
    if not isinstance(epv_outputs, dict):
        epv_outputs = epv if isinstance(epv, dict) else {}

    dcf_range = {key: dcf_outputs.get(key) for key in ("low", "base", "high") if key in dcf_outputs}
    if not dcf_range:
        legacy_dcf = valuations.get("dcf_lite", {}).get("outputs", {}) if isinstance(valuations, dict) else {}
        if isinstance(legacy_dcf, dict):
            dcf_range = legacy_dcf.get("per_share_range", {})
    epv_anchor = epv_outputs.get("value_per_share", UNKNOWN)
    implied_growth = reverse_dcf.get("implied_growth", UNKNOWN)
    if reverse_dcf.get("implied_growth_saturated"):
        implied_growth = "SATURATED_BOUND_NOT_A_SOLVE"

    hyps: list[Hypothesis] = []
    hyps.append(
        Hypothesis(
            hypothesis_id="H1",
            direction="LONG",
            claim="Intrinsic value range may be understated by current narrative, contingent on filing-reported operating stability.",
            valuation_anchor=f"DCF per-share range: {dcf_range if dcf_range else UNKNOWN}",
            confidence="LOW",
            citations=shared,
        )
    )

    hyps.append(
        Hypothesis(
            hypothesis_id="H2",
            direction="SHORT",
            claim="If reverse DCF implied growth is outside filing history feasibility, downside case strengthens even if zero-growth value looks acceptable.",
            valuation_anchor=f"Reverse DCF implied growth: {implied_growth}; EPV/share: {epv_anchor}",
            confidence="LOW",
            citations=citations[2:4] if len(citations) > 2 else shared,
        )
    )

    liq = fundamentals.get("liquidity_stress_score", UNKNOWN)
    hyps.append(
        Hypothesis(
            hypothesis_id="H3",
            direction="BALANCE_SHEET",
            claim="Balance sheet and liquidity path may dominate equity outcome over operating variance.",
            valuation_anchor=f"Liquidity stress score: {liq}",
            confidence="MEDIUM",
            citations=citations[4:6] if len(citations) > 4 else shared,
        )
    )
    return hyps


def _build_assumptions(packet: dict[str, Any], hypotheses: list[Hypothesis]) -> list[Assumption]:
    _ = hypotheses
    fundamentals = packet.get("fundamentals", {})
    valuations = packet.get("valuations", {})

    reverse_price = valuations.get("reverse_dcf", {}).get("inputs", {}).get("market_price", UNKNOWN)
    assumptions: list[Assumption] = [
        Assumption(
            assumption="Market price is available for margin-of-safety validation.",
            status="VERIFIED" if reverse_price != UNKNOWN else "UNKNOWN",
            critical=True,
            support=f"reverse_dcf.inputs.market_price={reverse_price}",
        ),
        Assumption(
            assumption="Free cash flow remains non-negative over the next filing cycle.",
            status="VERIFIED" if isinstance(fundamentals.get("fcf"), (int, float)) and fundamentals.get("fcf") >= 0 else "UNKNOWN",
            critical=True,
            support=f"fundamentals.fcf={fundamentals.get('fcf', UNKNOWN)}",
        ),
        Assumption(
            assumption="Refinancing and covenant risk does not materially worsen.",
            status="UNKNOWN",
            critical=True,
            support="Requires latest credit agreement exhibit extraction",
        ),
    ]
    return assumptions


def _build_research_plan(packet: dict[str, Any]) -> list[ResearchPlanStep]:
    ticker = packet.get("ticker", "UNKNOWN")
    fundamentals = packet.get("fundamentals", {})
    fcf_value = fundamentals.get("fcf", UNKNOWN)
    debt_value = fundamentals.get("net_debt", UNKNOWN)
    return [
        ResearchPlanStep(
            step_id="R1",
            action=f"Pull recent 10-Q/10-K filings for {ticker} from EDGAR submissions and archives.",
            rationale="Validate trend persistence and identify revisions in reported fundamentals.",
            expected_artifact="filings metadata + downloaded primary docs",
            section_targets=["Management's Discussion and Analysis", "Consolidated Statements of Cash Flows"],
            keywords=["revenue", "operating income", "cash provided by operating activities"],
            disconfirmation_check=f"Reject long-leaning thesis if CFO trend weakens versus packet FCF={fcf_value}.",
            metric_ties=["revenue", "operating_margin", "cfo", "fcf"],
        ),
        ResearchPlanStep(
            step_id="R2",
            action="Parse credit agreement and debt-related exhibits for covenant definitions and maturity ladders.",
            rationale="Refine liquidity and dilution pathway risk.",
            expected_artifact="extracted_facts entries with debt/liquidity citations",
            section_targets=["Notes to Financial Statements", "Debt exhibits (EX-10)", "Liquidity and Capital Resources"],
            keywords=["covenant", "minimum liquidity", "maturity", "refinancing", "waiver"],
            disconfirmation_check=f"Reject benign balance-sheet thesis if covenant headroom contradicts net_debt={debt_value}.",
            metric_ties=["net_debt", "liquidity_stress_score"],
        ),
        ResearchPlanStep(
            step_id="R3",
            action="Extract shares outstanding history from filing cover pages across recent reporting periods.",
            rationale="Confirm dilution trajectory for per-share intrinsic valuation.",
            expected_artifact="time series of shares_outstanding facts",
            section_targets=["Cover Page", "Equity footnotes", "Statement of Stockholders' Equity"],
            keywords=["shares outstanding", "share-based compensation", "at-the-market", "convertible"],
            disconfirmation_check="Invalidate valuation-per-share stability if shares increase materially without operating improvement.",
            metric_ties=["sbc_proxy_flag", "fcf_per_share_proxy"],
        ),
    ]


def _build_red_team(packet: dict[str, Any], hypotheses: list[Hypothesis]) -> RedTeamFinding:
    citations = _citations_from_packet(packet, limit=2)
    target = hypotheses[0].claim if hypotheses else "Primary valuation thesis"
    return RedTeamFinding(
        thesis_under_attack=target,
        strongest_counterargument="Valuation ranges are low-confidence because key drivers are UNKNOWN or weakly evidenced in the latest packet.",
        most_damaging_next_datapoint="A filing-backed decline in operating cash flow with concurrent dilution increase in the next 10-Q.",
        citations=citations,
    )


def _build_memo_outline(packet: dict[str, Any], hypotheses: list[Hypothesis]) -> MemoOutline:
    claims = []
    for h in hypotheses:
        claims.append(
            {
                "claim": h.claim,
                "citations": [c.model_dump() for c in h.citations],
            }
        )
    return MemoOutline(
        title=f"{packet.get('ticker', 'UNKNOWN')} valuation-led memo outline",
        sections=[
            "Business Snapshot",
            "Fundamentals",
            "Valuation Gate",
            "Hypothesis and What Must Be True",
            "Red Team",
            "Monitoring Plan",
        ],
        key_claims_with_citations=claims,
    )


def _build_decision(packet: dict[str, Any], assumptions: list[Assumption]) -> DecisionProposal:
    unknown_critical = len([a for a in assumptions if a.critical and a.status == "UNKNOWN"])
    valuations = packet.get("valuations", {})
    reverse_price = valuations.get("reverse_dcf", {}).get("inputs", {}).get("market_price", UNKNOWN)

    if unknown_critical >= 2:
        classification = "RESEARCH_ONLY"
    elif reverse_price == UNKNOWN:
        classification = "WATCHLIST"
    else:
        classification = "ABSTAIN"

    return DecisionProposal(
        classification=classification,
        horizon="weeks-to-months",
        catalyst_window="NONE",
        valuation_gap_summary="Derived solely from packet valuation outputs; no external facts used.",
        falsifiers=[
            "Material negative revision in next filing cash flow metrics",
            "New debt or dilution instrument disclosed in EDGAR exhibits",
        ],
        monitoring_triggers=[
            "New 10-Q/10-K filing posted",
            "Going concern or covenant language appears",
        ],
    )


def _output_dir(ticker: str, as_of_date: str) -> Path:
    cfg = get_config()
    path = cfg.analyst_outputs_dir / f"{ticker}_{as_of_date}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_output_file(ticker: str, as_of_date: str, output_type: str, payload: dict[str, Any]) -> Path:
    out_dir = _output_dir(ticker, as_of_date)
    path = out_dir / f"{output_type}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    output_hash = sha256_file(path)
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO analyst_outputs(ticker, as_of_date, output_type, output_path, output_hash, created_at)
            VALUES(?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker, as_of_date, output_type) DO UPDATE SET
                output_path=excluded.output_path,
                output_hash=excluded.output_hash
            """,
            (ticker, as_of_date, output_type, str(path), output_hash, utc_now_iso()),
        )
    return path


def run_analyst_agent_for_ticker(ticker: str) -> AnalystBundle | None:
    with get_db() as conn:
        row = _latest_packet_row(conn, ticker)
        if not row:
            return None
        as_of_date = row["as_of_date"]
        packet_path = Path(row["packet_path"])

    if not packet_path.exists():
        return None

    packet = json.loads(packet_path.read_text(encoding="utf-8"))

    hypotheses = _build_hypotheses(packet)
    assumptions = _build_assumptions(packet, hypotheses)
    research_plan = _build_research_plan(packet)
    red_team = _build_red_team(packet, hypotheses)
    memo_outline = _build_memo_outline(packet, hypotheses)
    decision = _build_decision(packet, assumptions)

    bundle = AnalystBundle(
        hypotheses=hypotheses,
        assumptions=assumptions,
        research_plan=research_plan,
        red_team=red_team,
        memo_outline=memo_outline,
        decision=decision,
    )

    payload = bundle.model_dump()
    _write_output_file(ticker, as_of_date, "hypotheses", {"hypotheses": payload["hypotheses"]})
    _write_output_file(ticker, as_of_date, "research_plan", {"research_plan": payload["research_plan"]})
    _write_output_file(ticker, as_of_date, "red_team", {"red_team": payload["red_team"]})
    _write_output_file(ticker, as_of_date, "memo_outline", {"memo_outline": payload["memo_outline"]})
    _write_output_file(ticker, as_of_date, "decision", {"decision": payload["decision"], "assumptions": payload["assumptions"]})

    logger.info("analyst_completed", extra={"stage_name": "analyst", "stage_ticker": ticker})
    return bundle


def run_red_team_for_ticker(ticker: str) -> dict[str, Any] | None:
    # Red-team is generated inside the analyst bundle in v0 for consistency.
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT output_path
            FROM analyst_outputs
            WHERE ticker = ? AND output_type = 'red_team'
            ORDER BY as_of_date DESC
            LIMIT 1
            """,
            (ticker,),
        ).fetchone()
    if not row:
        return None
    path = Path(row["output_path"])
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
