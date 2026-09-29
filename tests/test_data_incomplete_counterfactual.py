"""Counterfactual + acceptance regression for the DATA_INCOMPLETE tier.

This module replays a *cached-artifact-shaped* slice of the actionability
investigation directly through
``_selection_audit_for_ticker`` and counts the names whose selection-audit
status flips from a data-only BLOCKED/WATCHLIST_ONLY downgrade into the new
DATA_INCOMPLETE (resolve-then-promote) tier.

It is deliberately self-contained: it models the relevant packet/scenario
fields inline (no dependency on any ``data/`` artifact, which is gitignored and
uncommittable) so it is a deterministic, committable regression. No live LLM,
sector run, or network is touched.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.autonomous.run_contract import EvidenceReference, ToolCallRecord
from app.autonomous.sector_contract import (
    SectorCompanyFinancialPacket,
    SectorExpectedReturnScenario,
)
from app.autonomous.sector_runtime import _selection_audit_for_ticker


# --- Cached-artifact-shaped fixture rows (modeled from the investigation). ---


@dataclass(frozen=True)
class _Row:
    """One modeled audit row: identity + the inputs that drive its status."""

    ticker: str
    sector: str
    old_status: str  # status before the DATA_INCOMPLETE tier existed
    base_return: float
    downside_return: float
    data_quality_status: str = "OK"
    filing_risk_evidence_status: str | None = None
    packet_caps: tuple[str, ...] = ()
    solvency_risk: str | None = None
    impairment_class_primary: str | None = None


# The DTB row is the canonical data-only downgrade: a +57.8% base return that
# clears the 12% hurdle, blocked before the tier only by NO_FILING-class hard blockers
# (NO_FILING / NO_READABLE_ANNUAL_FILING / FILING_RISK_NO_FILING) plus a
# (non-binding) ELEVATED solvency confidence cap. With the tier it must move OUT of
# WATCHLIST_ONLY into DATA_INCOMPLETE.
_DTB = _Row(
    ticker="DTB",
    sector="utilities",
    old_status="WATCHLIST_ONLY",
    base_return=0.578,
    downside_return=0.117,
    data_quality_status="NO_FILING",
    filing_risk_evidence_status="NO_READABLE_ANNUAL_FILING",
    packet_caps=("FILING_RISK_NO_FILING",),
    solvency_risk="ELEVATED",
)

# NO_FILING-class names that ALSO miss the 12% hurdle (BASE_RETURN_BELOW_12PCT
# is a binding HARD blocker): they stay BLOCKED — the hurdle dominates the data
# gap, so they are NOT data-only flips.
_BELOW_HURDLE_NO_FILING = (
    _Row("CHDN", "media_entertainment", "WATCHLIST_ONLY", 0.089, -0.133, "NO_FILING", "NO_READABLE_ANNUAL_FILING", ("FILING_RISK_NO_FILING",)),
    _Row("BCO", "transportation_logistics", "WATCHLIST_ONLY", 0.064, -0.098, "NO_FILING", "NO_READABLE_ANNUAL_FILING", ("FILING_RISK_NO_FILING",)),
    _Row("AMCX", "telecom", "WATCHLIST_ONLY", 0.043, -0.047, "NO_FILING", "NO_READABLE_ANNUAL_FILING", ("FILING_RISK_NO_FILING",)),
)

# A clean clearing name with full evidence and no signals: a genuine PASS, not a
# data-only downgrade and not DATA_INCOMPLETE.
_CLEAN = _Row("HG", "insurance", "WATCHLIST_ONLY", 0.317, -0.029)


def _packet_for(row: _Row) -> SectorCompanyFinancialPacket:
    packet = SectorCompanyFinancialPacket(
        ticker=row.ticker,
        financial_status="Financially Viable",
        model_fit_status="VALID_GENERIC",
        data_quality_status=row.data_quality_status,
        current_price=50.0,
        valuation={"valuation_anchor": 100.0, "anchor_method": "dcf"},
    )
    if row.filing_risk_evidence_status is not None:
        packet.accounting_quality = {
            "filing_risk_evidence_status": row.filing_risk_evidence_status
        }
    if row.packet_caps:
        packet.confidence_caps = list(row.packet_caps)
    if row.solvency_risk is not None:
        packet.balance_sheet = {"solvency_risk": row.solvency_risk}
    if row.impairment_class_primary is not None:
        packet.business_quality = {
            "impairment_class_primary": row.impairment_class_primary
        }
    return packet


def _scenarios_for(row: _Row) -> list[SectorExpectedReturnScenario]:
    return [
        SectorExpectedReturnScenario(
            scenario_id=f"{row.ticker}_base_5Y",
            ticker=row.ticker,
            scenario_name="base",
            horizon_years=5,
            current_price=50.0,
            estimated_future_value_per_share=100.0,
            annualized_return=row.base_return,
        ),
        SectorExpectedReturnScenario(
            scenario_id=f"{row.ticker}_downside_5Y",
            ticker=row.ticker,
            scenario_name="downside",
            horizon_years=5,
            current_price=50.0,
            estimated_future_value_per_share=45.0,
            annualized_return=row.downside_return,
        ),
    ]


def _passing_tool_evidence(
    ticker: str,
) -> tuple[list[ToolCallRecord], list[EvidenceReference]]:
    tool_calls = [
        ToolCallRecord("TC1", "rank_expected_return_cases", {"tickers": [ticker], "horizon_years": 5}, "Rank.", status="OK", evidence_ref_ids=["E1"]),
        ToolCallRecord("TC2", "fetch_kpi_trends", {}, "Validate KPI.", status="OK", evidence_ref_ids=["E2"]),
        ToolCallRecord("TC3", "analyze_liquidity_stress", {}, "Validate liquidity.", status="OK", evidence_ref_ids=["E3"]),
    ]
    evidence = [
        EvidenceReference("E1", "tool_output", "rank_expected_return_cases", f"{ticker} ranked first.", ticker, tool_call_id="TC1", confidence="MODERATE"),
        EvidenceReference("E2", "tool_output", "fetch_kpi_trends", f"{ticker} KPI evidence.", ticker, tool_call_id="TC2", confidence="MODERATE"),
        EvidenceReference("E3", "tool_output", "analyze_liquidity_stress", f"{ticker} liquidity evidence.", ticker, tool_call_id="TC3", confidence="MODERATE"),
    ]
    return tool_calls, evidence


def _audit_row(row: _Row) -> dict:
    tool_calls, evidence = _passing_tool_evidence(row.ticker)
    return _selection_audit_for_ticker(
        selected_ticker=row.ticker,
        packets_by_ticker={row.ticker: _packet_for(row)},
        scenarios=_scenarios_for(row),
        tool_calls=tool_calls,
        evidence=evidence,
        degraded_states=[],
    )


# --- counterfactual: names x old-status x new-status. ---


def test_counterfactual_dtb_data_only_name_flips_to_data_incomplete():
    # The DTB-style row (clearing return + ONLY NO_FILING-class hard blockers)
    # was WATCHLIST_ONLY before the tier and must now read DATA_INCOMPLETE.
    audit = _audit_row(_DTB)

    assert _DTB.old_status == "WATCHLIST_ONLY"
    assert audit["status"] == "DATA_INCOMPLETE"
    assert audit["actionable"] is False
    assert audit["data_incomplete"] is True
    assert audit["data_resolution_needed"] == [
        "NO_FILING",
        "NO_READABLE_ANNUAL_FILING",
        "FILING_RISK_NO_FILING",
        "CAPITAL_LOSS_EVIDENCE_DEGRADED",
    ]
    assert audit["confidence_ceiling"] is None


def test_counterfactual_data_only_downgrade_count_is_one(monkeypatch):
    # Build the counterfactual table over the modeled investigation slice and
    # count names whose status flips from a data-only BLOCKED/WATCHLIST_ONLY
    # downgrade into DATA_INCOMPLETE. Among DTB + three below-hurdle NO_FILING
    # names + one clean name, exactly ONE (DTB) is a data-only flip.
    # The below-hurdle hard-block path is now the 'hard' rollback mode.
    from app.config import get_config

    monkeypatch.setenv("BASE_RETURN_HURDLE_MODE", "hard")
    get_config.cache_clear()
    rows = [_DTB, *_BELOW_HURDLE_NO_FILING, _CLEAN]

    table = {row.ticker: (row.old_status, _audit_row(row)["status"]) for row in rows}

    # Exact per-name old -> new transitions.
    assert table == {
        "DTB": ("WATCHLIST_ONLY", "DATA_INCOMPLETE"),
        "CHDN": ("WATCHLIST_ONLY", "BLOCKED"),
        "BCO": ("WATCHLIST_ONLY", "BLOCKED"),
        "AMCX": ("WATCHLIST_ONLY", "BLOCKED"),
        "HG": ("WATCHLIST_ONLY", "PASS"),
    }

    data_only_flips = [
        ticker
        for ticker, (old, new) in table.items()
        if old in {"BLOCKED", "WATCHLIST_ONLY"} and new == "DATA_INCOMPLETE"
    ]
    assert data_only_flips == ["DTB"]
    assert len(data_only_flips) == 1


def test_counterfactual_below_hurdle_no_filing_names_stay_blocked(monkeypatch):
    # NO_FILING names that also miss the 12% hurdle keep BLOCKED: a binding hard
    # hurdle blocker dominates the (now non-binding) data gap — they do NOT leak
    # into DATA_INCOMPLETE.
    # The below-hurdle hard-block path is now the 'hard' rollback mode.
    from app.config import get_config

    monkeypatch.setenv("BASE_RETURN_HURDLE_MODE", "hard")
    get_config.cache_clear()
    for row in _BELOW_HURDLE_NO_FILING:
        audit = _audit_row(row)
        assert audit["status"] == "BLOCKED", row.ticker
        assert audit["data_incomplete"] is False, row.ticker


def test_counterfactual_clean_name_passes_not_data_incomplete():
    # A clearing name with full evidence and no signals is a genuine PASS, never
    # mislabeled DATA_INCOMPLETE.
    audit = _audit_row(_CLEAN)
    assert audit["status"] == "PASS"
    assert audit["data_incomplete"] is False


# --- acceptance guards: solvency/economics must not leak through. ---


def test_data_only_name_with_genuine_business_blocker_stays_blocked():
    # The DTB row PLUS a genuine BUSINESS_QUALITY hard blocker
    # (PROBABLE_PERMANENT_CAPITAL_LOSS from a PROBABLE_IMPAIRMENT class) must
    # still hard-block — a real economics/solvency risk does NOT get tolerated
    # as a mere data gap.
    row = _Row(
        ticker="DTBX",
        sector="utilities",
        old_status="WATCHLIST_ONLY",
        base_return=0.578,
        downside_return=0.117,
        data_quality_status="NO_FILING",
        filing_risk_evidence_status="NO_READABLE_ANNUAL_FILING",
        packet_caps=("FILING_RISK_NO_FILING",),
        impairment_class_primary="PROBABLE_IMPAIRMENT",
    )
    audit = _audit_row(row)

    assert audit["status"] == "BLOCKED"
    assert audit["actionable"] is False
    assert audit["data_incomplete"] is False
    assert "PROBABLE_PERMANENT_CAPITAL_LOSS" in audit["hard_blockers"]


def test_thin_cushion_data_only_name_stays_watchlist_only():
    # A thin (in [0.12, 0.15)) base return raises THIN_RETURN_CUSHION, a binding
    # HURDLE cap. Alongside a NO_FILING gap, the hurdle cap dominates and the
    # name stays WATCHLIST_ONLY (HURDLE precedence over DATA_INCOMPLETE).
    row = _Row(
        ticker="THIN",
        sector="utilities",
        old_status="WATCHLIST_ONLY",
        base_return=0.13,
        downside_return=-0.02,
        data_quality_status="NO_FILING",
    )
    audit = _audit_row(row)

    assert audit["status"] == "WATCHLIST_ONLY"
    assert audit["data_incomplete"] is False


def test_below_hurdle_data_only_name_stays_blocked(monkeypatch):
    # A below-hurdle (0.05) base return raises BASE_RETURN_BELOW_12PCT_HURDLE, a
    # binding HARD blocker. With a NO_FILING gap it stays BLOCKED — the hurdle
    # miss dominates the data gap.
    # This hard-block path is now the 'hard' rollback mode (default is 'soft').
    from app.config import get_config

    monkeypatch.setenv("BASE_RETURN_HURDLE_MODE", "hard")
    get_config.cache_clear()
    row = _Row(
        ticker="LOW",
        sector="utilities",
        old_status="WATCHLIST_ONLY",
        base_return=0.05,
        downside_return=-0.02,
        data_quality_status="NO_FILING",
    )
    audit = _audit_row(row)

    assert audit["status"] == "BLOCKED"
    assert audit["data_incomplete"] is False
