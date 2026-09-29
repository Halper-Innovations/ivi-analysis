"""Integration tests for sector-relative cross-sectional ranking.

Regression coverage that the autonomous packet's ``_valuation``
continues to surface the raw top-level ``implied_growth`` key that the
cross-sectional gap factor consumes. The scalar ``expectations_gap`` key and the
``expectations_gap_bucket`` are separate outputs and MUST remain a scalar /
bucket string respectively; these tests do NOT redefine them.
"""

from __future__ import annotations

import json

from app.alpha.schemas import TickerSignalPacket
from app.autonomous.sector_financial_packets import _valuation


def _init_temp_db(monkeypatch, tmp_path):
    from app.config import get_config
    from app.db import init_db

    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    get_config.cache_clear()
    init_db()
    return db_path


def _seed_reverse_dcf_row(ticker: str, as_of_date: str, outputs_json: dict) -> None:
    from app.db import get_db

    with get_db() as conn:
        conn.execute(
            """INSERT INTO valuations
               (ticker, as_of_date, method, inputs_json, outputs_json, warnings_json, created_at)
               VALUES (?, ?, 'reverse_dcf', '{}', ?, '[]', ?)""",
            (ticker.upper(), as_of_date, json.dumps(outputs_json), f"{as_of_date}T00:00:00+00:00"),
        )


def _minimal_packet(ticker: str) -> TickerSignalPacket:
    return TickerSignalPacket(
        ticker=ticker,
        dcf_value=100.0,
        current_price=50.0,
    )


# ── raw implied_growth gap factor surfaced top-level on the packet ──────


def test_expectations_gap_factor_surfaces_raw_implied_growth(monkeypatch, tmp_path):
    """AAA: a valid reverse_dcf row exposes implied_growth as a top-level float."""
    _init_temp_db(monkeypatch, tmp_path)
    _seed_reverse_dcf_row(
        "AAA",
        "2026-05-16",
        {
            "status": "OK",
            "outputs": {"implied_growth": 0.05, "feasibility_gap_score": 0.0},
            "expectations_gap": {
                "gap": -0.05,
                "bucket": "CHEAP_VS_EXPECTATIONS",
                "supportable_growth": 0.10,
                "implied_growth_saturated": False,
            },
        },
    )

    valuation = _valuation(_minimal_packet("AAA"), "dcf", 100.0, sector=None, as_of_date="2026-05-16")

    assert valuation["implied_growth"] == 0.05
    assert isinstance(valuation["implied_growth"], float)


def test_expectations_gap_factor_none_when_no_reverse_dcf_row(monkeypatch, tmp_path):
    """BBB: no reverse_dcf row -> implied_growth is None, bucket is UNRELIABLE."""
    _init_temp_db(monkeypatch, tmp_path)

    valuation = _valuation(_minimal_packet("BBB"), "dcf", 100.0, sector=None, as_of_date="2026-05-16")

    assert valuation["implied_growth"] is None
    assert valuation["expectations_gap_bucket"] == "EXPECTATIONS_GAP_UNRELIABLE"


def test_expectations_gap_factor_coerces_unknown_implied_growth_to_none(monkeypatch, tmp_path):
    """CCC: a non-numeric 'UNKNOWN' implied_growth sentinel coerces to None (does not raise)."""
    _init_temp_db(monkeypatch, tmp_path)
    _seed_reverse_dcf_row(
        "CCC",
        "2026-05-16",
        {
            "status": "UNKNOWN",
            "outputs": {"implied_growth": "UNKNOWN"},
            "expectations_gap": {
                "gap": None,
                "bucket": "EXPECTATIONS_GAP_UNRELIABLE",
                "supportable_growth": 0.10,
            },
        },
    )

    valuation = _valuation(_minimal_packet("CCC"), "dcf", 100.0, sector=None, as_of_date="2026-05-16")

    assert valuation["implied_growth"] is None


# ── factor-vector extraction + cross-sectional ranker integration ───────


def _ranking_packet(
    ticker: str,
    *,
    discount_to_anchor=None,
    roic=None,
    roic_wacc_spread=None,
    fcf_margin=None,
    implied_growth=None,
):
    from app.autonomous.sector_contract import SectorCompanyFinancialPacket

    return SectorCompanyFinancialPacket(
        ticker=ticker,
        financial_status="Financially Viable",
        model_fit_status="VALID_GENERIC",
        data_quality_status="OK",
        current_price=50.0,
        returns_on_capital={
            k: v
            for k, v in (
                ("roic", roic),
                ("roic_wacc_spread", roic_wacc_spread),
            )
            if v is not None
        },
        cash_conversion={"fcf_margin": fcf_margin} if fcf_margin is not None else {},
        valuation={
            **(
                {"discount_to_anchor": discount_to_anchor}
                if discount_to_anchor is not None
                else {}
            ),
            **({"implied_growth": implied_growth} if implied_growth is not None else {}),
            "valuation_anchor": 100.0,
            "anchor_method": "dcf",
        },
    )


def _base_scenario(ticker: str, annualized_return: float):
    from app.autonomous.sector_contract import SectorExpectedReturnScenario

    return SectorExpectedReturnScenario(
        scenario_id=f"{ticker}_base_5Y",
        ticker=ticker,
        scenario_name="base",
        horizon_years=5,
        current_price=50.0,
        estimated_future_value_per_share=100.0,
        annualized_return=annualized_return,
    )


def _rank_rows(packets, scenarios=None):
    from app.autonomous.sector_runtime import _relative_ranking

    return _relative_ranking(
        company_packets=packets,
        scenarios=scenarios or [],
        tool_calls=[],
        evidence=[],
        degraded_states=[],
        company_autonomy_runs=[],
    )


def test_relative_ranking_quality_leader_beats_largest_discount_junk():
    """LEADER (best blended z) ranks 1 and is a buy_candidate; JUNK (biggest raw
    discount but negative ROIC + worst expectations gap) is NOT a buy_candidate."""
    # value z (discount higher-better): LEADER 0.1162, MID -1.2787, JUNK 1.1625
    # quality z (roic higher-better):   LEADER 1.1076, MID 0.2077, JUNK -1.3153
    # gap z (implied_growth lower-better, negated): LEADER 0.9139, MID 0.4777, JUNK -1.3916
    # composite (0.4/0.4/0.2): LEADER 0.6723, MID -0.3329, JUNK -0.3394
    # buy_cutoff = ceil(3 * 0.20) = 1
    packets = [
        _ranking_packet("LEADER", discount_to_anchor=0.30, roic=0.25, implied_growth=0.03),
        _ranking_packet("MID", discount_to_anchor=0.10, roic=0.12, implied_growth=0.10),
        _ranking_packet("JUNK", discount_to_anchor=0.45, roic=-0.10, implied_growth=0.40),
    ]
    rows = _rank_rows(packets)
    by_ticker = {row["ticker"]: row for row in rows}

    assert by_ticker["LEADER"]["cross_sectional_rank"] == 1
    assert by_ticker["LEADER"]["cross_sectional_score"] == 0.6723
    assert by_ticker["LEADER"]["buy_candidate"] is True
    assert by_ticker["JUNK"]["buy_candidate"] is False
    assert by_ticker["JUNK"]["cross_sectional_score"] == -0.3394


def test_relative_ranking_attaches_cross_sectional_keys_on_every_row():
    packets = [
        _ranking_packet("LEADER", discount_to_anchor=0.30, roic=0.25, implied_growth=0.03),
        _ranking_packet("MID", discount_to_anchor=0.10, roic=0.12, implied_growth=0.10),
        _ranking_packet("JUNK", discount_to_anchor=0.45, roic=-0.10, implied_growth=0.40),
    ]
    rows = _rank_rows(packets)

    required = {
        "cross_sectional_score",
        "cross_sectional_rank",
        "cross_sectional_percentile",
        "factor_zscores",
        "buy_candidate",
        "buy_candidate_reason",
    }
    for row in rows:
        assert required.issubset(row.keys())


def test_relative_ranking_sorts_by_composite_then_base_return():
    """Two equal-composite (identical factor) packets break the tie by the higher
    best_base_annualized_return."""
    packets = [
        _ranking_packet("TIE_LO", discount_to_anchor=0.20, roic=0.15, implied_growth=0.08),
        _ranking_packet("TIE_HI", discount_to_anchor=0.20, roic=0.15, implied_growth=0.08),
    ]
    scenarios = [
        _base_scenario("TIE_LO", 0.05),
        _base_scenario("TIE_HI", 0.20),
    ]
    rows = _rank_rows(packets, scenarios=scenarios)

    # Identical factors -> population stdev 0 -> all z 0.0 -> composite 0.0 for both.
    assert rows[0]["cross_sectional_score"] == 0.0
    assert rows[1]["cross_sectional_score"] == 0.0
    assert rows[0]["ticker"] == "TIE_HI"
    assert rows[1]["ticker"] == "TIE_LO"


def test_relative_ranking_all_none_factors_ranked_last_not_buy():
    packets = [
        _ranking_packet("LEADER", discount_to_anchor=0.30, roic=0.25, implied_growth=0.03),
        _ranking_packet("MID", discount_to_anchor=0.10, roic=0.12, implied_growth=0.10),
        _ranking_packet("NOFACTORS"),
    ]
    rows = _rank_rows(packets)
    by_ticker = {row["ticker"]: row for row in rows}

    assert by_ticker["NOFACTORS"]["cross_sectional_score"] is None
    assert by_ticker["NOFACTORS"]["buy_candidate"] is False
    assert by_ticker["NOFACTORS"]["buy_candidate_reason"] == "NO_RANKABLE_FACTORS"
    assert rows[-1]["ticker"] == "NOFACTORS"


def test_relative_ranking_composite_floor_demotes_top_at_sector_mean():
    """composite>0 floor (applied at the integration layer): a top-quantile row
    whose composite is AT the sector mean (composite == 0.0, the 'best house in a
    bad neighborhood') is demoted to buy_candidate=False with reason
    BELOW_SECTOR_MEAN — the pure ranker would have flagged it True at rank 1."""
    packets = [
        _ranking_packet("TIE_LO", discount_to_anchor=0.20, roic=0.15, implied_growth=0.08),
        _ranking_packet("TIE_HI", discount_to_anchor=0.20, roic=0.15, implied_growth=0.08),
    ]
    scenarios = [
        _base_scenario("TIE_LO", 0.05),
        _base_scenario("TIE_HI", 0.20),
    ]
    rows = _rank_rows(packets, scenarios=scenarios)
    by_ticker = {row["ticker"]: row for row in rows}

    # TIE_LO is rank 1 (ranker would flag it buy_candidate at composite 0.0).
    assert by_ticker["TIE_LO"]["cross_sectional_rank"] == 1
    assert by_ticker["TIE_LO"]["cross_sectional_score"] == 0.0
    assert by_ticker["TIE_LO"]["buy_candidate"] is False
    assert by_ticker["TIE_LO"]["buy_candidate_reason"] == "BELOW_SECTOR_MEAN"
    assert by_ticker["TIE_HI"]["buy_candidate"] is False


def test_relative_ranking_composite_floor_demotes_single_member_sector():
    """A single-member sector yields composite 0.0 (all-equal z) with the pure
    ranker flagging rank 1 as a buy_candidate; the integration-layer floor
    demotes it to buy_candidate=False with reason BELOW_SECTOR_MEAN."""
    packets = [
        _ranking_packet("ONLY", discount_to_anchor=0.20, roic=0.15, implied_growth=0.08),
    ]
    rows = _rank_rows(packets)

    assert rows[0]["ticker"] == "ONLY"
    assert rows[0]["cross_sectional_score"] == 0.0
    assert rows[0]["cross_sectional_rank"] == 1
    assert rows[0]["buy_candidate"] is False
    assert rows[0]["buy_candidate_reason"] == "BELOW_SECTOR_MEAN"


def test_relative_ranking_composite_floor_keeps_positive_composite_buy():
    """The floor only demotes composite <= 0: a top-quantile row with a strictly
    positive composite (above the sector mean) remains a buy_candidate."""
    packets = [
        _ranking_packet("LEADER", discount_to_anchor=0.30, roic=0.25, implied_growth=0.03),
        _ranking_packet("MID", discount_to_anchor=0.10, roic=0.12, implied_growth=0.10),
        _ranking_packet("JUNK", discount_to_anchor=0.45, roic=-0.10, implied_growth=0.40),
    ]
    rows = _rank_rows(packets)
    by_ticker = {row["ticker"]: row for row in rows}

    assert by_ticker["LEADER"]["cross_sectional_score"] == 0.6723
    assert by_ticker["LEADER"]["buy_candidate"] is True
    assert by_ticker["LEADER"]["buy_candidate_reason"] is None


# ── 12% base-return hurdle demoted from hard BLOCK to soft confidence cap ─


def _hurdle_audit(monkeypatch, mode: str, base_return: float):
    """Audit a single clean packet whose only deterministic gate is the base-return
    hurdle, under the given BASE_RETURN_HURDLE_MODE. Returns the audit dict."""
    from app.config import get_config

    monkeypatch.setenv("BASE_RETURN_HURDLE_MODE", mode)
    get_config.cache_clear()

    from app.autonomous.sector_runtime import _selection_audit_for_ticker

    packet = _ranking_packet("HURD", discount_to_anchor=0.30, roic=0.25, implied_growth=0.03)
    scenarios = [_base_scenario("HURD", base_return)]
    return _selection_audit_for_ticker(
        selected_ticker="HURD",
        packets_by_ticker={"HURD": packet},
        scenarios=scenarios,
        tool_calls=[],
        evidence=[],
        degraded_states=[],
    )


def test_hurdle_soft_mode_below_hurdle_caps_not_blocks(monkeypatch):
    """Soft mode (default): a 0.08 best base return surfaces a soft confidence cap,
    NOT a hard blocker, and marks the return cushion BELOW_HURDLE."""
    audit = _hurdle_audit(monkeypatch, "soft", 0.08)

    assert audit["status"] != "BLOCKED"
    assert "BASE_RETURN_BELOW_12PCT_HURDLE" not in audit["hard_blockers"]
    assert "BASE_RETURN_BELOW_HURDLE_SOFT" in audit["confidence_caps"]
    assert audit["return_cushion_status"] == "BELOW_HURDLE"


def test_hurdle_soft_mode_below_hurdle_is_watchlist_only(monkeypatch):
    """Soft mode: the soft hurdle cap is a binding HURDLE cap, so the candidate
    lands in WATCHLIST_ONLY with a MODERATE/LOW ceiling (never AVOID/BLOCKED)."""
    audit = _hurdle_audit(monkeypatch, "soft", 0.08)

    assert audit["status"] == "WATCHLIST_ONLY"
    assert audit["confidence_ceiling"] in {"MODERATE", "LOW"}


def test_hurdle_soft_cap_classifies_as_hurdle(monkeypatch):
    """The new soft cap code is HURDLE-class so binding_hurdle_caps routes it to
    WATCHLIST_ONLY rather than letting it read as a clean PASS."""
    monkeypatch.setenv("BASE_RETURN_HURDLE_MODE", "soft")
    from app.config import get_config

    get_config.cache_clear()
    from app.autonomous.sector_runtime import classify_audit_signal

    assert classify_audit_signal("BASE_RETURN_BELOW_HURDLE_SOFT") == "HURDLE"


def test_hurdle_hard_mode_below_hurdle_still_blocks(monkeypatch):
    """Backward-compat: hard mode preserves the old behavior — the 0.08 case
    appends BASE_RETURN_BELOW_12PCT_HURDLE to hard_blockers and status is BLOCKED."""
    audit = _hurdle_audit(monkeypatch, "hard", 0.08)

    assert audit["status"] == "BLOCKED"
    assert "BASE_RETURN_BELOW_12PCT_HURDLE" in audit["hard_blockers"]
    assert "BASE_RETURN_BELOW_HURDLE_SOFT" not in audit["confidence_caps"]


def test_hurdle_soft_buy_candidate_below_hurdle_reaches_watchlist(monkeypatch):
    """A clean top-quantile name below the soft hurdle reaches WATCHLIST_ONLY
    (not BLOCKED): the cross-sectional rank survives the demoted hurdle."""
    from app.config import get_config

    monkeypatch.setenv("BASE_RETURN_HURDLE_MODE", "soft")
    get_config.cache_clear()

    from app.autonomous.sector_runtime import _selection_audit_for_ticker

    packet = _ranking_packet("HURD", discount_to_anchor=0.30, roic=0.25, implied_growth=0.03)
    rows = _rank_rows(
        [
            packet,
            _ranking_packet("MID", discount_to_anchor=0.10, roic=0.12, implied_growth=0.10),
            _ranking_packet("JUNK", discount_to_anchor=0.45, roic=-0.10, implied_growth=0.40),
        ]
    )
    by_ticker = {row["ticker"]: row for row in rows}
    assert by_ticker["HURD"]["buy_candidate"] is True

    audit = _selection_audit_for_ticker(
        selected_ticker="HURD",
        packets_by_ticker={"HURD": packet},
        scenarios=[_base_scenario("HURD", 0.08)],
        tool_calls=[],
        evidence=[],
        degraded_states=[],
    )
    assert audit["status"] == "WATCHLIST_ONLY"
    assert audit["status"] != "BLOCKED"


# ── top-quantile buy_candidate -> ACTIONABLE conviction + memo surfacing ─


def _artifact_with_ranking(ranking_rows):
    """Build a minimal run artifact whose relative_ranking holds the given rows.

    selected_ticker is None / final_verdict NO_SELECTION so _conviction_grade does
    NOT take the selected-ticker short-circuit and instead reads the audit status +
    buy_candidate flag from the relative_ranking row.
    """
    from app.autonomous.sector_contract import AutonomousSectorFinancialRunArtifact

    return AutonomousSectorFinancialRunArtifact(
        run_id="run-b6t5",
        sector="enterprise_software",
        market_cap_focus="mid_cap",
        objective="test",
        as_of_date="2026-06-01",
        created_at="2026-06-01T00:00:00+00:00",
        status="COMPLETED",
        final_verdict="NO_SELECTION",
        selected_ticker=None,
        confidence=None,
        relative_ranking=ranking_rows,
    )


def test_conviction_grade_buy_candidate_pass_is_actionable():
    """A top-quantile buy_candidate row with PASS audit status grades ACTIONABLE
    (not the default MODERATE)."""
    from app.autonomous.sector_report import _conviction_grade

    artifact = _artifact_with_ranking(
        [{"ticker": "LEADER", "audit_status": "PASS", "buy_candidate": True}]
    )

    assert _conviction_grade(artifact, "LEADER") == "ACTIONABLE"


def test_conviction_grade_buy_candidate_watchlist_only_stays_watchlist():
    """A buy_candidate that is WATCHLIST_ONLY by audit status stays WATCHLIST —
    the cross-sectional rank does NOT override a binding cap."""
    from app.autonomous.sector_report import _conviction_grade

    artifact = _artifact_with_ranking(
        [{"ticker": "CAPPED", "audit_status": "WATCHLIST_ONLY", "buy_candidate": True}]
    )

    assert _conviction_grade(artifact, "CAPPED") == "WATCHLIST"


def test_conviction_grade_non_buy_candidate_pass_unchanged_moderate():
    """A PASS row that is NOT a buy_candidate keeps the legacy MODERATE grade."""
    from app.autonomous.sector_report import _conviction_grade

    artifact = _artifact_with_ranking(
        [{"ticker": "PLAIN", "audit_status": "PASS", "buy_candidate": False}]
    )

    assert _conviction_grade(artifact, "PLAIN") == "MODERATE"


def test_compact_relative_ranking_surfaces_buy_candidate_and_percentile():
    """The shared-memo compacted ranking dict carries cross_sectional_percentile
    and buy_candidate so the memo can render the relative BUY rank."""
    from app.autonomous.sector_runtime import _compact_relative_ranking_for_shared_memo

    compact = _compact_relative_ranking_for_shared_memo(
        {
            "rank": 1,
            "ticker": "LEADER",
            "audit_status": "PASS",
            "buy_candidate": True,
            "cross_sectional_percentile": 100.0,
            "cross_sectional_score": 0.6723,
            "cross_sectional_rank": 1,
        }
    )

    assert "cross_sectional_percentile" in compact
    assert "buy_candidate" in compact
    assert compact["cross_sectional_percentile"] == 100.0
    assert compact["buy_candidate"] is True
