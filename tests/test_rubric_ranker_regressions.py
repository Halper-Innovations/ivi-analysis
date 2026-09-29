"""Regression tests for the scoring path: app/score/rubric.py,
app/score/ranker.py and the packet/fundamentals producers that feed them.

Each test asserts the correct answer for a defect that was found and fixed in the
scoring slice. All fixtures are hermetic: temp DB via ``init_db``, no network.
"""

from __future__ import annotations

import json

import app.evidence.packet_builder as pb
from app.db import get_db, init_db, utc_now_iso
from app.fundamentals import metrics
from app.fundamentals.normalize import UNKNOWN
from app.score.ranker import _latest_research_signals
from app.score.rubric import score_packet

TICKER = "TEST"
AS_OF = "2026-06-30"


def _init_temp_db(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(data_dir / "engine.db"))
    monkeypatch.setenv("VOE_SAFE_MODE", "true")
    from app.config import get_config

    get_config.cache_clear()
    cfg = get_config()
    init_db(cfg)
    return cfg


def _packet(fundamentals: dict) -> dict:
    return {
        "ticker": TICKER,
        "as_of_date": AS_OF,
        "fundamentals": fundamentals,
        "valuations": {},
        "filings": [],
        "extracted_facts": [],
        "deltas_vs_prior_period": {},
    }


def _stored_metrics(conn, as_of: str) -> dict:
    row = conn.execute(
        "SELECT metrics_json FROM fundamentals WHERE ticker = ? AND as_of_date = ?",
        (TICKER, as_of),
    ).fetchone()
    return json.loads(row["metrics_json"])


def _insert_filing(conn, *, accession: str, form: str, filed: str) -> int:
    now = utc_now_iso()
    cur = conn.execute(
        "INSERT INTO filings(cik, ticker, accession, form_type, filing_date, period_end,"
        " primary_doc_url, status, created_at, updated_at)"
        " VALUES('0000000000', ?, ?, ?, ?, NULL, 'https://www.sec.gov/x', 'parsed', ?, ?)",
        (TICKER, accession, form, filed, now, now),
    )
    return int(cur.lastrowid)


# ── app/fundamentals/metrics.py: the no-filing fallback invents a liquidity score


COMPANYFACTS_MAP = {
    "revenue": 1000.0,
    "gross_profit": 400.0,
    "operating_income": 100.0,
    "net_income": 80.0,
    "cfo": 120.0,
    "capex": 30.0,
    "cash": 50.0,
    "total_debt": 500.0,
}


def test_no_filing_fallback_scores_liquidity_as_perfect(monkeypatch, tmp_path):
    """An issuer with no parsed filing has no liquidity evidence; the stored
    score must be UNKNOWN and the rubric must apply its own neutral 4.0.

    Mechanism: ``compute_fundamentals_for_ticker`` sets ``liq_score = 0`` on the
    companyfacts fallback (``app/fundamentals/metrics.py``, "liq_score = 0")
    and ``score_packet`` turns 0 into ``clamp(10 - 0) = 10.0``. The rubric's
    treatment of an unknown score is 4.0; a parsed filing carrying going-concern
    + covenant + refinancing facts scores 4.0 too. The packet carries only
    ``metrics_json``, so the "companyfacts_fallback" quality flag never reaches
    the rubric.

    Observed: stored liquidity_stress_score 0, balance_sheet_dilution 10.0 —
    the never-examined issuer outscores the going-concern issuer by 6 points.
    Correct: stored UNKNOWN, balance_sheet_dilution 4.0.
    (The companyfacts map lookup is stubbed; the mechanism is downstream of it.)
    """
    _init_temp_db(monkeypatch, tmp_path)
    monkeypatch.setattr(
        metrics,
        "_companyfacts_map_for_ticker",
        lambda conn, ticker, as_of_date=None: (dict(COMPANYFACTS_MAP), "2025-12-31"),
    )
    assert metrics.compute_fundamentals_for_ticker(TICKER, as_of_date=AS_OF) is True
    with get_db() as conn:
        stored = _stored_metrics(conn, AS_OF)
    subscores, _, _, _ = score_packet(_packet(stored))
    assert subscores["balance_sheet_dilution"] == 4.0
    assert stored["liquidity_stress_score"] == UNKNOWN


def test_rubric_scores_an_unknown_liquidity_score_neutral():
    """Control: the rubric's own convention for an unmeasured liquidity score."""
    fundamentals = {**COMPANYFACTS_MAP, "liquidity_stress_score": UNKNOWN}
    subscores, _, _, _ = score_packet(_packet(fundamentals))
    assert subscores["balance_sheet_dilution"] == 4.0


# ── app/score/ranker.py: research signals chosen from the wrong date ─────────


def _insert_signals(
    conn, *, as_of: str, run_id: str, created_at: str, recency: int, count: int, flags
):
    conn.execute(
        """
        INSERT INTO research_signals(
            ticker, as_of_date, run_id, recency_days_min, item_count_30d,
            has_earnings_release, has_investor_presentation, sentiment_flags_json,
            key_topics_json, evidence_item_ids_json, summary_json, created_at
        ) VALUES(?, ?, ?, ?, ?, 0, 0, ?, '[]', '[]', '{}', ?)
        """,
        (TICKER, as_of, run_id, recency, count, json.dumps(list(flags)), created_at),
    )


def test_research_signals_from_after_the_scoring_date_are_used(monkeypatch, tmp_path):
    """Scoring as of 2025-03-31 must not see a signals row dated 2025-09-30.

    Mechanism: the first two queries are bounded (exact run, then
    ``as_of_date <= ?``); the third is ``WHERE ticker = ? ORDER BY as_of_date
    DESC`` and is reached exactly when nothing at or before the scoring date
    exists. ``recency_days_min`` / ``item_count_30d`` in that row are measured
    from 2025-09-30, and its sentiment flags become the March score's reasons.

    Observed: the 2025-09-30 row is returned and ``compute_research_adjustments``
    applies +2.0 (fresh, positive_guidance) to the 2025-03-31 score instead of
    the -4.0 no-research adjustment. Correct: None.
    """
    _init_temp_db(monkeypatch, tmp_path)
    with get_db() as conn:
        _insert_signals(
            conn,
            as_of="2025-09-30",
            run_id="run-future",
            created_at="2025-09-30T12:00:00Z",
            recency=2,
            count=5,
            flags=["positive_guidance"],
        )
        assert _latest_research_signals(conn, TICKER, "2025-03-31") is None


def test_backfilled_older_signals_outrank_the_scoring_dates_row(monkeypatch, tmp_path):
    """When a row for the scoring date exists it must be the one used.

    Mechanism: ``WHERE as_of_date <= ? ORDER BY created_at DESC`` — a row for
    2024-12-31 written on 2025-07-15 (a backfill) is newer by ``created_at``
    than the 2025-06-30 row written on 2025-06-30, so the 2025-06-30 score is
    fed a 170-day-stale, negative-guidance row while a same-day row sits unused.

    Observed: as_of_date "2024-12-31" (run "run-backfill"). Correct: "2025-06-30".
    """
    _init_temp_db(monkeypatch, tmp_path)
    with get_db() as conn:
        _insert_signals(
            conn,
            as_of="2025-06-30",
            run_id="run-jun",
            created_at="2025-06-30T12:00:00Z",
            recency=3,
            count=4,
            flags=[],
        )
        _insert_signals(
            conn,
            as_of="2024-12-31",
            run_id="run-backfill",
            created_at="2025-07-15T12:00:00Z",
            recency=170,
            count=0,
            flags=["negative_guidance"],
        )
        got = _latest_research_signals(conn, TICKER, "2025-06-30")
    assert got is not None
    assert got["as_of_date"] == "2025-06-30"


# ── legacy packet path: the filing route stores UNKNOWN margins the overlay
#    never fills, and the filings/facts query has no as-of bound


THIRTY_PERCENT_MARGIN_FRAME = {
    "ticker": TICKER,
    "as_of_date": AS_OF,
    "rows": [
        {
            "year": 2025,
            "revenue": 1000.0,
            "operating_income": 300.0,
            "op_margin": 0.30,
            "fcf": 250.0,
            "fcf_margin": 0.25,
            "cfo": 280.0,
            "capex": 30.0,
        }
    ],
    "row_traces": {},
    "derived_signals": {},
    "gaps": [],
}


def test_filing_route_margins_stay_unknown_after_companyfacts_overlay(monkeypatch, tmp_path):
    """A 30%-operating-margin, 25%-FCF-margin issuer with a parsed 10-K must
    score 20/20 on business quality on the legacy packet path.

    Mechanism: ``_compute_fundamentals_for_filing`` builds its map from the
    ``financials`` table, which no production code populates (the parser only
    DELETEs from it), so ``compute_metrics_from_financials({})`` stores
    revenue / operating_margin / fcf_margin as "UNKNOWN". ``_build_packet_payload``
    then runs ``_overlay_companyfacts_trends``, whose fill is guarded by
    ``if key not in fundamentals`` — every key IS present (as "UNKNOWN"), so the
    companyfacts numbers land in ``fundamentals["rows"]`` but never in the flat
    keys ``score_packet`` reads. Net: business_quality_durability is the neutral
    10.0 for every issuer on this path, and revenue/fcf momentum deltas can
    never form. An issuer WITHOUT a parsed filing (companyfacts fallback) gets
    real margins — the better-documented company scores worse.

    Observed: fundamentals["operating_margin"] "UNKNOWN", rows[-1]["op_margin"]
    0.3, business_quality_durability 10.0. Correct: 0.3 and 20.0.
    (The companyfacts-frame DB read is stubbed; the mechanism is the overlay guard.)
    """
    filed = "2026-02-15"  # the filing route keys the fundamentals row on the filing date
    _init_temp_db(monkeypatch, tmp_path)
    monkeypatch.setattr(
        pb,
        "_companyfacts_frame",
        lambda conn, *, ticker, as_of_date: dict(THIRTY_PERCENT_MARGIN_FRAME),
    )
    with get_db() as conn:
        _insert_filing(conn, accession="0000000000-26-000001", form="10-K", filed=filed)
    assert metrics.compute_fundamentals_for_ticker(TICKER, as_of_date=AS_OF) is True
    with get_db() as conn:
        assert _stored_metrics(conn, filed)["operating_margin"] == UNKNOWN  # what the route stores
        payload = pb._build_packet_payload(conn, TICKER, filed)
    assert payload is not None
    assert payload["fundamentals"]["rows"][-1]["op_margin"] == 0.3  # the number is right there
    subscores, _, _, _ = score_packet(payload)
    assert payload["fundamentals"]["operating_margin"] == 0.3
    assert subscores["business_quality_durability"] == 20.0


def test_legacy_packet_uses_filings_made_after_its_as_of_date(monkeypatch, tmp_path):
    """A packet dated 2025-03-31 must list and use only filings made by then.

    Mechanism: the fundamentals and valuation lookups in ``_build_packet_payload``
    are exact-as-of, and the metadata fallback ``_cached_filing_metadata_for_ticker``
    is as-of bounded, but the filings query is ``WHERE ticker = ? ORDER BY
    COALESCE(filing_date, '1900-01-01') DESC LIMIT 4``. Its ``extracted_facts``
    (restatement, going_concern, ...) come from those filings, and
    ``score_packet`` computes special_situations from them.

    Observed: filings_used dated 2025-09-15, 2025-08-15, 2025-07-15, 2025-06-15
    (the one filing made before the as-of is not even listed); four restatement
    facts; special_situations 10 for a March packet. Correct: only the
    2025-02-15 filing, no facts, special_situations 0.0.
    """
    packet_as_of = "2025-03-31"
    _init_temp_db(monkeypatch, tmp_path)
    with get_db() as conn:
        now = utc_now_iso()
        conn.execute(
            "INSERT INTO fundamentals(ticker, as_of_date, metrics_json, quality_flags_json, created_at)"
            " VALUES(?, ?, ?, '{}', ?)",
            (TICKER, packet_as_of, json.dumps({"revenue": UNKNOWN}), now),
        )
        _insert_filing(conn, accession="0000000000-25-000001", form="10-K", filed="2025-02-15")
        for i, month in ((2, 6), (3, 7), (4, 8), (5, 9)):
            filing_id = _insert_filing(
                conn, accession=f"0000000000-25-00000{i}", form="8-K", filed=f"2025-{month:02d}-15"
            )
            conn.execute(
                "INSERT INTO extracted_facts(filing_id, fact_type, value_json, source_url, snippet,"
                " section_label, created_at)"
                " VALUES(?, 'restatement', '{}', 'https://www.sec.gov/x', '', NULL, ?)",
                (filing_id, now),
            )
        payload = pb._build_packet_payload(conn, TICKER, packet_as_of)
    assert payload is not None
    subscores, _, _, _ = score_packet(payload)
    assert [f["filing_date"] for f in payload["filings_used"]] == ["2025-02-15"]
    assert payload["extracted_facts"] == []
    assert subscores["special_situations"] == 0.0


# ── app/score/rubric.py: data completeness counts non-metric keys ─────────────


STRUCTURAL_KEYS = {
    "ticker",
    "run_id",
    "as_of_date",
    "rows",
    "row_traces",
    "derived_signals",
    "gaps",
}


def test_data_completeness_credits_structural_keys():
    """A fundamentals block with no known metric must score 0/20 on completeness.

    Mechanism: ``score_packet`` computes ``known_metrics = sum(1 for _, v in
    fundamentals.items() if v != UNKNOWN)`` over the whole dict.
    ``_flat_fundamentals_from_frame`` (the dossier packet path) always adds
    ticker, run_id, as_of_date, rows, row_traces, derived_signals and gaps —
    seven values that are never "UNKNOWN" (``None`` counts too).

    Observed: 7/46 "known" for an empty frame -> data_completeness 3.04.
    Correct: 0.0 (and the same dict with the structural keys removed already
    scores 0.0).
    """
    empty = pb._flat_fundamentals_from_frame(
        {"ticker": TICKER, "run_id": None, "as_of_date": AS_OF}
    )
    assert all(v == UNKNOWN for k, v in empty.items() if k not in STRUCTURAL_KEYS)
    metric_only = {k: v for k, v in empty.items() if k not in STRUCTURAL_KEYS}
    assert score_packet(_packet(metric_only))[0]["data_completeness"] == 0.0
    assert score_packet(_packet(empty))[0]["data_completeness"] == 0.0
