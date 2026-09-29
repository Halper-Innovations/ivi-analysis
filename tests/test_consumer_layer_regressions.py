"""Regressions for the consumer layer: where a persisted valuation
becomes something a human sees (web company page, dossier valuation section,
synthesis memo signals, watchlist price trigger).

Each test asserts the correct answer for a defect that has been fixed; any
future defect of this kind belongs here as a strict ``xfail`` until fixed.

Fixtures are hermetic (temp DB via ``init_db``; no network). Valuation rows are
inserted in the exact shape ``app/valuation/valuation_writer.py`` persists
them. Row authorization comes from the repo's autouse test fixture
(``tests/conftest.py``: the decision-eligibility predicates return True for
legacy suites); the same reproductions were also executed with a real
integrity manifest and artifact-bound fingerprints and produced identical
output (scratch reproductions).
"""

from __future__ import annotations

import json


from app.db import get_db, init_db
from app.web.readmodel import company

D1 = "2026-08-01"
D2 = "2026-08-15"


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


def _insert_valuation(conn, *, ticker, as_of_date, method, outputs, inputs=None, run_id="run"):
    """Mirror of the writer's per-method persist: warnings_json is always "[]"."""
    conn.execute(
        """
        INSERT INTO valuations (
            ticker, as_of_date, method, inputs_json, outputs_json, warnings_json,
            created_at, source_run_id, valuation_writer_version
        ) VALUES (?, ?, ?, ?, ?, '[]', ?, ?, 'fixture-writer')
        """,
        (
            ticker,
            as_of_date,
            method,
            json.dumps(inputs or {}, sort_keys=True),
            json.dumps(outputs, sort_keys=True),
            f"{as_of_date}T12:00:00+00:00",
            run_id,
        ),
    )


PRICE_INPUTS = {"current_price": 28.0, "price_as_of_date": D1, "shares": 100.0, "net_debt": 50.0}


def _seed_unblocked_run(conn, ticker, as_of, *, dcf_base=50.0, scorecard_extra=None):
    _insert_valuation(
        conn,
        ticker=ticker,
        as_of_date=as_of,
        method="dcf",
        inputs=PRICE_INPUTS,
        outputs={"status": "OK", "low": 40.0, "base": dcf_base, "high": 60.0, "flags": []},
    )
    _insert_valuation(
        conn,
        ticker=ticker,
        as_of_date=as_of,
        method="epv",
        inputs=PRICE_INPUTS,
        outputs={"status": "OK", "value_per_share": 20.0, "flags": []},
    )
    _insert_valuation(
        conn,
        ticker=ticker,
        as_of_date=as_of,
        method="graham",
        inputs=PRICE_INPUTS,
        outputs={"status": "OK", "value_per_share": 18.0, "flags": []},
    )
    scorecard = {
        "signal": "HOLD",
        "legacy_signal": "FAIRLY_VALUED",
        "type": "STANDARD",
        "pricing_zone": "GROWTH_DEPENDENT",
        "pricing_zone_detail": {
            "current_price": 28.0,
            "epv_adjusted": 20.0,
            "dcf_base": dcf_base,
            "graham_value_per_share": 18.0,
            "gate_action": "PROCEED",
        },
        "quality_context": {"gate_action": "PROCEED"},
    }
    if scorecard_extra:
        scorecard["pricing_zone_detail"].update(scorecard_extra.get("pricing_zone_detail", {}))
        scorecard["quality_context"].update(scorecard_extra.get("quality_context", {}))
    _insert_valuation(
        conn,
        ticker=ticker,
        as_of_date=as_of,
        method="scorecard",
        inputs=PRICE_INPUTS,
        outputs=scorecard,
    )


def _seed_blocked_run(conn, ticker, as_of):
    """The writer's BLOCK path persists ONLY a scorecard row and returns."""
    _insert_valuation(
        conn,
        ticker=ticker,
        as_of_date=as_of,
        method="scorecard",
        inputs={**PRICE_INPUTS, "price_as_of_date": as_of},
        outputs={
            "signal": "VALUATION_BLOCKED",
            "pricing_zone": "VALUATION_BLOCKED",
            "pricing_zone_detail": {"gate_action": "BLOCK", "gate_reason": "going concern"},
            "quality_context": {
                "gate_action": "BLOCK",
                "gate_reason": "going concern",
                "gate_reason_codes": ["GOING_CONCERN"],
            },
        },
    )
    conn.execute(
        "UPDATE valuations SET quality_gate_verdict = 'BLOCK', gate_reason_codes = '[\"GOING_CONCERN\"]'"
        " WHERE ticker = ? AND as_of_date = ? AND method = 'scorecard'",
        (ticker, as_of),
    )


# ── app/web/readmodel/company.py: a BLOCKED run leaves the previous run's cards on the page


def test_company_page_shows_superseded_method_cards_beside_a_blocked_run(
    monkeypatch, tmp_path
):
    """When the newest run is BLOCKED the page must not present the previous
    run's method values as current.

    Mechanism: the writer's BLOCK path persists only a ``scorecard`` row for the
    new date and returns; the previous run's ``dcf``/``epv``/``graham`` rows stay.
    ``company._latest_valuation_rows`` (``as_of_date=None``) ranks
    ``ROW_NUMBER() OVER (PARTITION BY method ORDER BY as_of_date DESC)`` — the
    newest row of EACH method — so ``latest_valuations`` returns the 2026-08-15
    blocked scorecard next to 2026-08-01 dcf/epv/graham cards, and
    ``gauge_shelves`` draws them. The dossier surface gets this right
    (``valuation_render._load_method_payloads``: newest date, then exact date):
    it prints "GATE BLOCKED" and no values.

    Observed: cards dated {2026-08-01 x3, 2026-08-15}, three shelves (DCF
    40/50/60, EPV 20, Graham 18) for a name whose latest verdict is
    VALUATION_BLOCKED. Correct: every card from the newest run's date, no shelves.
    """
    _init_temp_db(monkeypatch, tmp_path)
    with get_db() as conn:
        _seed_unblocked_run(conn, "TEST", D1)
        _seed_blocked_run(conn, "TEST", D2)
        cards = company.latest_valuations(conn, "TEST")
    by_method = {c["method"]: c for c in cards}
    assert by_method["scorecard"]["as_of_date"] == D2
    assert by_method["scorecard"]["extras"]["signal"] == "VALUATION_BLOCKED"
    assert {c["as_of_date"] for c in cards} == {D2}
    assert company.gauge_shelves(cards, anchor_method=None, anchor_value=None) == []


def test_dossier_surface_shows_gate_blocked_for_the_same_rows(monkeypatch, tmp_path):
    """Control: the same two runs through the dossier/analyze block."""
    from app.valuation.valuation_render import render_valuation_decision_block

    _init_temp_db(monkeypatch, tmp_path)
    with get_db() as conn:
        _seed_unblocked_run(conn, "TEST", D1)
        _seed_blocked_run(conn, "TEST", D2)
    block = render_valuation_decision_block("TEST")
    assert "GATE BLOCKED" in block
    assert "$50.00" not in block


# ── durable (spike-corrected) DCF lives only in the scorecard; consumers read the raw row

DURABLE_OVERRIDE = {
    "pricing_zone_detail": {"dcf_base": 30.0, "dcf_raw_base": 50.0, "dcf_durable_base": 30.0},
    "quality_context": {
        "dcf_durable": {"base": 30.0, "status": "OK"},
        "valuation_headwinds": ["DCF_INFLATED_BY_NONRECURRING_REVENUE"],
    },
}


def _seed_spike_name(conn, ticker="TEST"):
    """Writer shape after a revenue spike: the ``dcf`` row is the RAW result
    (``method_rows`` uses ``dcf_result``); the durable base is persisted only as
    ``pricing_zone_detail["dcf_base"]`` (+ ``dcf_raw_base``, ``dcf_durable_base``)
    on the scorecard, which the zone, the watchlist anchor and the backtest use."""
    _seed_unblocked_run(conn, ticker, D1, dcf_base=50.0, scorecard_extra=DURABLE_OVERRIDE)
    conn.execute(
        "UPDATE valuations SET valuation_headwinds = '[\"DCF_INFLATED_BY_NONRECURRING_REVENUE\"]'"
        " WHERE ticker = ? AND method = 'scorecard'",
        (ticker,),
    )
    _insert_valuation(
        conn,
        ticker=ticker,
        as_of_date=D1,
        method="reverse_dcf",
        inputs={**PRICE_INPUTS, "price": 28.0},
        outputs={"status": "OK", "outputs": {"implied_growth": 0.04}},
    )


def _dcf_table_row(text: str) -> str:
    return next(line for line in text.splitlines() if "Discounted Owner Earnings (base)" in line)


def test_dossier_prints_two_different_dcf_values_on_one_page(monkeypatch, tmp_path):
    """One page, one DCF number.

    Mechanism: ``_append_valuation_section_inner`` renders the decision block
    (``build_method_rows`` substitutes ``pzd["dcf_base"]`` and annotates
    "durable (spike-corrected); raw $50.00") and then its own "Intrinsic Value
    Estimates" table from ``dcf.get("base")`` — the raw row — with an empty
    notes cell.

    Observed: "| DCF (anchor) | $30.00 | ... | durable (spike-corrected); raw
    $50.00 |" followed by "| Discounted Owner Earnings (base) | $50.00 | OK |  |".
    Correct: the table row carries $30.00 (the raw $50.00, if shown, labeled as
    rejected).
    """
    from app.valuation.valuation_writer import append_valuation_section

    _init_temp_db(monkeypatch, tmp_path)
    with get_db() as conn:
        _seed_spike_name(conn)
    md = tmp_path / "dossier.md"
    md.write_text("# dossier\n", encoding="utf-8")
    append_valuation_section("TEST", D1, str(md))
    text = md.read_text(encoding="utf-8")
    assert "| DCF **(anchor)** | $30.00 |" in text  # the decision block is right
    assert "$30.00" in _dcf_table_row(text)


def test_web_dcf_card_shows_the_raw_dcf_the_writer_rejected(monkeypatch, tmp_path):
    """The DCF a human sees on the company page must be the DCF the platform
    values the company with.

    Mechanism: ``company._fair_value("dcf", outputs)`` reads ``outputs["base"]``
    of the ``dcf`` row (raw); ``pricing_zone_detail["dcf_base"]`` (durable) is
    never consulted. The watchlist anchor for the same name is the durable value
    (``signal_assembler`` packet.dcf_value = pzd["dcf_base"]).

    Observed: card fair_value {low 40, base 50, high 60}; evolution point 50.0
    for a name the writer flagged DCF_INFLATED_BY_NONRECURRING_REVENUE and
    anchored at 30. Correct: base 30.0 (or the raw value labeled as such).
    """
    _init_temp_db(monkeypatch, tmp_path)
    with get_db() as conn:
        _seed_spike_name(conn)
        cards = company.latest_valuations(conn, "TEST")
    dcf = next(c for c in cards if c["method"] == "dcf")
    # The durable DCF has no low/high scenarios, so it is a line at 30.0
    # (migrated 2026-09-29: as a band without ends the gauge drew it at $0).
    assert dcf["fair_value"] == {"kind": "line", "value": 30.0}


def test_synthesis_dcf_discount_signal_uses_the_raw_dcf(monkeypatch, tmp_path):
    """Memo evidence must be built from the DCF the platform stands behind.

    Mechanism: ``_extract_valuation_signals`` takes ``dcf_base`` from
    ``valuations["dcf"]["outputs"]["base"]`` (raw 50) and the price from the
    reverse-DCF inputs (28): upside 79% > 50% -> DCF_DISCOUNT, HIGH,
    SUPPORTS_UNDERVALUED. Against the durable 30 the upside is 7% — no signal.

    Observed: ("DCF_DISCOUNT", "SUPPORTS_UNDERVALUED", "HIGH", "DCF indicates
    about 79% upside versus the current price anchor."). Correct: no DCF_DISCOUNT.
    """
    from app.synthesis import variant_builder as vb

    _init_temp_db(monkeypatch, tmp_path)
    with get_db() as conn:
        _seed_spike_name(conn)
    payload = vb._load_valuation_payload(ticker="TEST", as_of_date=D1)
    signals, _, _ = vb._extract_valuation_signals(ticker="TEST", valuations=payload)
    assert [s.signal_type for s in signals if s.signal_type.startswith("DCF")] == []


# ── gauge_shelves: the anchor the buy target hangs from is not on the gauge


def test_gauge_drops_the_anchor_when_the_anchor_method_is_a_shelf_method():
    """Docstring contract of ``gauge_shelves``: "the number the buy target hangs
    from is always on the gauge".

    Mechanism: the anchor line is appended only when ``anchor_method`` is NOT
    one of dcf/epv/graham; for a shelf method the function emphasizes the
    card's own value and discards ``anchor_value``. Whenever the watchlist
    anchor differs from the card (durable vs raw DCF, or an anchor recorded
    from an earlier run) the emphasized number is not the anchor.

    Observed: anchor dcf 30.0 -> shelves [DCF 40/50/60 emphasized, EPV 20];
    30.0 appears nowhere; the buy target shown elsewhere is 22.50. Correct: a
    shelf carrying 30.0.
    """
    cards = [
        {"method": "dcf", "fair_value": {"kind": "band", "low": 40.0, "base": 50.0, "high": 60.0}},
        {"method": "epv", "fair_value": {"kind": "line", "value": 20.0}},
    ]
    shelves = company.gauge_shelves(cards, anchor_method="dcf", anchor_value=30.0)
    assert any(30.0 in (s.get("value"), s.get("base")) for s in shelves)


def test_gauge_carries_a_non_shelf_anchor():
    """Control: for a non-shelf anchor method the anchor line IS appended."""
    cards = [{"method": "epv", "fair_value": {"kind": "line", "value": 20.0}}]
    shelves = company.gauge_shelves(cards, anchor_method="ncav", anchor_value=30.0)
    assert [s for s in shelves if s["emphasized"]] == [
        {"method": "ncav", "label": "ncav", "value": 30.0, "emphasized": True}
    ]


# ── app/watchlist/triggers.py: an in-band anchor turns a correct price into PRICE_DATA_SUSPECT

MEDIAN = 10.0  # trailing fiscal-year median price
ANCHOR = 45.0  # 4.5x the median: evaluate_anchor says OK (band is [0.2x, 5.0x])


def _entry(anchor: float):
    from app.watchlist.contract import WatchlistEntry

    return WatchlistEntry(
        ticker="TEST",
        status="ACTIVE",
        conviction_grade="ACTIONABLE",
        confidence="HIGH",
        conviction_source="sector_final_decision",
        valuation_anchor_method="dcf",
        valuation_anchor_value=anchor,
        buy_price_target=anchor * 0.75,
        current_price_at_addition=MEDIAN,
        thesis_text="Deep discount to intrinsic value.",
        source_run_id="run",
        source_sector="industrials",
        added_at="2026-08-28T12:00:00+00:00",
    )


def _run_trigger(cfg, anchor: float):
    """Two consecutive heartbeats with the same quote (a fresh at-target
    crossing needs two heartbeats to confirm); returns both results."""
    from app.market.price_provider import PriceSnapshot
    from app.watchlist.store import add_or_update, get_latest
    from app.watchlist.triggers import check_entry_trigger

    add_or_update(_entry(anchor), db_path=cfg.db_path)
    results = []
    for as_of, checked_at in (
        ("2026-08-31", "2026-09-01T00:00:00+00:00"),
        ("2026-09-01", "2026-09-02T00:00:00+00:00"),
    ):
        entry = get_latest("TEST", db_path=cfg.db_path)
        assert entry is not None
        snapshot = PriceSnapshot(
            ticker="TEST",
            as_of_date=as_of,
            price=MEDIAN,
            retrieved_at=checked_at,
            confidence="HIGH",
        )
        results.append(
            check_entry_trigger(
                entry,
                db_path=cfg.db_path,
                checked_at=checked_at,
                price_lookup=lambda t, s=snapshot: s,
                historical_median_lookup=lambda t: MEDIAN,
                catalyst_lookup=lambda t: "NONE",
            )
        )
    return results


def _init_watchlist_db(monkeypatch, tmp_path):
    from app.watchlist.schema import ensure_watchlist_schema

    cfg = _init_temp_db(monkeypatch, tmp_path)
    ensure_watchlist_schema(cfg.db_path)
    return cfg


def test_anchor_at_exactly_four_times_the_median_deploys():
    """Control: the same quote with an anchor of 40 (4.0x, not > 4.0x) passes
    the price gate — the verdict flips on a one-cent change in the anchor."""
    from app.watchlist.anchor_sanity import evaluate_anchor
    from app.watchlist.triggers import _price_sanity_warning

    assert evaluate_anchor(ANCHOR, MEDIAN).verdict == "OK"
    assert (
        _price_sanity_warning(
            current_price=MEDIAN, historical_median_price=MEDIAN, valuation_anchor_value=40.0
        )
        is None
    )


def test_in_band_anchor_labels_a_price_at_the_median_suspect(monkeypatch, tmp_path):
    """A price equal to the trailing multi-year median is never suspect data.

    Mechanism: ``check_entry_trigger`` first runs ``evaluate_anchor`` (OK for
    4.5x) and, per its own comment, falls "through to the normal price logic"
    for an in-band anchor. That logic (``_price_sanity_reference``) replaces
    the median with the anchor whenever anchor/median > 4.0, and
    ``_price_sanity_warning`` then flags price < 0.25 x anchor. For any anchor in
    (4.0x, 5.0x] the row needs a price >= 1.0x-1.25x the median to escape while
    its buy target is 3.0x-3.75x the median. The status_reason blames the
    latest price; the row leaves the default review queue
    (``store.watchlist_queue``: ``w.status != 'PRICE_DATA_SUSPECT'``) and every
    heartbeat re-flags it.

    Observed: "PRICE_DATA_SUSPECT:latest_price=10.00:valuation_anchor_scale_check
    =45.00:historical_median=10.00:threshold=below_0.25x"; new_status
    PRICE_DATA_SUSPECT on both heartbeats; default queue empty. Correct: no
    price warning; the price gate runs (10 <= 33.75 -> pending, then
    DEPLOY_READY on the second heartbeat, as the anchor-40 control shows) — or,
    if the two gates are reconciled the other way, an anchor QUARANTINE that
    blames the anchor. Either way the price is not labeled suspect.
    """
    from app.watchlist.store import watchlist_queue
    from app.watchlist.triggers import _price_sanity_warning

    cfg = _init_watchlist_db(monkeypatch, tmp_path)
    first, second = _run_trigger(cfg, ANCHOR)
    assert first.new_status != "PRICE_DATA_SUSPECT", first.warning
    assert second.new_status != "PRICE_DATA_SUSPECT", second.warning
    assert [r["ticker"] for r in watchlist_queue(db_path=cfg.db_path)] == ["TEST"]
    assert (
        _price_sanity_warning(
            current_price=MEDIAN, historical_median_price=MEDIAN, valuation_anchor_value=ANCHOR
        )
        is None
    )


def test_control_anchor_at_four_times_the_median_reaches_deploy_ready(monkeypatch, tmp_path):
    """Control (end to end): anchor 40 with the same quote -> pending on the
    first heartbeat, DEPLOY_READY on the second."""
    cfg = _init_watchlist_db(monkeypatch, tmp_path)
    first, second = _run_trigger(cfg, 40.0)
    assert first.transition == "at-target-pending-confirmation"
    assert second.new_status == "DEPLOY_READY"


# ── app/watchlist/store.py: an LLM-supplied buy target is never sanity-checked

FY_MEDIAN = 100.0  # trusted trailing fiscal-year median price (stubbed provider)


def _seed_fy_row(cfg):
    """One FY row so ``_historical_fiscal_year_median_price`` has a period_end."""
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO companyfacts_facts (ticker, fiscal_year, period_type, period_end,
                line_item, value, units, source_url, fetched_at, filed_date)
            VALUES ('TEST', 2024, 'FY', '2024-12-31', 'Revenues', 1000.0, 'USD',
                'test://companyfacts', '2026-05-31T00:00:00+00:00', '2025-02-20')
            """
        )


def _stub_fy_median(monkeypatch):
    from app.market.price_provider import PriceSnapshot

    class _Provider:
        provider_name = "stub"

        def get_price_asof(self, ticker, as_of_date):
            return PriceSnapshot(
                ticker=ticker, as_of_date=as_of_date, price=FY_MEDIAN, source="stub"
            )

    monkeypatch.setattr("app.watchlist.triggers.build_price_provider", lambda *a, **k: _Provider())
    monkeypatch.setattr("app.watchlist.store.artifact_decision_eligibility", lambda payload: "PASS")


def _intake(monkeypatch, tmp_path, valuation: dict, *, current_price: float):
    """Run one candidate through ``_entry_from_candidate`` (the add-time path)."""
    from app.autonomous.sector_contract import (
        AutonomousSectorFinancialRunArtifact,
        SectorCompanyFinancialPacket,
    )
    from app.watchlist.store import _entry_from_candidate

    cfg = _init_watchlist_db(monkeypatch, tmp_path)
    _seed_fy_row(cfg)
    _stub_fy_median(monkeypatch)
    packet = SectorCompanyFinancialPacket(
        ticker="TEST",
        financial_status="OK",
        model_fit_status="OK",
        data_quality_status="OK",
        current_price=current_price,
        valuation=valuation,
    )
    ranking = {
        "ticker": "TEST",
        "company_autonomy_verdict": "ACTIONABLE",
        "company_autonomy_confidence": "HIGH",
        "positioning_summary": "fixture row.",
    }
    artifact = AutonomousSectorFinancialRunArtifact(
        run_id="run",
        sector="test_sector",
        market_cap_focus="mid_cap",
        objective="consumer-layer fixture",
        as_of_date="2026-05-31",
        created_at="2026-05-31T12:00:00Z",
        completed_at="2026-05-31T12:01:00Z",
        status="COMPLETED",
        final_verdict="WATCHLIST",
        selected_ticker=None,
        confidence="HIGH",
        candidate_selection={"selected_tickers": ["TEST"]},
        company_packets=[packet],
        relative_ranking=[ranking],
        memo_body={"candidates": {}},
    )
    entry = _entry_from_candidate(
        artifact,
        ticker="TEST",
        packet=packet,
        ranking=ranking,
        added_at="2026-05-31T12:00:00+00:00",
        db_path=cfg.db_path,
    )
    assert entry is not None
    return entry


def test_a_seventy_x_anchor_is_quarantined_at_add(monkeypatch, tmp_path):
    """Control: the same magnitude slip in the ANCHOR is caught by the band."""
    entry = _intake(
        monkeypatch,
        tmp_path,
        {"anchor_method": "DCF", "valuation_anchor": 70.0 * FY_MEDIAN},
        current_price=60.0,
    )
    assert entry.status == "QUARANTINE"
    assert (
        entry.status_reason == "QUARANTINE_ANCHOR_ABOVE_BAND:anchor=7000.00:ref=100.00:ratio=70.0"
    )


def test_computed_buy_target_is_a_discount_to_the_anchor(monkeypatch, tmp_path):
    """Control: without an explicit target the per-name discount applies (100 -> 85)."""
    entry = _intake(
        monkeypatch,
        tmp_path,
        {"anchor_method": "DCF", "valuation_anchor": 100.0},
        current_price=60.0,
    )
    assert entry.status == "DEPLOY_READY"
    assert entry.buy_price_target == 85.0


def test_llm_buy_target_seventy_times_the_reference_is_deploy_ready(monkeypatch, tmp_path):
    """The number the trigger fires on must pass the same sanity band as the
    number it was supposedly derived from.

    Mechanism: ``_buy_price_target`` returns the first positive
    ``buy_below_price`` / ``buy_price_target`` / ``buy_below`` in
    ``packet.valuation`` before any computation ("explicit LLM-supplied targets
    always win"). ``_entry_from_candidate`` then runs ``evaluate_anchor`` on
    ``valuation_anchor`` only and sets DEPLOY_READY on ``price <= target``. The
    price-anchor band exists "so a magnitude/units artifact is never shipped as a buy
    target" — and the one field that IS the buy target bypasses it. The same
    holds with no anchor at all (nothing to assess, still DEPLOY_READY).

    Observed (anchor 100, FY median 100, price 60, buy_price_target 7000):
    status DEPLOY_READY, buy_price_target 7000.0, status_reason None. Correct:
    not DEPLOY_READY — quarantined like the anchor case, or the explicit target
    discarded for the computed 85.
    """
    entry = _intake(
        monkeypatch,
        tmp_path,
        {"anchor_method": "DCF", "valuation_anchor": 100.0, "buy_price_target": 7000.0},
        current_price=60.0,
    )
    assert entry.status != "DEPLOY_READY"
    assert entry.status == "QUARANTINE" or entry.buy_price_target != 7000.0


def test_llm_buy_target_above_the_anchor_deploys_above_intrinsic_value(
    monkeypatch, tmp_path
):
    """A buy target must not exceed the anchor it is a discount to.

    Same mechanism as the 70x case, at the other end: the per-name margin of
    safety (``compute_buy_target``) is a 12-40% DISCOUNT to the anchor, and an
    explicit ``buy_below_price`` of 1.2x the anchor inverts it.

    Observed (anchor 100, price 110, buy_below_price 120): DEPLOY_READY with
    buy_price_target 120.0 — a "buy now" at 10% above intrinsic value. Correct:
    not DEPLOY_READY (target capped at the anchor, or the explicit target
    rejected).
    """
    entry = _intake(
        monkeypatch,
        tmp_path,
        {"anchor_method": "DCF", "valuation_anchor": 100.0, "buy_below_price": 120.0},
        current_price=110.0,
    )
    assert entry.status != "DEPLOY_READY"


def test_an_explicit_buy_target_cannot_loosen_the_names_own_margin_of_safety(
    monkeypatch, tmp_path
):
    """An explicit target just under the anchor still needs the name's discount.

    Anchor 100, explicit buy_below_price 99, price 98. Capping the explicit target
    only at the anchor made this DEPLOY_READY at a 1% discount to intrinsic value.
    The computed margin of safety never goes below a 10% discount
    (compute_buy_discount clamps to [0.10, 0.45]), so the target is at most 90 and
    a price of 98 is not a buy.
    """
    entry = _intake(
        monkeypatch,
        tmp_path,
        {"anchor_method": "DCF", "valuation_anchor": 100.0, "buy_below_price": 99.0},
        current_price=98.0,
    )
    assert entry.buy_price_target is not None
    assert entry.buy_price_target <= 90.0
    assert entry.status != "DEPLOY_READY"


def test_a_stricter_explicit_buy_target_is_kept(monkeypatch, tmp_path):
    """Control: an explicit 50 against anchor 100 is stricter than any computed target."""
    entry = _intake(
        monkeypatch,
        tmp_path,
        {"anchor_method": "DCF", "valuation_anchor": 100.0, "buy_below_price": 50.0},
        current_price=98.0,
    )
    assert entry.buy_price_target == 50.0
