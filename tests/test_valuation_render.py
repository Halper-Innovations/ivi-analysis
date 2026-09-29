"""Tests for the decision-useful valuation output block (goal task OUTPUT)."""

from __future__ import annotations

from app.db import get_db, init_db, utc_now_iso


def _init_cfg(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(data_dir / "engine.db"))
    monkeypatch.setenv("VOE_LLM_PROVIDER", "disabled")
    from app.config import get_config

    get_config.cache_clear()
    cfg = get_config()
    init_db(cfg)
    return cfg


_FIELDS = {
    "revenue": [900.0, 950.0, 1000.0],
    "operating_income": [180.0, 190.0, 200.0],
    "net_income": [120.0, 130.0, 140.0],
    "equity": [500.0, 550.0, 600.0],
    "cfo": [170.0, 185.0, 200.0],
    "capex": [30.0, 32.0, 35.0],
    "shares_outstanding": [100.0, 100.0, 100.0],
    "total_debt": [200.0, 200.0, 200.0],
    "cash": [80.0, 90.0, 100.0],
    "preferred_equity": [0.0, 0.0, 0.0],
    "noncontrolling_interest": [0.0, 0.0, 0.0],
    "goodwill": [90.0, 95.0, 100.0],
    "intangible_assets": [40.0, 45.0, 50.0],
}


def _seed_and_value(ticker, monkeypatch, tmp_path, fields=None, price=None):
    _init_cfg(monkeypatch, tmp_path)
    with get_db() as conn:
        for line_item, values in (fields or _FIELDS).items():
            for year, value in zip([2021, 2022, 2023], values, strict=True):
                conn.execute(
                    "INSERT INTO companyfacts_facts "
                    "(ticker, fiscal_year, period_type, period_end, line_item, "
                    "value, units, source_url, fetched_at, filed_date, accession) "
                    "VALUES (?, ?, 'FY', ?, ?, ?, 'USD', ?, ?, ?, ?)",
                    (
                        ticker,
                        year,
                        f"{year}-12-31",
                        line_item,
                        float(value),
                        f"https://example.test/companyfacts/{ticker}",
                        utc_now_iso(),
                        f"{year + 1}-02-15",
                        f"{ticker}-{year}",
                    ),
                )
        conn.commit()
    from app.valuation.valuation_writer import ensure_valuation

    ensure_valuation(ticker, "2024-04-02", provider=None, force_refresh=True, price_override=price)


def test_decision_block_renders_table_provenance_and_conventions(monkeypatch, tmp_path):
    _seed_and_value("RNDR", monkeypatch, tmp_path, price=12.0)
    from app.valuation.valuation_render import render_valuation_decision_block

    block = render_valuation_decision_block("RNDR", "2024-04-02")
    # Anchor provenance: method + reason rendered
    assert "**Anchor:**" in block
    assert "MAX_POSITIVE_DCF_EPV" in block or "SECTOR_SPECIFIC" in block
    # Per-method table includes core methods AND the lenses
    for label in (
        "| DCF",
        "| EPV",
        "| Graham",
        "| NCAV",
        "| EV/EBIT lens",
        "| FCF-yield lens",
        "| Tangible floor",
    ):
        assert label in block, f"missing {label}"
    # Every MoS number is convention-labeled
    assert "textbook: (anchor − price) / anchor" in block
    assert "Margin of safety" in block
    # The anchor row is marked
    assert "**(anchor)**" in block


def test_decision_block_blocked_names_show_gate_and_unblock(monkeypatch, tmp_path):
    fields = dict(_FIELDS)
    # Genuine secular decline >30% with declining latest year -> BLOCK
    fields["revenue"] = [1000.0, 850.0, 550.0]
    fields["net_income"] = [-10.0, -20.0, -30.0]
    _seed_and_value("BLKD", monkeypatch, tmp_path, fields=fields)
    from app.valuation.valuation_render import render_valuation_decision_block

    block = render_valuation_decision_block("BLKD", "2024-04-02")
    assert "GATE BLOCKED" in block
    assert "SEVERE_SECULAR_DECLINE" in block
    assert "would unblock when" in block


def test_decision_block_empty_for_unknown_ticker(monkeypatch, tmp_path):
    _init_cfg(monkeypatch, tmp_path)
    from app.valuation.valuation_render import render_valuation_decision_block

    assert render_valuation_decision_block("NOPE", "2024-04-02") == ""


def test_decision_block_durable_beside_raw(monkeypatch, tmp_path):
    fields = dict(_FIELDS)
    fields["revenue"] = [900.0, 950.0, 2000.0]  # latest-year spike
    _seed_and_value("SPKR", monkeypatch, tmp_path, fields=fields, price=10.0)
    from app.valuation.valuation_render import render_valuation_decision_block

    block = render_valuation_decision_block("SPKR", "2024-04-02")
    assert "durable (spike-corrected); raw $" in block


def test_decision_block_anomaly_zone_suppresses_anchor(monkeypatch, tmp_path):
    """Review RENDER-ANOMALY-ANCHOR (+ GZ-1 dup): VALUATION_ANOMALY names must
    not get an Anchor/buy-at/MoS header — production (signal_assembler) nulls
    these and the backtest excludes them; the render block was the one output
    surface still issuing buy guidance on the suppressed subpopulation."""
    fields = dict(_FIELDS)
    # Negative operating income every year -> EPV negative; CFO positive ->
    # DCF positive: the anomaly shape whose surviving method leaked as anchor.
    fields["operating_income"] = [-50.0, -60.0, -70.0]
    _seed_and_value("ANOM", monkeypatch, tmp_path, fields=fields, price=6.0)
    from app.valuation.valuation_render import render_valuation_decision_block

    block = render_valuation_decision_block("ANOM", "2024-04-02")
    assert "VALUATION_ANOMALY" in block
    assert "buy at <=" not in block
    assert "Margin of safety" not in block
    assert "**(anchor)**" not in block
    assert "production suppresses anchors" in block
    # Per-method provenance table still renders (values stay visible)
    assert "| DCF" in block
    assert "| EPV" in block


def test_negative_epv_is_shown_as_a_signed_non_price_with_its_status(monkeypatch, tmp_path):
    """A negative EPV printed as "$-6.74" with no
    status in the decision block, and the dossier's EPV row left its notes cell
    empty while every sibling row printed its flags. A negative earnings-power
    value is not a price: it is written -$6.74 with its status beside it, and
    the dossier row carries the method's flags.

    2026-09-29 (EPV fix): with no earnings power the EPV publishes no
    per-share value at all, only its EPV_NEGATIVE status and flags, so the
    row reads "— (EPV_NEGATIVE)" — still never a price."""
    fields = dict(_FIELDS)
    fields["operating_income"] = [-50.0, -60.0, -70.0]
    _seed_and_value("ANOM", monkeypatch, tmp_path, fields=fields, price=6.0)
    from app.valuation.valuation_render import render_valuation_decision_block
    from app.valuation.valuation_writer import _append_valuation_section_inner

    block = render_valuation_decision_block("ANOM", "2024-04-02")
    dossier = tmp_path / "dossier.md"
    dossier.write_text("", encoding="utf-8")
    _append_valuation_section_inner("ANOM", "2024-04-02", str(dossier))
    text = dossier.read_text(encoding="utf-8")

    assert "| EPV | — (EPV_NEGATIVE) |" in block
    assert "$-" not in block
    assert (
        "| Earnings Power Value | N/A | EPV_NEGATIVE | "
        "EPV_TAX_RATE_STATUTORY_DEFAULT, EPV_NO_TAX_SHIELD_ON_LOSSES, EPV_NO_EARNINGS_POWER, "
        # No SIC on file in the fixture: REIT status is unknown and said so (2026-09-29).
        "REIT_STATUS_UNKNOWN |"
    ) in text
    assert "$-" not in text
