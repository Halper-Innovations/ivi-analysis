from __future__ import annotations

import sqlite3
from pathlib import Path

from app.db import init_db
from app.market.price_provider import PriceSnapshot
from app.watchlist.contract import WatchlistEntry
from app.watchlist.store import (
    add_or_update,
    add_price_snapshot,
    get_history,
    get_latest,
    get_latest_price,
)
from app.watchlist.triggers import check_entry_trigger, check_watchlist_triggers


def _init_temp_db(monkeypatch, tmp_path) -> Path:
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    from app.config import get_config

    get_config.cache_clear()
    init_db()
    return db_path


def _entry(
    db_path: Path,
    *,
    ticker: str = "AAA",
    status: str = "ACTIVE",
    buy_price_target: float = 75.0,
    valuation_anchor_value: float = 100.0,
    source_run_id: str = "sector_run_1",
) -> WatchlistEntry:
    add_or_update(
        WatchlistEntry(
            ticker=ticker,
            status=status,
            conviction_grade="WATCHLIST_ONLY",
            valuation_anchor_method="DCF",
            valuation_anchor_value=valuation_anchor_value,
            buy_price_target=buy_price_target,
            current_price_at_addition=90.0,
            thesis_text="AAA is good if margins stabilize.",
            key_risks=["Gross margin compression"],
            falsifiers=["Revenue decline accelerates"],
            open_questions=["Can working capital normalize?"],
            source_run_id=source_run_id,
            source_sector="industrial_tech",
            added_at="2026-05-01T12:00:00+00:00",
        ),
        db_path=db_path,
    )
    latest = get_latest(ticker, db_path=db_path)
    assert latest is not None
    return latest


def _price(ticker: str = "AAA", price: float = 70.0, source: str = "yahoo") -> PriceSnapshot:
    return PriceSnapshot(
        ticker=ticker,
        as_of_date="2026-05-08",
        price=price,
        currency="USD",
        source=source,
        retrieved_at="2026-05-10T12:00:00+00:00",
        confidence="HIGH",
    )


def _snapshot_count(db_path: Path) -> int:
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute("SELECT COUNT(*) FROM watchlist_price_snapshots").fetchone()
        return int(row[0])
    finally:
        conn.close()


def test_check_entry_flips_active_to_deploy_ready_and_logs_snapshot(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    entry = _entry(db_path, status="ACTIVE", buy_price_target=75.0)
    # A fresh at-target crossing flips only after two consecutive
    # heartbeats — seed the prior confirming snapshot.
    add_price_snapshot(
        int(entry.id or 0),
        price=70.0,
        checked_at="2026-05-09T12:00:00+00:00",
        source="yahoo",
        db_path=db_path,
    )

    result = check_entry_trigger(
        entry,
        db_path=db_path,
        checked_at="2026-05-10T12:00:00+00:00",
        price_lookup=lambda ticker: _price(ticker, 70.0, "yahoo"),
    )
    latest = get_latest("AAA", db_path=db_path)
    history = get_history("AAA", db_path=db_path)
    latest_snapshot = get_latest_price(entry.id or 0, db_path=db_path)

    assert result.prior_status == "ACTIVE"
    assert result.new_status == "DEPLOY_READY"
    assert result.latest_price == 70.0
    assert result.transition == "crossed below buy-target"
    assert latest is not None
    assert latest.status == "DEPLOY_READY"
    assert latest.status_reason == "price $70.00 crossed below buy-target $75.00"
    assert history[-2]["field_name"] == "status"
    assert history[-2]["old_value"] == "ACTIVE"
    assert history[-2]["new_value"] == "DEPLOY_READY"
    assert history[-2]["source"] == "trigger"
    assert latest_snapshot == {
        "id": 2,
        "watchlist_id": entry.id,
        "price": 70.0,
        "checked_at": "2026-05-10T12:00:00+00:00",
        "source": "yahoo",
    }


def test_check_entry_flips_deploy_ready_to_active(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    entry = _entry(db_path, status="DEPLOY_READY", buy_price_target=75.0)

    result = check_entry_trigger(
        entry,
        db_path=db_path,
        checked_at="2026-05-10T12:00:00+00:00",
        price_lookup=lambda ticker: _price(ticker, 80.0, "stooq"),
    )
    latest = get_latest("AAA", db_path=db_path)

    assert result.new_status == "ACTIVE"
    assert result.transition == "rose above buy-target"
    assert result.source == "stooq"
    assert latest is not None
    assert latest.status == "ACTIVE"
    assert latest.status_reason == "price $80.00 rose above buy-target $75.00"


def test_no_status_change_updates_snapshot_without_history(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    entry = _entry(db_path, status="ACTIVE", buy_price_target=75.0)
    history_before = get_history("AAA", db_path=db_path)

    result = check_entry_trigger(
        entry,
        db_path=db_path,
        checked_at="2026-05-10T12:00:00+00:00",
        price_lookup=lambda ticker: _price(ticker, 80.0, "yahoo"),
    )
    latest = get_latest("AAA", db_path=db_path)
    history_after = get_history("AAA", db_path=db_path)
    latest_snapshot = get_latest_price(entry.id or 0, db_path=db_path)

    assert result.new_status == "ACTIVE"
    assert result.transition == "none"
    assert latest is not None
    assert latest.status == "ACTIVE"
    assert latest.status_reason is None
    assert len(history_after) == len(history_before)
    assert latest_snapshot is not None
    assert latest_snapshot["price"] == 80.0
    assert latest_snapshot["source"] == "yahoo"


def test_price_unavailable_skips_without_mutation(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    entry = _entry(db_path, status="ACTIVE", buy_price_target=75.0)
    history_before = get_history("AAA", db_path=db_path)

    result = check_entry_trigger(
        entry,
        db_path=db_path,
        price_lookup=lambda ticker: None,
    )
    latest = get_latest("AAA", db_path=db_path)

    assert result.warning == "PRICE_UNAVAILABLE"
    assert result.new_status == "ACTIVE"
    assert latest is not None
    assert latest.status == "ACTIVE"
    assert len(get_history("AAA", db_path=db_path)) == len(history_before)
    assert _snapshot_count(db_path) == 0


def test_suspect_price_is_flagged_without_snapshot_or_trigger_transition(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    entry = _entry(
        db_path,
        ticker="BKNG",
        status="DEPLOY_READY",
        buy_price_target=3250.15,
        valuation_anchor_value=3000.0,
    )

    result = check_entry_trigger(
        entry,
        db_path=db_path,
        checked_at="2026-05-10T12:00:00+00:00",
        price_lookup=lambda ticker: _price(ticker, 154.13, "yahoo"),
        historical_median_lookup=lambda ticker: 3000.0,
    )
    latest = get_latest("BKNG", db_path=db_path)

    assert result.warning == "PRICE_DATA_SUSPECT"
    assert result.prior_status == "DEPLOY_READY"
    assert result.new_status == "PRICE_DATA_SUSPECT"
    assert result.latest_price == 154.13
    assert result.transition == "price-data-suspect"
    assert latest is not None
    assert latest.status == "PRICE_DATA_SUSPECT"
    assert latest.status_reason == (
        "PRICE_DATA_SUSPECT:latest_price=154.13:"
        "historical_fiscal_year_median=3000.00:historical_median=3000.00:"
        "threshold=below_0.25x"
    )
    assert _snapshot_count(db_path) == 0


def test_normal_price_movement_passes_sanity_check(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    entry = _entry(db_path, status="ACTIVE", buy_price_target=75.0)

    result = check_entry_trigger(
        entry,
        db_path=db_path,
        checked_at="2026-05-10T12:00:00+00:00",
        price_lookup=lambda ticker: _price(ticker, 90.0, "yahoo"),
        historical_median_lookup=lambda ticker: 100.0,
    )

    assert result.warning is None
    assert result.new_status == "ACTIVE"
    assert _snapshot_count(db_path) == 1


def test_genuine_half_drawdown_passes_sanity_check(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    entry = _entry(db_path, status="ACTIVE", buy_price_target=45.0)

    result = check_entry_trigger(
        entry,
        db_path=db_path,
        checked_at="2026-05-10T12:00:00+00:00",
        price_lookup=lambda ticker: _price(ticker, 50.0, "yahoo"),
        historical_median_lookup=lambda ticker: 100.0,
    )

    assert result.warning is None
    assert result.new_status == "ACTIVE"
    assert _snapshot_count(db_path) == 1


def test_genuine_double_passes_sanity_check(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    entry = _entry(db_path, status="ACTIVE", buy_price_target=175.0)

    result = check_entry_trigger(
        entry,
        db_path=db_path,
        checked_at="2026-05-10T12:00:00+00:00",
        price_lookup=lambda ticker: _price(ticker, 200.0, "yahoo"),
        historical_median_lookup=lambda ticker: 100.0,
    )

    assert result.warning is None
    assert result.new_status == "ACTIVE"
    assert _snapshot_count(db_path) == 1


def test_dry_run_reports_transition_without_mutation(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    entry = _entry(db_path, status="ACTIVE", buy_price_target=75.0)
    add_price_snapshot(
        int(entry.id or 0),
        price=70.0,
        checked_at="2026-05-09T12:00:00+00:00",
        source="yahoo",
        db_path=db_path,
    )

    result = check_entry_trigger(
        entry,
        db_path=db_path,
        dry_run=True,
        checked_at="2026-05-10T12:00:00+00:00",
        price_lookup=lambda ticker: _price(ticker, 70.0, "yahoo"),
    )
    latest = get_latest("AAA", db_path=db_path)

    assert result.new_status == "DEPLOY_READY"
    assert result.dry_run is True
    assert result.mutated is False
    assert latest is not None
    assert latest.status == "ACTIVE"
    assert _snapshot_count(db_path) == 1


def test_contradicted_entry_skipped_without_force(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    _entry(db_path, status="CONTRADICTED", buy_price_target=75.0)

    result = check_watchlist_triggers("AAA", db_path=db_path)[0]
    latest = get_latest("AAA", db_path=db_path)

    assert result.warning == "STATUS_SKIPPED:CONTRADICTED"
    assert result.latest_price is None
    assert latest is not None
    assert latest.status == "CONTRADICTED"
    assert _snapshot_count(db_path) == 0


def test_force_allows_contradicted_entry_check(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    entry = _entry(db_path, status="CONTRADICTED", buy_price_target=75.0)
    add_price_snapshot(
        int(entry.id or 0),
        price=70.0,
        checked_at="2026-05-09T12:00:00+00:00",
        source="yahoo",
        db_path=db_path,
    )

    result = check_entry_trigger(
        entry,
        db_path=db_path,
        force=True,
        checked_at="2026-05-10T12:00:00+00:00",
        price_lookup=lambda ticker: _price(ticker, 70.0, "yahoo"),
    )
    latest = get_latest("AAA", db_path=db_path)

    assert result.prior_status == "CONTRADICTED"
    assert result.new_status == "DEPLOY_READY"
    assert latest is not None
    assert latest.status == "DEPLOY_READY"
    assert _snapshot_count(db_path) == 2


def test_post_asof_companyfacts_cannot_clear_anchor_quarantine(
    monkeypatch,
    tmp_path,
):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    entry = _entry(
        db_path,
        status="QUARANTINE",
        buy_price_target=20.0,
        valuation_anchor_value=100.0,
    )
    add_price_snapshot(
        int(entry.id or 0),
        price=15.0,
        checked_at="2026-05-07T12:00:00+00:00",
        source="yahoo",
        db_path=db_path,
    )
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            INSERT INTO companyfacts_facts(
                ticker, fiscal_year, period_type, period_end, line_item,
                value, units, source_url, fetched_at, filed_date
            )
            VALUES(
                'AAA', 2024, 'FY', '2024-12-31', 'revenue',
                100.0, 'USD_millions', 'https://example.test/companyfacts',
                '2026-05-09T12:00:00Z', '2026-05-09'
            )
            """
        )
        conn.commit()
    finally:
        conn.close()

    class FiscalPriceProvider:
        def __init__(self) -> None:
            self.calls = 0

        def get_price_asof(self, ticker: str, as_of_date: str):
            self.calls += 1
            return PriceSnapshot(
                ticker=ticker,
                as_of_date=as_of_date,
                price=60.0,
                currency="USD",
                source="fixture",
                retrieved_at="2026-05-10T12:00:00+00:00",
                confidence="HIGH",
            )

    provider = FiscalPriceProvider()
    monkeypatch.setattr(
        "app.watchlist.triggers.build_price_provider",
        lambda **_kwargs: provider,
    )

    result = check_entry_trigger(
        entry,
        db_path=db_path,
        force=True,
        checked_at="2026-05-10T12:00:00+00:00",
        price_lookup=lambda ticker: PriceSnapshot(
            ticker=ticker,
            as_of_date="2026-05-08",
            price=15.0,
            currency="USD",
            source="yahoo",
            retrieved_at="2026-05-10T12:00:00+00:00",
            confidence="HIGH",
        ),
    )

    assert provider.calls == 0
    assert result.new_status == "QUARANTINE"
    assert result.transition == "anchor-quarantine"
    assert result.warning == ("QUARANTINE_ANCHOR_ABOVE_BAND:anchor=100.00:ref=15.00:ratio=6.667")


def test_fiscal_year_end_dates_require_filed_date_and_source(
    monkeypatch,
    tmp_path,
):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executemany(
            """
            INSERT INTO companyfacts_facts(
                ticker, fiscal_year, period_type, period_end, line_item,
                value, units, source_url, fetched_at, filed_date
            )
            VALUES('AAA', ?, 'FY', ?, ?, 100.0, 'USD_millions', ?, ?, ?)
            """,
            [
                (
                    2024,
                    "2024-12-31",
                    "revenue",
                    "",
                    "2025-02-15T12:00:00Z",
                    "2025-02-15",
                ),
                (
                    2023,
                    "2023-12-31",
                    "operating_income",
                    "https://example.test/companyfacts",
                    "2024-02-15T12:00:00Z",
                    None,
                ),
            ],
        )
        conn.commit()
    finally:
        conn.close()
    from app.watchlist.triggers import _companyfacts_fiscal_year_end_dates

    assert (
        _companyfacts_fiscal_year_end_dates(
            "AAA",
            db_path=db_path,
            as_of_date="2026-05-08",
        )
        == []
    )


# --- gate precedence: the promises the comment above the anchor check makes ------------
# (The comment once described a trigger precedence the code did not implement. With the
# price-sanity reference fixed to the traded median, code and comment agree; these pin the
# two promises so they cannot drift again.)


def test_out_of_band_anchor_is_quarantined_before_the_price_is_called_suspect(
    monkeypatch, tmp_path
):
    """Anchor 6.7x the median is the ANCHOR's fault: the row is QUARANTINEd with a reason that
    blames the anchor, never PRICE_DATA_SUSPECT blaming the quote."""
    db_path = _init_temp_db(monkeypatch, tmp_path)
    entry = _entry(db_path, status="ACTIVE", buy_price_target=75.0, valuation_anchor_value=100.0)

    result = check_entry_trigger(
        entry,
        db_path=db_path,
        checked_at="2026-05-10T12:00:00+00:00",
        price_lookup=lambda ticker: _price(ticker, 15.0, "yahoo"),
        historical_median_lookup=lambda ticker: 15.0,
        catalyst_lookup=lambda ticker: "NONE",
    )

    assert result.new_status == "QUARANTINE"
    assert result.transition == "anchor-quarantine"
    assert result.warning == "QUARANTINE_ANCHOR_ABOVE_BAND:anchor=100.00:ref=15.00:ratio=6.667"


def test_quarantined_row_lifts_through_the_normal_price_logic_once_the_anchor_is_in_band(
    monkeypatch, tmp_path
):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    entry = _entry(db_path, status="QUARANTINE", buy_price_target=30.0, valuation_anchor_value=40.0)
    add_price_snapshot(
        int(entry.id or 0),
        price=10.0,
        checked_at="2026-05-09T12:00:00+00:00",
        source="yahoo",
        db_path=db_path,
    )

    result = check_entry_trigger(
        entry,
        db_path=db_path,
        force=True,
        checked_at="2026-05-10T12:00:00+00:00",
        price_lookup=lambda ticker: _price(ticker, 10.0, "yahoo"),
        historical_median_lookup=lambda ticker: 10.0,
        catalyst_lookup=lambda ticker: "NONE",
    )

    assert result.prior_status == "QUARANTINE"
    assert result.new_status == "DEPLOY_READY"
    assert get_latest("AAA", db_path=db_path).status == "DEPLOY_READY"
