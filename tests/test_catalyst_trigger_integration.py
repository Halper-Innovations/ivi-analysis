"""Integration tests for the catalyst overlay wired into the price trigger.

A CONFIRMED catalyst upgrades ONLY the price-trigger axis
(DEPLOY_READY -> BUY_CONFIRMED). It never changes the conviction grade in v1, and
the price gate is necessary: an above-target name can never reach BUY_CONFIRMED
no matter the catalyst, and the catalyst lookup is not even invoked above target
(so the Form-4 fetch is skipped for names that are not within reach).
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from app.market.price_provider import PriceSnapshot
from app.watchlist.contract import WATCHLIST_STATUSES, WatchlistEntry
from app.watchlist.digest import render_digest
from app.watchlist.store import add_or_update, get_latest, record_trigger_status_change
from app.watchlist.triggers import check_entry_trigger


def _init_temp_db(monkeypatch, tmp_path) -> Path:
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    from app.config import get_config

    get_config.cache_clear()
    from app.db import init_db

    init_db()
    return db_path


def _entry(
    db_path: Path,
    *,
    ticker: str = "AAA",
    status: str = "ACTIVE",
    buy_price_target: float = 10.0,
    valuation_anchor_value: float = 12.0,
) -> WatchlistEntry:
    add_or_update(
        WatchlistEntry(
            ticker=ticker,
            status=status,
            conviction_grade="WATCHLIST_ONLY",
            valuation_anchor_method="DCF",
            valuation_anchor_value=valuation_anchor_value,
            buy_price_target=buy_price_target,
            current_price_at_addition=11.0,
            thesis_text="AAA is cheap if margins hold.",
            key_risks=["Margin compression"],
            falsifiers=["Revenue decline accelerates"],
            open_questions=["Can FCF normalize?"],
            source_run_id="sector_run_1",
            source_sector="industrial_tech",
            added_at="2026-05-01T12:00:00+00:00",
        ),
        db_path=db_path,
    )
    latest = get_latest(ticker, db_path=db_path)
    assert latest is not None
    return latest


def _price(ticker: str, price: float) -> PriceSnapshot:
    return PriceSnapshot(
        ticker=ticker,
        as_of_date="2026-05-08",
        price=price,
        currency="USD",
        source="yahoo",
        retrieved_at="2026-05-10T12:00:00+00:00",
        confidence="HIGH",
    )


def _seed_confirming_snapshot(db_path, entry) -> None:
    """Seed the prior at-target heartbeat so a fresh crossing confirms."""
    from app.watchlist.store import add_price_snapshot

    add_price_snapshot(
        int(entry.id or 0),
        price=9.0,
        checked_at="2026-05-09T12:00:00+00:00",
        source="yahoo",
        db_path=db_path,
    )


def _raises_if_called(ticker: str) -> str:
    raise AssertionError(f"catalyst_lookup must not be invoked above buy-target ({ticker})")


def test_below_target_with_confirmed_catalyst_becomes_buy_confirmed(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    entry = _entry(db_path, buy_price_target=10.0)
    _seed_confirming_snapshot(db_path, entry)

    result = check_entry_trigger(
        entry,
        db_path=db_path,
        checked_at="2026-05-10T12:00:00+00:00",
        price_lookup=lambda ticker: _price(ticker, 9.0),
        historical_median_lookup=lambda ticker: 11.0,
        catalyst_lookup=lambda ticker: "CONFIRMED",
    )

    assert result.new_status == "BUY_CONFIRMED"


def test_below_target_with_no_catalyst_stays_deploy_ready(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    entry = _entry(db_path, buy_price_target=10.0)
    _seed_confirming_snapshot(db_path, entry)

    result = check_entry_trigger(
        entry,
        db_path=db_path,
        checked_at="2026-05-10T12:00:00+00:00",
        price_lookup=lambda ticker: _price(ticker, 9.0),
        historical_median_lookup=lambda ticker: 11.0,
        catalyst_lookup=lambda ticker: "NONE",
    )

    assert result.new_status == "DEPLOY_READY"


def test_below_target_with_weak_catalyst_stays_deploy_ready(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    entry = _entry(db_path, buy_price_target=10.0)
    _seed_confirming_snapshot(db_path, entry)

    result = check_entry_trigger(
        entry,
        db_path=db_path,
        checked_at="2026-05-10T12:00:00+00:00",
        price_lookup=lambda ticker: _price(ticker, 9.0),
        historical_median_lookup=lambda ticker: 11.0,
        catalyst_lookup=lambda ticker: "WEAK",
    )

    assert result.new_status == "DEPLOY_READY"


def test_above_target_with_confirmed_catalyst_stays_active_and_skips_lookup(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    entry = _entry(db_path, buy_price_target=10.0)

    result = check_entry_trigger(
        entry,
        db_path=db_path,
        checked_at="2026-05-10T12:00:00+00:00",
        price_lookup=lambda ticker: _price(ticker, 12.0),
        historical_median_lookup=lambda ticker: 11.0,
        catalyst_lookup=_raises_if_called,
    )

    assert result.new_status == "ACTIVE"


def test_active_to_buy_confirmed_transition_label(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    entry = _entry(db_path, status="ACTIVE", buy_price_target=10.0)
    _seed_confirming_snapshot(db_path, entry)

    result = check_entry_trigger(
        entry,
        db_path=db_path,
        checked_at="2026-05-10T12:00:00+00:00",
        price_lookup=lambda ticker: _price(ticker, 9.0),
        historical_median_lookup=lambda ticker: 11.0,
        catalyst_lookup=lambda ticker: "CONFIRMED",
    )

    assert result.transition == "catalyst-confirmed below buy-target"


def test_record_trigger_status_change_accepts_buy_confirmed(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    _entry(db_path, buy_price_target=10.0)

    assert "BUY_CONFIRMED" in WATCHLIST_STATUSES
    record_trigger_status_change(
        "AAA",
        status="BUY_CONFIRMED",
        reason="catalyst-confirmed below buy-target",
        db_path=db_path,
    )
    latest = get_latest("AAA", db_path=db_path)
    assert latest is not None
    assert latest.status == "BUY_CONFIRMED"


# --- digest surfaces BUY (catalyst-confirmed) vs cheap-awaiting-catalyst ---

_DIGEST_NOW = datetime(2026, 5, 10, 12, 0, tzinfo=timezone.utc)
_BUY_REASON = "catalyst-confirmed below buy-target: price $9.00 <= buy-target $10.00"


def test_digest_renders_buy_confirmed_and_deploy_ready_sections(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    _entry(db_path, ticker="BUYME", status="ACTIVE", buy_price_target=10.0)
    record_trigger_status_change(
        "BUYME",
        status="BUY_CONFIRMED",
        reason=_BUY_REASON,
        db_path=db_path,
    )
    _entry(db_path, ticker="WAIT", status="DEPLOY_READY", buy_price_target=10.0)

    digest = render_digest(days_back=1, now=_DIGEST_NOW, db_path=db_path)

    assert "## At Target + Catalyst Confirmed" in digest
    assert "## Newly Deploy-Ready" in digest
    assert digest.index("## At Target + Catalyst Confirmed") != digest.index(
        "## Newly Deploy-Ready"
    )


def test_digest_buy_confirmed_section_includes_status_reason(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    _entry(db_path, ticker="BUYME", status="ACTIVE", buy_price_target=10.0)
    record_trigger_status_change(
        "BUYME",
        status="BUY_CONFIRMED",
        reason=_BUY_REASON,
        db_path=db_path,
    )

    digest = render_digest(days_back=1, now=_DIGEST_NOW, db_path=db_path)

    buy_section = digest.split("## At Target + Catalyst Confirmed", 1)[1].split("\n## ", 1)[0]
    assert _BUY_REASON in buy_section


def test_digest_omits_buy_confirmed_header_when_no_rows(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    _entry(db_path, ticker="WAIT", status="DEPLOY_READY", buy_price_target=10.0)

    digest = render_digest(days_back=1, now=_DIGEST_NOW, db_path=db_path)

    assert "## At Target + Catalyst Confirmed" not in digest


def test_dry_run_trigger_check_persists_no_catalyst_events(monkeypatch, tmp_path):
    # A dry_run trigger check must be side-effect-free, but the default
    # catalyst lookup persisted catalyst_events rows for within-reach names even
    # under dry_run (catalyst_context_for_ticker defaults persist=True).
    import sqlite3

    db_path = _init_temp_db(monkeypatch, tmp_path)
    entry = _entry(db_path, buy_price_target=10.0, valuation_anchor_value=12.0)

    result = check_entry_trigger(
        entry,
        db_path=db_path,
        dry_run=True,
        checked_at="2026-05-10T12:00:00+00:00",
        price_lookup=lambda ticker: _price(ticker, 8.0),  # 8.0 <= 10.0 -> within reach
        historical_median_lookup=lambda ticker: 12.0,  # in-band anchor -> reaches catalyst lookup
    )

    with sqlite3.connect(str(db_path)) as conn:
        count = conn.execute("SELECT COUNT(*) FROM catalyst_events").fetchone()[0]
    assert count == 0
    assert result.dry_run is True


def test_post_asof_buyback_filing_cannot_create_trigger_catalyst(
    monkeypatch,
    tmp_path,
):
    import sqlite3

    db_path = _init_temp_db(monkeypatch, tmp_path)
    entry = _entry(
        db_path,
        buy_price_target=10.0,
        valuation_anchor_value=12.0,
    )
    _seed_confirming_snapshot(db_path, entry)
    with sqlite3.connect(str(db_path)) as conn:
        conn.executemany(
            """
            INSERT INTO companyfacts_facts(
                ticker, fiscal_year, period_type, period_end, line_item,
                value, units, source_url, fetched_at, filed_date, accession
            )
            VALUES(
                'AAA', ?, 'FY', ?, 'share_repurchases_amount',
                ?, 'USD_millions', 'https://example.test/companyfacts', ?, ?, ?
            )
            """,
            [
                (
                    2024,
                    "2024-12-31",
                    200.0,
                    "2026-05-09T12:00:00Z",
                    "2026-05-09",
                    "0000000000-24-000001",
                ),
                (
                    2023,
                    "2023-12-31",
                    100.0,
                    "2026-05-09T12:00:00Z",
                    "2026-05-09",
                    "0000000000-23-000001",
                ),
            ],
        )
        conn.commit()

    result = check_entry_trigger(
        entry,
        db_path=db_path,
        checked_at="2026-05-10T12:00:00+00:00",
        price_lookup=lambda ticker: _price(ticker, 9.0),
        historical_median_lookup=lambda ticker: 11.0,
    )

    assert result.new_status == "DEPLOY_READY"
    with sqlite3.connect(str(db_path)) as conn:
        labels = dict(
            conn.execute(
                """
                SELECT catalyst_type, signal_label
                FROM catalyst_events
                WHERE ticker = 'AAA' AND as_of_date = '2026-05-08'
                """
            ).fetchall()
        )
    assert labels == {
        "BUYBACK_ACCELERATION": "NONE",
        "INSIDER_BUY_CLUSTER": "NONE",
    }


def test_blank_buyback_accession_cannot_affect_watchlist_trigger(
    monkeypatch,
    tmp_path,
):
    import sqlite3

    db_path = _init_temp_db(monkeypatch, tmp_path)
    entry = _entry(
        db_path,
        buy_price_target=10.0,
        valuation_anchor_value=12.0,
    )
    _seed_confirming_snapshot(db_path, entry)
    with sqlite3.connect(str(db_path)) as conn:
        conn.executemany(
            """
            INSERT INTO companyfacts_facts(
                ticker, fiscal_year, period_type, period_end, line_item,
                value, units, source_url, fetched_at, filed_date, accession
            )
            VALUES(
                'AAA', ?, 'FY', ?, 'share_repurchases_amount',
                ?, 'USD_millions', 'https://example.test/companyfacts', ?, ?, ?
            )
            """,
            [
                (
                    2024,
                    "2024-12-31",
                    200.0,
                    "2025-02-15T12:00:00Z",
                    "2025-02-15",
                    "",
                ),
                (
                    2023,
                    "2023-12-31",
                    100.0,
                    "2024-02-15T12:00:00Z",
                    "2024-02-15",
                    "0000000000-23-000001",
                ),
            ],
        )
        conn.commit()

    result = check_entry_trigger(
        entry,
        db_path=db_path,
        checked_at="2026-05-10T12:00:00+00:00",
        price_lookup=lambda ticker: _price(ticker, 9.0),
        historical_median_lookup=lambda ticker: 11.0,
    )

    assert result.new_status == "DEPLOY_READY"
    with sqlite3.connect(str(db_path)) as conn:
        row = conn.execute(
            """
            SELECT signal_label, detail_json
            FROM catalyst_events
            WHERE ticker = 'AAA'
              AND as_of_date = '2026-05-08'
              AND catalyst_type = 'BUYBACK_ACCELERATION'
            """
        ).fetchone()
    assert row[0] == "NONE"
    detail = json.loads(row[1])
    assert detail["status"] == "NEEDS_DATA"
    assert detail["reason_codes"] == ["MISSING_ACCESSION"]
    assert detail["source_lineage"] == []
