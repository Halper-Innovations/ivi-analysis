"""Price integrity: staleness ceilings, mtime fallback removal, provider
chain traversal, two-heartbeat at-target confirmation."""

from __future__ import annotations

import sqlite3

from app.config import get_config
from app.market.price_provider import (
    ChainedPriceProvider,
    PriceSnapshot,
    _load_db_quote_snapshot,
)


def _snap(ticker: str, price: float, as_of: str, source: str = "test") -> PriceSnapshot:
    return PriceSnapshot(
        ticker=ticker,
        as_of_date=as_of,
        price=price,
        currency="USD",
        source=source,
        retrieved_at=f"{as_of}T12:00:00+00:00",
        confidence="HIGH",
    )


class _StubProvider:
    def __init__(self, name: str, snapshot: PriceSnapshot | None, reason: str = "SYMBOL_NOT_FOUND"):
        self.provider_name = name
        self._snapshot = snapshot
        self._reason = reason
        self.calls = 0

    def get_price_asof(self, ticker: str, as_of_date: str) -> PriceSnapshot | None:
        self.calls += 1
        return self._snapshot

    def get_last_diagnostic(self, ticker: str, as_of_date: str):
        return {"result": {"reason_code": self._reason, "retryable": False}}


def test_chain_tries_every_provider_until_success():
    # Third provider succeeds even though the first fails NON-retryably —
    # the old 2-provider cap and first-provider chain-break are both gone.
    p1 = _StubProvider("a", None, reason="SYMBOL_NOT_FOUND")
    p2 = _StubProvider("b", None, reason="TIMEOUT")
    p3 = _StubProvider("c", _snap("AAA", 12.5, "2026-05-08"))
    chain = ChainedPriceProvider([p1, p2, p3])

    snapshot = chain.get_price_asof("AAA", "2026-05-08")

    assert snapshot is not None and snapshot.price == 12.5
    assert (p1.calls, p2.calls, p3.calls) == (1, 1, 1)


def test_chain_stops_at_first_success():
    p1 = _StubProvider("a", _snap("AAA", 9.0, "2026-05-08"))
    p2 = _StubProvider("b", _snap("AAA", 1.0, "2026-05-08"))
    chain = ChainedPriceProvider([p1, p2])

    snapshot = chain.get_price_asof("AAA", "2026-05-08")

    assert snapshot is not None and snapshot.price == 9.0
    assert (p1.calls, p2.calls) == (1, 0)


def test_db_quote_snapshot_rejects_over_age_rows(monkeypatch, tmp_path):
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "engine.db"))
    get_config.cache_clear()
    from app.db import init_db

    init_db()
    conn = sqlite3.connect(str(tmp_path / "engine.db"))
    conn.execute(
        """
        INSERT INTO price_quotes(
            ticker, provider, as_of_date, price, currency, status,
            fetched_at, expires_at, raw_json, quote_hash
        ) VALUES ('AAA', 'stooq', '2026-05-01', 10.0, 'USD', 'OK',
                  '2026-05-01T12:00:00+00:00', '2026-05-02T12:00:00+00:00', '{}', 'h')
        """
    )
    conn.commit()
    conn.close()

    # 3 days behind the requested as-of: within the 7-day ceiling.
    fresh = _load_db_quote_snapshot(ticker="AAA", as_of_date="2026-05-04")
    assert fresh is not None and fresh.price == 10.0
    # 20 days behind: over the ceiling -> NO_PRICE, not a stale price.
    stale = _load_db_quote_snapshot(ticker="AAA", as_of_date="2026-05-21")
    assert stale is None
    get_config.cache_clear()


def test_mtime_glob_fallback_is_gone():
    import app.valuation.valuation_writer as vw

    assert not hasattr(vw, "_latest_cached_price_artifact")


def test_trigger_refuses_stale_price_and_demotes_deploy_ready(monkeypatch, tmp_path):
    from tests.test_watchlist_triggers import _entry, _init_temp_db
    from app.watchlist.triggers import check_entry_trigger
    from app.watchlist.store import get_latest

    db_path = _init_temp_db(monkeypatch, tmp_path)
    entry = _entry(db_path, status="DEPLOY_READY", buy_price_target=75.0)

    stale_snap = _snap("AAA", 70.0, "2026-05-01", source="yahoo")
    result = check_entry_trigger(
        entry,
        db_path=db_path,
        checked_at="2026-05-10T12:00:00+00:00",
        price_lookup=lambda ticker: stale_snap,
    )

    assert result.new_status == "PRICE_DATA_SUSPECT"
    assert result.transition == "price-stale-demotion"
    assert result.warning is not None and result.warning.startswith("PRICE_STALE:")
    assert result.price_age_days == 9
    latest = get_latest("AAA", db_path=db_path)
    assert latest is not None and latest.status == "PRICE_DATA_SUSPECT"


def test_trigger_skips_stale_price_for_active_row(monkeypatch, tmp_path):
    from tests.test_watchlist_triggers import _entry, _init_temp_db, _snapshot_count
    from app.watchlist.triggers import check_entry_trigger

    db_path = _init_temp_db(monkeypatch, tmp_path)
    entry = _entry(db_path, status="ACTIVE", buy_price_target=75.0)

    result = check_entry_trigger(
        entry,
        db_path=db_path,
        checked_at="2026-05-10T12:00:00+00:00",
        price_lookup=lambda ticker: _snap("AAA", 70.0, "2026-05-01", source="yahoo"),
    )

    assert result.new_status == "ACTIVE"
    assert result.transition == "skipped"
    assert result.warning is not None and result.warning.startswith("PRICE_STALE:")
    # A stale price never lands in the snapshot log (it would poison the
    # two-heartbeat confirmation and render as "latest").
    assert _snapshot_count(db_path) == 0


def test_fresh_at_target_crossing_needs_two_heartbeats(monkeypatch, tmp_path):
    from tests.test_watchlist_triggers import _entry, _init_temp_db, _price
    from app.watchlist.triggers import check_entry_trigger
    from app.watchlist.store import get_latest

    db_path = _init_temp_db(monkeypatch, tmp_path)
    entry = _entry(db_path, status="ACTIVE", buy_price_target=75.0)

    # Heartbeat 1: at target, but no prior confirming snapshot -> stays
    # ACTIVE, pending confirmation; the snapshot is recorded.
    first = check_entry_trigger(
        entry,
        db_path=db_path,
        checked_at="2026-05-09T12:00:00+00:00",
        price_lookup=lambda ticker: _price(ticker, 70.0, "yahoo"),
    )
    assert first.new_status == "ACTIVE"
    assert first.transition == "at-target-pending-confirmation"
    assert first.warning is not None and first.warning.startswith(
        "AT_TARGET_PENDING_CONFIRMATION"
    )
    latest = get_latest("AAA", db_path=db_path)
    assert latest is not None and latest.status == "ACTIVE"

    # Heartbeat 2: still at target -> confirmed flip.
    second = check_entry_trigger(
        latest,
        db_path=db_path,
        checked_at="2026-05-10T12:00:00+00:00",
        price_lookup=lambda ticker: _price(ticker, 70.0, "yahoo"),
    )
    assert second.new_status == "DEPLOY_READY"
    assert second.transition == "crossed below buy-target"
    final = get_latest("AAA", db_path=db_path)
    assert final is not None and final.status == "DEPLOY_READY"


def test_digest_imperative_lines_carry_price_basis(monkeypatch, tmp_path):
    from tests.test_watchlist_digest import _entry as digest_entry, _init_temp_db as digest_init
    from app.watchlist.digest import render_digest

    db_path = digest_init(monkeypatch, tmp_path)
    digest_entry(
        db_path,
        ticker="AAA",
        status="DEPLOY_READY",
        conviction_grade="ACTIONABLE",
        confidence="HIGH",
        conviction_source="sector_final_decision",
        buy_price_target=100.0,
        current_price_at_addition=90.0,
    )

    digest = render_digest(days_back=1, db_path=db_path)

    # No snapshot exists, so the addition-time fallback must be labeled.
    assert "\n  - (price basis: ADDITION-TIME price" in digest
