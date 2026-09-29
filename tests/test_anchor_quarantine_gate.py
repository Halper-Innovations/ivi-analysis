"""Watchlist add-time quarantine gate tests.

Verifies that ``_entry_from_candidate`` quarantines per-share anchors that fall
outside the [0.2x, 5.0x] band of the TRUSTED trailing fiscal-year median price,
and that a corrupt addition price no longer (mis)quarantines a name whose anchor
is in-band against the trusted trailing median.

Literal expected values are derived from real
watchlist anchors (BTM / BKNG).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from app.autonomous.sector_contract import (
    AutonomousSectorFinancialRunArtifact,
    SectorCompanyFinancialPacket,
)
from app.market.price_provider import PriceSnapshot
from app.watchlist.contract import (
    WATCHLIST_ACTIVE_STATUSES,
    WATCHLIST_STATUSES,
)


@pytest.fixture(autouse=True)
def _legacy_packet_fixture_bypasses_new_integrity_gate(monkeypatch):
    monkeypatch.setattr(
        "app.watchlist.store.artifact_decision_eligibility", lambda payload: "PASS"
    )
from app.watchlist.contract import WatchlistEntry
from app.watchlist.store import add_or_update
from app.watchlist.store import populate_from_sector_artifact
from app.watchlist.store import get_latest
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


def _seed_companyfacts_fy(db_path: Path, ticker: str) -> None:
    """Seed a single FY row so _historical_fiscal_year_median_price has a period_end."""
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            INSERT INTO companyfacts_facts
                (
                    ticker, fiscal_year, period_type, period_end, line_item,
                    value, units, source_url, fetched_at, filed_date
                )
            VALUES (
                ?, ?, 'FY', ?, 'Revenues', 1000.0, 'USD',
                'test://companyfacts', '2026-05-31T00:00:00+00:00',
                '2025-02-20'
            )
            """,
            (ticker.upper(), 2024, "2024-12-31"),
        )
        conn.commit()
    finally:
        conn.close()


def _stub_price_provider(monkeypatch, median_prices: dict[str, float]) -> None:
    """Monkeypatch the price provider so the FY-median lookup returns the stub price."""

    class _StubProvider:
        provider_name = "stub"

        def get_price_asof(self, ticker: str, as_of_date: str) -> PriceSnapshot | None:
            price = median_prices.get(ticker.upper())
            if price is None:
                return None
            return PriceSnapshot(
                ticker=ticker.upper(),
                as_of_date=as_of_date,
                price=float(price),
                source="stub",
            )

    monkeypatch.setattr(
        "app.watchlist.triggers.build_price_provider",
        lambda *args, **kwargs: _StubProvider(),
    )


def _packet(
    ticker: str,
    *,
    anchor: float,
    buy_price_target: float,
    current_price: float,
) -> SectorCompanyFinancialPacket:
    return SectorCompanyFinancialPacket(
        ticker=ticker,
        financial_status="OK",
        model_fit_status="OK",
        data_quality_status="OK",
        current_price=current_price,
        valuation={
            "anchor_method": "DCF",
            "valuation_anchor": anchor,
            "buy_price_target": buy_price_target,
        },
    )


def _artifact(ticker: str, packet: SectorCompanyFinancialPacket) -> AutonomousSectorFinancialRunArtifact:
    return AutonomousSectorFinancialRunArtifact(
        run_id=f"autonomous_sector_test_{ticker.lower()}",
        sector="test_sector",
        market_cap_focus="mid_cap",
        objective="Quarantine gate test.",
        as_of_date="2026-05-31",
        created_at="2026-05-31T12:00:00Z",
        completed_at="2026-05-31T12:01:00Z",
        status="COMPLETED",
        final_verdict="WATCHLIST",
        selected_ticker=None,
        confidence="MODERATE",
        candidate_selection={"selected_tickers": [ticker]},
        company_packets=[packet],
        relative_ranking=[
            {
                "ticker": ticker,
                "company_autonomy_verdict": "WATCHLIST_ONLY",
                "company_autonomy_confidence": "MODERATE",
                "positioning_summary": f"{ticker} test row.",
            }
        ],
        memo_body={"candidates": {}},
    )


def test_btm_shaped_anchor_is_quarantined_at_add(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    _seed_companyfacts_fy(db_path, "BTM")
    _stub_price_provider(monkeypatch, {"BTM": 2.93})

    packet = _packet("BTM", anchor=36.20, buy_price_target=27.2, current_price=30.0)
    populate_from_sector_artifact(_artifact("BTM", packet), db_path=db_path)
    latest = get_latest("BTM", db_path=db_path)

    assert latest is not None
    assert latest.status == "QUARANTINE"
    assert latest.status_reason is not None
    assert latest.status_reason.startswith("QUARANTINE_ANCHOR_ABOVE_BAND")
    assert latest.status_reason == "QUARANTINE_ANCHOR_ABOVE_BAND:anchor=36.20:ref=2.93:ratio=12.355"
    assert latest.status != "DEPLOY_READY"


def test_bkng_shaped_corrupt_addition_price_is_not_quarantined(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    _seed_companyfacts_fy(db_path, "BKNG")
    _stub_price_provider(monkeypatch, {"BKNG": 5000.0})

    # Corrupt addition price 177.25 would naively look broken, but the trusted
    # trailing median (5000.0) rescues BKNG: anchor/median = 0.887 is in-band.
    packet = _packet("BKNG", anchor=4435.46, buy_price_target=3326.6, current_price=177.25)
    populate_from_sector_artifact(_artifact("BKNG", packet), db_path=db_path)
    latest = get_latest("BKNG", db_path=db_path)

    assert latest is not None
    assert latest.status in {"ACTIVE", "DEPLOY_READY"}
    assert latest.status != "QUARANTINE"


def test_unassessable_anchor_at_add_surfaces_reason_not_silent(monkeypatch, tmp_path):
    # At add time an anchor with NO trusted reference at all (FY-median and
    # current price both missing) is UNASSESSABLE and must be surfaced in
    # status_reason, not silently admitted with an unverified anchor (plan 242).
    db_path = _init_temp_db(monkeypatch, tmp_path)
    # No companyfacts FY seeded -> median None; current_price None -> no fallback.
    packet = _packet("NOREF", anchor=50.0, buy_price_target=40.0, current_price=None)
    populate_from_sector_artifact(_artifact("NOREF", packet), db_path=db_path)
    latest = get_latest("NOREF", db_path=db_path)

    assert latest is not None
    assert latest.status != "DEPLOY_READY"
    assert latest.status_reason is not None
    assert "UNASSESSABLE_NO_REFERENCE" in latest.status_reason


def test_quarantine_is_an_active_and_known_status():
    assert "QUARANTINE" in WATCHLIST_ACTIVE_STATUSES
    assert "QUARANTINE" in WATCHLIST_STATUSES


# --- quarantine / un-quarantine at trigger recheck ------------------


def _trigger_entry(
    db_path: Path,
    *,
    ticker: str,
    status: str,
    buy_price_target: float,
    valuation_anchor_value: float,
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
            thesis_text=f"{ticker} trigger recheck fixture.",
            source_run_id="sector_run_b4t5",
            source_sector="test_sector",
            added_at="2026-05-01T12:00:00+00:00",
        ),
        db_path=db_path,
    )
    latest = get_latest(ticker, db_path=db_path)
    assert latest is not None
    return latest


def test_btm_entry_is_quarantined_at_recheck_not_deploy_ready(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    entry = _trigger_entry(
        db_path,
        ticker="BTM",
        status="ACTIVE",
        buy_price_target=27.2,
        valuation_anchor_value=36.20,
    )

    result = check_entry_trigger(
        entry,
        db_path=db_path,
        checked_at="2026-05-10T12:00:00+00:00",
        price_lookup=lambda ticker: PriceSnapshot(
            ticker=ticker, as_of_date="2026-05-08", price=2.93, source="stub"
        ),
        historical_median_lookup=lambda ticker: 2.93,
    )
    latest = get_latest("BTM", db_path=db_path)

    assert result.new_status == "QUARANTINE"
    assert result.warning is not None
    assert result.warning.startswith("QUARANTINE_ANCHOR_ABOVE_BAND")
    assert result.new_status != "DEPLOY_READY"
    assert latest is not None
    assert latest.status == "QUARANTINE"


def test_bkng_mislabeled_suspect_lifts_when_anchor_in_band(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    entry = _trigger_entry(
        db_path,
        ticker="BKNG",
        status="PRICE_DATA_SUSPECT",
        buy_price_target=3326.6,
        valuation_anchor_value=4435.46,
    )

    result = check_entry_trigger(
        entry,
        db_path=db_path,
        checked_at="2026-05-10T12:00:00+00:00",
        price_lookup=lambda ticker: PriceSnapshot(
            ticker=ticker, as_of_date="2026-05-08", price=4900.0, source="stub"
        ),
        historical_median_lookup=lambda ticker: 5000.0,
    )

    assert result.new_status in {"ACTIVE", "DEPLOY_READY"}
    assert result.new_status != "QUARANTINE"


def test_quarantined_entry_lifts_when_anchor_repaired(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    entry = _trigger_entry(
        db_path,
        ticker="ZZZ",
        status="QUARANTINE",
        buy_price_target=90.0,
        valuation_anchor_value=80.0,
    )

    # QUARANTINE is force-checkable (not trigger-eligible), so a forced recheck
    # re-evaluates and lifts the quarantine once the anchor is back in-band.
    result = check_entry_trigger(
        entry,
        db_path=db_path,
        force=True,
        checked_at="2026-05-10T12:00:00+00:00",
        price_lookup=lambda ticker: PriceSnapshot(
            ticker=ticker, as_of_date="2026-05-08", price=95.0, source="stub"
        ),
        historical_median_lookup=lambda ticker: 100.0,
    )

    assert result.new_status in {"ACTIVE", "DEPLOY_READY"}


def test_no_reference_recheck_quarantines_out_of_band_anchor_not_deploy_ready(monkeypatch, tmp_path):
    # When the trailing FY-median reference is missing, an out-of-band anchor
    # must NOT silently fall through to DEPLOY_READY. The recheck falls back to
    # the latest price as the reference (mirroring the add-time gate) and
    # QUARANTINEs the bad anchor.
    db_path = _init_temp_db(monkeypatch, tmp_path)
    entry = _trigger_entry(
        db_path,
        ticker="NOREF",
        status="ACTIVE",
        buy_price_target=12.0,
        valuation_anchor_value=1000.0,  # absurd per-share anchor vs a ~$10 price
    )

    result = check_entry_trigger(
        entry,
        db_path=db_path,
        checked_at="2026-05-10T12:00:00+00:00",
        price_lookup=lambda ticker: PriceSnapshot(
            ticker=ticker, as_of_date="2026-05-08", price=10.0, source="stub"
        ),
        historical_median_lookup=lambda ticker: None,  # no trailing reference
    )

    assert result.new_status not in {"DEPLOY_READY", "BUY_CONFIRMED"}
    assert result.new_status == "QUARANTINE"
    assert result.warning is not None
    assert result.warning.startswith("QUARANTINE_ANCHOR_ABOVE_BAND")


def test_unassessable_anchor_recheck_is_not_promoted(monkeypatch, tmp_path):
    # A non-positive / unassessable anchor (no usable reference at all) must
    # never be silently promoted to a buy status even when price <= target.
    db_path = _init_temp_db(monkeypatch, tmp_path)
    entry = _trigger_entry(
        db_path,
        ticker="ZEROA",
        status="ACTIVE",
        buy_price_target=12.0,
        valuation_anchor_value=0.0,
    )

    result = check_entry_trigger(
        entry,
        db_path=db_path,
        checked_at="2026-05-10T12:00:00+00:00",
        price_lookup=lambda ticker: PriceSnapshot(
            ticker=ticker, as_of_date="2026-05-08", price=10.0, source="stub"
        ),
        historical_median_lookup=lambda ticker: None,
    )

    assert result.new_status not in {"DEPLOY_READY", "BUY_CONFIRMED"}
    assert result.warning is not None
    assert "UNASSESSABLE_NO_REFERENCE" in result.warning
