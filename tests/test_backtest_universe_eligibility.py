from __future__ import annotations

from dataclasses import dataclass

from app.config import get_config
from app.db import connect, get_db, init_db


def _init(monkeypatch, tmp_path):
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    get_config.cache_clear()
    init_db()


def _seed_fact(
    conn,
    ticker,
    fiscal_year,
    period_end,
    line_item,
    value,
    *,
    filed_date,
    units="shares_millions",
    source_url="https://example.test/companyfacts",
    accession="0000000000-22-000001",
):
    conn.execute(
        "INSERT INTO companyfacts_facts(ticker, fiscal_year, period_type, period_end, "
        "line_item, value, units, source_url, filed_date, accession, fetched_at) "
        "VALUES(?, ?, 'FY', ?, ?, ?, ?, ?, ?, ?, '2025-01-01')",
        (
            ticker,
            fiscal_year,
            period_end,
            line_item,
            value,
            units,
            source_url,
            filed_date,
            accession,
        ),
    )


@dataclass
class _Snap:
    ticker: str
    as_of_date: str
    price: float


class _FakeProvider:
    def __init__(self, prices):
        self._prices = prices  # {ticker: price}

    def get_price_asof(self, ticker, as_of_date):
        p = self._prices.get(ticker)
        return _Snap(ticker, as_of_date, p) if p is not None else None


def test_eligibility_is_point_in_time_and_survivorship_aware(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    from app.backtest import universe

    # INCL: filed FY2021 (period_end 2021-12-31) -> visible as-of 2022-07-01 (cutoff 2022-04-02)
    # DEAD: same vintage, still in companyfacts even though it later delisted -> must be INCLUDED
    # FUTR: only filed FY2024 (period_end 2024-12-31) -> NOT visible as-of 2022-07-01
    with get_db() as conn:
        _seed_fact(
            conn,
            "INCL",
            2021,
            "2021-12-31",
            "shares_outstanding",
            50.0,
            filed_date="2022-03-01",
            accession="0000000000-22-000001",
        )
        _seed_fact(
            conn,
            "DEAD",
            2021,
            "2021-12-31",
            "shares_outstanding",
            20.0,
            filed_date="2022-03-15",
            accession="0000000000-22-000002",
        )
        _seed_fact(
            conn,
            "FUTR",
            2024,
            "2024-12-31",
            "shares_outstanding",
            100.0,
            filed_date="2025-02-15",
            accession="0000000000-25-000001",
        )
        _seed_fact(
            conn,
            "POST",
            2021,
            "2021-12-31",
            "shares_outstanding",
            30.0,
            filed_date="2022-05-01",
            accession="0000000000-22-000003",
        )
        _seed_fact(
            conn,
            "CHRON",
            2021,
            "2021-12-31",
            "shares_outstanding",
            30.0,
            filed_date="2021-12-01",
            accession="0000000000-21-000003",
        )
        _seed_fact(
            conn,
            "NOUNIT",
            2021,
            "2021-12-31",
            "shares_outstanding",
            30.0,
            filed_date="2022-03-01",
            units="",
            accession="0000000000-22-000004",
        )
        _seed_fact(
            conn,
            "NOACC",
            2021,
            "2021-12-31",
            "shares_outstanding",
            30.0,
            filed_date="2022-03-01",
            accession="",
        )

    provider = _FakeProvider(
        {
            "INCL": 30.0,
            "DEAD": 10.0,
            "FUTR": 25.0,
            "POST": 10.0,
            "CHRON": 10.0,
            "NOUNIT": 10.0,
            "NOACC": 10.0,
        }
    )
    # Make cap come straight from a stubbed resolver (price x shares), deterministically.
    monkeypatch.setattr(
        universe,
        "resolve_market_cap_from_price_asof",
        lambda *, ticker, as_of_date, price, run_id=None: (
            {"INCL": 1500.0, "DEAD": 200.0, "FUTR": 2500.0}.get(ticker),
            {},
        ),
    )

    eligible = universe.eligible_universe_asof("2022-07-01", provider=provider)
    by_ticker = {e.ticker: e for e in eligible}

    assert "INCL" in by_ticker
    assert "DEAD" in by_ticker  # survivorship-aware: delisted-but-then-filed is kept
    assert "FUTR" not in by_ticker  # not yet filed as-of T
    assert "POST" not in by_ticker  # period ended, but filing landed after cutoff
    assert "CHRON" not in by_ticker  # impossible period_end > filed_date
    assert "NOUNIT" not in by_ticker  # canonical unit is mandatory
    assert "NOACC" not in by_ticker  # exact filing accession is mandatory
    assert by_ticker["INCL"].cap_category_asof == "small"
    assert by_ticker["DEAD"].cap_category_asof == "micro"
    assert by_ticker["INCL"].market_cap_asof == 1500.0
    assert by_ticker["DEAD"].market_cap_asof == 200.0
    assert by_ticker["INCL"].price_asof == 30.0


def test_explicit_db_path_is_authoritative_for_membership(
    monkeypatch,
    tmp_path,
):
    _init(monkeypatch, tmp_path)
    from app.backtest import universe

    with get_db() as conn:
        _seed_fact(
            conn,
            "CONFIG",
            2021,
            "2021-12-31",
            "shares_outstanding",
            50.0,
            filed_date="2022-03-01",
        )

    alternate_path = tmp_path / "alternate" / "historical.db"
    alternate_path.parent.mkdir(parents=True)
    conn = connect(alternate_path)
    try:
        init_db(conn=conn)
        _seed_fact(
            conn,
            "EXACT",
            2021,
            "2021-12-31",
            "shares_outstanding",
            20.0,
            filed_date="2022-03-01",
            accession="0000000000-22-000099",
        )
        conn.commit()
    finally:
        conn.close()

    monkeypatch.setattr(
        universe,
        "resolve_market_cap_from_price_asof",
        lambda *, ticker, as_of_date, price, run_id=None: (200.0, {}),
    )
    eligible = universe.eligible_universe_asof(
        "2022-07-01",
        provider=_FakeProvider({"CONFIG": 10.0, "EXACT": 10.0}),
        db_path=alternate_path,
    )

    assert [name.ticker for name in eligible] == ["EXACT"]
