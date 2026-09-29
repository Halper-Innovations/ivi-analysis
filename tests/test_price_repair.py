from __future__ import annotations

import json
import sqlite3
from hashlib import sha256
from pathlib import Path

import pytest

from app.config import AppConfig
from app.market.price_provider import PriceSnapshot
from app.autonomous.price_repair import resolve_v2_price


def _cfg(tmp_path: Path, *, db_path: Path | None = None) -> AppConfig:
    return AppConfig(
        data_dir=tmp_path / "data",
        db_path=db_path or tmp_path / "configured.db",
        outputs_dir=tmp_path / "outputs",
        cache_dir=tmp_path / "cache",
        sectors_dir=tmp_path / "sectors",
        price_fallback_days=7,
        quote_ttl_seconds=86400,
        price_provider="disabled",
    )


def _init_price_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute(
        """
        CREATE TABLE price_quotes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            provider TEXT NOT NULL,
            as_of_date TEXT NOT NULL,
            price REAL,
            currency TEXT,
            source_url TEXT,
            status TEXT NOT NULL,
            fetched_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            raw_json TEXT NOT NULL,
            quote_hash TEXT NOT NULL,
            UNIQUE(ticker, provider, as_of_date)
        )
        """
    )
    conn.commit()
    conn.close()


def _insert_quote(
    path: Path,
    *,
    ticker: str = "AAA",
    provider: str = "seed",
    as_of_date: str = "2026-06-01",
    price: float = 50.0,
    currency: str = "USD",
) -> None:
    conn = sqlite3.connect(str(path))
    conn.execute(
        """
        INSERT INTO price_quotes(
            ticker, provider, as_of_date, price, currency, source_url,
            status, fetched_at, expires_at, raw_json, quote_hash
        ) VALUES(?, ?, ?, ?, ?, ?, 'OK', ?, ?, '{}', ?)
        """,
        (
            ticker,
            provider,
            as_of_date,
            price,
            currency,
            "https://prices.example/quote",
            f"{as_of_date}T12:00:00+00:00",
            "2026-06-30T12:00:00+00:00",
            f"hash-{provider}-{as_of_date}",
        ),
    )
    conn.commit()
    conn.close()


class _FakeProvider:
    provider_name = "fake"

    def __init__(
        self,
        snapshot: PriceSnapshot | None,
        *,
        reason_code: str = "SYMBOL_NOT_FOUND",
    ) -> None:
        self.snapshot = snapshot
        self.reason_code = reason_code
        self.calls = 0

    def get_price_asof(self, ticker: str, as_of_date: str) -> PriceSnapshot | None:
        self.calls += 1
        return self.snapshot

    def get_last_diagnostic(self, ticker: str, as_of_date: str) -> dict[str, object]:
        return {
            "ticker": ticker,
            "requested_as_of": as_of_date,
            "provider_attempts": [
                {
                    "provider": "fake",
                    "status": self.reason_code,
                    "error_code": self.reason_code,
                }
            ],
            "result": {
                "status": "UNKNOWN",
                "reason_code": self.reason_code,
                "reason_detail": f"fake {self.reason_code}",
                "retryable": False,
            },
        }


def _provider_snapshot(
    *,
    price: float = 88.0,
    as_of_date: str = "2026-06-01",
    currency: str = "USD",
) -> PriceSnapshot:
    return PriceSnapshot(
        ticker="AAA",
        as_of_date=as_of_date,
        price=price,
        currency=currency,
        source="fake",
        retrieved_at="2026-06-01T12:00:00+00:00",
        url="https://prices.example/fake",
        confidence="HIGH",
    )


def test_cap_stage_price_has_precedence_and_is_persisted(tmp_path: Path) -> None:
    db_path = tmp_path / "injected.db"
    _init_price_db(db_path)
    _insert_quote(db_path, price=50.0)
    provider = _FakeProvider(_provider_snapshot(price=88.0))

    result = resolve_v2_price(
        "AAA",
        as_of_date="2026-06-01",
        db_path=db_path,
        cap_stage_price={
            "ticker": "AAA",
            "as_of_date": "2026-06-01",
            "price_as_of_date": "2026-05-31",
            "price_used": 125.0,
            "price_currency": "USD",
            "price_source": "cap_packet",
            "price_source_url": "https://prices.example/cap",
            "price_confidence": "HIGH",
        },
        cfg=_cfg(tmp_path),
        provider=provider,
    )

    assert result.status == "RESOLVED"
    assert result.source_resolution == "cap_stage"
    assert result.snapshot is not None
    assert result.snapshot.price == 125.0
    assert result.snapshot.as_of_date == "2026-05-31"
    assert result.persisted is True
    assert [attempt.source for attempt in result.attempts] == ["cap_stage"]
    assert provider.calls == 0
    conn = sqlite3.connect(str(db_path))
    row = conn.execute(
        "SELECT price, currency FROM price_quotes WHERE provider = 'cap_packet'"
    ).fetchone()
    conn.close()
    assert row == (125.0, "USD")


def test_injected_price_quotes_wins_and_global_config_db_is_ignored(
    tmp_path: Path,
) -> None:
    injected_db = tmp_path / "injected.db"
    configured_db = tmp_path / "configured.db"
    _init_price_db(injected_db)
    _init_price_db(configured_db)
    _insert_quote(injected_db, provider="injected", price=55.0)
    _insert_quote(configured_db, provider="configured", price=999.0)
    provider = _FakeProvider(_provider_snapshot(price=88.0))

    result = resolve_v2_price(
        "AAA",
        as_of_date="2026-06-01",
        db_path=injected_db,
        cfg=_cfg(tmp_path, db_path=configured_db),
        provider=provider,
    )

    assert result.status == "RESOLVED"
    assert result.source_resolution == "price_quotes"
    assert result.snapshot is not None
    assert result.snapshot.price == 55.0
    assert result.persisted is True
    assert [attempt.source for attempt in result.attempts] == [
        "cap_stage",
        "price_quotes",
    ]
    assert provider.calls == 0


def test_future_cap_price_is_rejected_before_valid_injected_quote(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "injected.db"
    _init_price_db(db_path)
    _insert_quote(db_path, as_of_date="2026-05-31", price=51.0)

    result = resolve_v2_price(
        "AAA",
        as_of_date="2026-06-01",
        db_path=db_path,
        cap_stage_price={
            "ticker": "AAA",
            "price_used": 200.0,
            "price_as_of_date": "2026-06-02",
            "price_source": "future_source",
        },
        cfg=_cfg(tmp_path),
        allow_provider=False,
    )

    assert result.status == "RESOLVED"
    assert result.source_resolution == "price_quotes"
    assert result.snapshot is not None
    assert result.snapshot.price == 51.0
    assert result.attempts[0].status == "REJECTED"
    assert result.attempts[0].reason_code == "FUTURE_PRICE"


def test_stale_sources_exhaust_to_terminal_offline_needs_data(tmp_path: Path) -> None:
    db_path = tmp_path / "injected.db"
    _init_price_db(db_path)
    _insert_quote(db_path, as_of_date="2026-05-20", price=49.0)

    result = resolve_v2_price(
        "AAA",
        as_of_date="2026-06-01",
        db_path=db_path,
        cap_stage_price={
            "ticker": "AAA",
            "price_used": 48.0,
            "price_as_of_date": "2026-05-21",
            "price_source": "stale_cap",
        },
        cfg=_cfg(tmp_path),
        allow_provider=False,
        max_age_days=7,
    )

    assert result.status == "NEEDS_DATA"
    assert result.reason_code == "OFFLINE_NO_CACHE"
    assert result.retryable is False
    assert result.terminal_for_attempt is True
    assert result.snapshot is None
    assert result.attempts[0].reason_code == "STALE_PRICE"
    assert result.attempts[1].reason_code == "STALE_PRICE"
    assert result.attempts[-1].reason_code == "OFFLINE_NO_CACHE"


def test_non_usd_source_is_rejected_before_later_usd_source(tmp_path: Path) -> None:
    db_path = tmp_path / "injected.db"
    _init_price_db(db_path)
    _insert_quote(db_path, price=50.0)
    provider = _FakeProvider(_provider_snapshot(price=88.0))

    result = resolve_v2_price(
        "AAA",
        as_of_date="2026-06-01",
        db_path=db_path,
        cap_stage_price={
            "ticker": "AAA",
            "price_used": 42.0,
            "price_as_of_date": "2026-06-01",
            "price_currency": "EUR",
            "price_source": "foreign_quote",
        },
        cfg=_cfg(tmp_path),
        provider=provider,
    )

    assert result.status == "RESOLVED"
    assert result.reason_code == "PRICE_RESOLVED"
    assert result.source_resolution == "price_quotes"
    assert result.snapshot is not None
    assert (result.snapshot.price, result.snapshot.currency) == (50.0, "USD")
    assert result.persisted is True
    assert result.attempts[0].status == "REJECTED"
    assert result.attempts[0].reason_code == "NON_USD_PRICE_UNSUPPORTED"
    assert result.attempts[0].terminal_for_attempt is False
    assert provider.calls == 0


def test_all_non_usd_sources_end_in_precise_terminal_needs_data(tmp_path: Path) -> None:
    db_path = tmp_path / "injected.db"
    _init_price_db(db_path)
    _insert_quote(db_path, price=50.0, currency="CAD")

    result = resolve_v2_price(
        "AAA",
        as_of_date="2026-06-01",
        db_path=db_path,
        cap_stage_price={
            "ticker": "AAA",
            "price_used": 42.0,
            "price_as_of_date": "2026-06-01",
            "price_currency": "EUR",
            "price_source": "foreign_quote",
        },
        cfg=_cfg(tmp_path),
        allow_provider=False,
    )

    assert result.status == "NEEDS_DATA"
    assert result.reason_code == "NON_USD_PRICE_UNSUPPORTED"
    assert result.source_resolution is None
    assert result.snapshot is None
    assert result.terminal_for_attempt is True
    assert [attempt.reason_code for attempt in result.attempts[:2]] == [
        "NON_USD_PRICE_UNSUPPORTED",
        "NON_USD_PRICE_UNSUPPORTED",
    ]


def test_missing_currency_is_not_assumed_usd_and_later_usd_source_wins(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "injected.db"
    _init_price_db(db_path)
    _insert_quote(db_path, price=50.0, currency="USD")

    result = resolve_v2_price(
        "AAA",
        as_of_date="2026-06-01",
        db_path=db_path,
        cap_stage_price={
            "ticker": "AAA",
            "price_used": 42.0,
            "price_as_of_date": "2026-06-01",
            "price_source": "currency_unknown",
        },
        cfg=_cfg(tmp_path),
        allow_provider=False,
    )

    assert result.status == "RESOLVED"
    assert result.source_resolution == "price_quotes"
    assert result.snapshot is not None
    assert result.snapshot.currency == "USD"
    assert result.attempts[0].reason_code == "PRICE_CURRENCY_UNRESOLVED"


def test_run_scoped_output_wins_before_disk_and_provider(tmp_path: Path) -> None:
    db_path = tmp_path / "injected.db"
    _init_price_db(db_path)
    cfg = _cfg(tmp_path)
    run_dir = cfg.outputs_dir / "prices" / "run-123"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "AAA.json").write_text(
        json.dumps(
            {
                "status": "OK",
                "requested_as_of_date": "2026-06-01",
                "snapshot": {
                    "ticker": "AAA",
                    "as_of_date": "2026-05-31",
                    "price": 31.0,
                    "currency": "USD",
                    "source": "run_provider",
                    "retrieved_at": "2026-06-01T01:00:00+00:00",
                },
            }
        ),
        encoding="utf-8",
    )
    disk_path = cfg.cache_dir / "prices" / "AAA.json"
    disk_path.parent.mkdir(parents=True, exist_ok=True)
    disk_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "requested_as_of_date": "2026-06-01",
                        "source": "disk_provider",
                        "snapshot": {
                            "ticker": "AAA",
                            "as_of_date": "2026-06-01",
                            "price": 42.0,
                            "currency": "USD",
                            "source": "disk_provider",
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    provider = _FakeProvider(_provider_snapshot(price=88.0))

    result = resolve_v2_price(
        "AAA",
        as_of_date="2026-06-01",
        db_path=db_path,
        run_id="run-123",
        cfg=cfg,
        provider=provider,
    )

    assert result.status == "RESOLVED"
    assert result.source_resolution == "run_scoped_output"
    assert result.snapshot is not None
    assert result.snapshot.price == 31.0
    assert [attempt.source for attempt in result.attempts] == [
        "cap_stage",
        "price_quotes",
        "run_scoped_output",
    ]
    assert provider.calls == 0


def test_disk_cache_wins_after_run_miss(tmp_path: Path) -> None:
    db_path = tmp_path / "injected.db"
    _init_price_db(db_path)
    cfg = _cfg(tmp_path)
    disk_path = cfg.cache_dir / "prices" / "AAA.json"
    disk_path.parent.mkdir(parents=True, exist_ok=True)
    disk_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "requested_as_of_date": "2026-06-01",
                        "source": "disk_provider",
                        "snapshot": {
                            "ticker": "AAA",
                            "as_of_date": "2026-05-30",
                            "price": 42.0,
                            "currency": "USD",
                            "source": "disk_provider",
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    provider = _FakeProvider(_provider_snapshot(price=88.0))

    result = resolve_v2_price(
        "AAA",
        as_of_date="2026-06-01",
        db_path=db_path,
        cfg=cfg,
        provider=provider,
    )

    assert result.status == "RESOLVED"
    assert result.source_resolution == "disk_cache"
    assert result.snapshot is not None
    assert result.snapshot.price == 42.0
    assert [attempt.source for attempt in result.attempts] == [
        "cap_stage",
        "price_quotes",
        "run_scoped_output",
        "disk_cache",
    ]
    assert provider.calls == 0


def test_prior_sector_artifact_wins_before_provider(tmp_path: Path) -> None:
    db_path = tmp_path / "injected.db"
    _init_price_db(db_path)
    cfg = _cfg(tmp_path)
    sector_run = cfg.sectors_dir / "sector_run_001"
    sector_run.mkdir(parents=True, exist_ok=True)
    (sector_run / "valuation_AAA.json").write_text(
        json.dumps(
            {
                "ticker": "AAA",
                "as_of_date": "2026-06-01",
                "input_snapshot": {
                    "current_price": 63.0,
                    "price_asof_used": "2026-05-31",
                    "currency": "USD",
                },
            }
        ),
        encoding="utf-8",
    )
    provider = _FakeProvider(_provider_snapshot(price=88.0))

    result = resolve_v2_price(
        "AAA",
        as_of_date="2026-06-01",
        db_path=db_path,
        cfg=cfg,
        provider=provider,
    )

    assert result.status == "RESOLVED"
    assert result.source_resolution == "prior_sector_artifacts"
    assert result.snapshot is not None
    assert result.snapshot.price == 63.0
    assert result.attempts[-1].source == "prior_sector_artifacts"
    assert provider.calls == 0


def test_provider_hit_persists_into_injected_database(tmp_path: Path) -> None:
    db_path = tmp_path / "injected.db"
    _init_price_db(db_path)
    cfg = _cfg(tmp_path)
    provider = _FakeProvider(_provider_snapshot(price=77.0, as_of_date="2026-05-31"))

    result = resolve_v2_price(
        "AAA",
        as_of_date="2026-06-01",
        db_path=db_path,
        cfg=cfg,
        provider=provider,
    )

    assert result.status == "RESOLVED"
    assert result.source_resolution == "provider"
    assert result.snapshot is not None
    assert result.snapshot.price == 77.0
    assert result.persisted is True
    assert provider.calls == 1
    conn = sqlite3.connect(str(db_path))
    row = conn.execute(
        "SELECT provider, as_of_date, price, currency, status FROM price_quotes"
    ).fetchone()
    conn.close()
    assert row == ("fake", "2026-05-31", 77.0, "USD", "OK")


def test_provider_hit_excludes_credentials_from_result_database_and_quote_hash(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "injected.db"
    _init_price_db(db_path)
    cfg = _cfg(tmp_path)
    secret = "TEST-PRICE-KEY-456"
    provider = _FakeProvider(
        PriceSnapshot(
            ticker="AAA",
            as_of_date="2026-05-31",
            price=77.0,
            currency="USD",
            source="eodhd",
            retrieved_at="2026-06-01T12:00:00+00:00",
            url=(
                "https://eodhd.com/api/eod/AAA.US?api_token=TEST-PRICE-KEY-456"
                "&from=2026-05-25&to=2026-05-31&fmt=json"
            ),
            confidence="HIGH",
        )
    )

    result = resolve_v2_price(
        "AAA",
        as_of_date="2026-06-01",
        db_path=db_path,
        cfg=cfg,
        provider=provider,
    )

    assert result.status == "RESOLVED"
    assert result.snapshot is not None
    assert result.snapshot.url == (
        "https://eodhd.com/api/eod/AAA.US?from=2026-05-25&to=2026-05-31&fmt=json"
    )
    assert secret not in json.dumps(result.to_dict(), sort_keys=True)

    conn = sqlite3.connect(str(db_path))
    source_url, raw_json, quote_hash = conn.execute(
        "SELECT source_url, raw_json, quote_hash FROM price_quotes"
    ).fetchone()
    conn.close()
    assert source_url == (
        "https://eodhd.com/api/eod/AAA.US?from=2026-05-25&to=2026-05-31&fmt=json"
    )
    assert secret not in raw_json
    assert quote_hash == sha256(raw_json.encode("utf-8")).hexdigest()


@pytest.mark.parametrize(
    "reason_code",
    ["TIMEOUT", "RATE_LIMIT", "BUDGET_EXHAUSTED", "HTTP_5XX"],
)
def test_retryable_provider_failures_leave_attempt_incomplete(
    tmp_path: Path,
    reason_code: str,
) -> None:
    db_path = tmp_path / "injected.db"
    _init_price_db(db_path)
    provider = _FakeProvider(None, reason_code=reason_code)

    result = resolve_v2_price(
        "AAA",
        as_of_date="2026-06-01",
        db_path=db_path,
        cfg=_cfg(tmp_path),
        provider=provider,
    )

    assert result.status == "INCOMPLETE"
    assert result.reason_code == reason_code
    assert result.retryable is True
    assert result.terminal_for_attempt is False
    assert result.attempts[-1].provider_diagnostic is not None


@pytest.mark.parametrize("reason_code", ["OFFLINE_NO_CACHE", "SYMBOL_NOT_FOUND"])
def test_terminal_provider_misses_become_needs_data(
    tmp_path: Path,
    reason_code: str,
) -> None:
    db_path = tmp_path / "injected.db"
    _init_price_db(db_path)
    provider = _FakeProvider(None, reason_code=reason_code)

    result = resolve_v2_price(
        "AAA",
        as_of_date="2026-06-01",
        db_path=db_path,
        cfg=_cfg(tmp_path),
        provider=provider,
    )

    assert result.status == "NEEDS_DATA"
    assert result.reason_code == reason_code
    assert result.retryable is False
    assert result.terminal_for_attempt is True
    assert result.attempts[-1].terminal_for_attempt is True
