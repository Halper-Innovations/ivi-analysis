"""Optional, disposable benchmark quotes for the read-only investor marks command."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

from app.config import AppConfig
from app.market.price_provider import (
    ChainedPriceProvider,
    EODHDProvider,
    PriceProvider,
    PriceSnapshot,
    StooqProvider,
    YahooFinanceProvider,
    get_default_provider,
)


@dataclass(frozen=True)
class BenchmarkPrice:
    ticker: str
    requested_date: str
    as_of_date: str
    price: float
    provider: str
    price_basis: str = "UNADJUSTED"


@dataclass
class BenchmarkRefresh:
    prices: dict[tuple[str, str], BenchmarkPrice] = field(default_factory=dict)
    request_count: int = 0
    request_count_unit: str = "provider anchor calls; HTTP requests not measured"
    failures: list[str] = field(default_factory=list)


class _MemoryPriceCache:
    """Implement the provider cache contract without retaining refreshed quotes."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.snapshots: dict[tuple[str, str, str], PriceSnapshot] = {}

    def path_for_ticker(self, ticker: str) -> Path:
        # Providers use this path only to describe their diagnostics.
        return self.directory / f"{ticker.upper()}.json"

    def load(self, ticker: str, *, requested_as_of_date: str, source: str) -> PriceSnapshot | None:
        return self.snapshots.get((ticker, requested_as_of_date, source))

    def store(
        self,
        ticker: str,
        *,
        requested_as_of_date: str,
        source: str,
        snapshot: PriceSnapshot,
    ) -> PriceSnapshot:
        self.snapshots[(ticker, requested_as_of_date, source)] = snapshot
        return snapshot


def _leaves(provider: PriceProvider) -> list[PriceProvider]:
    if isinstance(provider, ChainedPriceProvider):
        return [leaf for child in provider.providers for leaf in _leaves(child)]
    return [provider]


def _raw_price(provider: PriceProvider, snapshot: PriceSnapshot) -> float | None:
    value = snapshot.raw_price
    if value is None and isinstance(provider, YahooFinanceProvider) and not provider.auto_adjust:
        value = snapshot.price
    if value is None or not math.isfinite(value) or value <= 0:
        return None
    return float(value)


def refresh_benchmarks(cfg: AppConfig, dates: set[str]) -> BenchmarkRefresh:
    """Fetch raw IWM/SPY anchors only when the caller explicitly requests refresh.

    Keep the configured factory's provider order and EODHD credential. Each
    provider anchor call is counted, including failed calls. This is deliberately
    not an HTTP count: Yahoo may make several requests internally, and provider
    diagnostics also count parsing/cache work. No resolver or database is used.
    """
    if str(cfg.price_provider or "").strip().lower() in {"disabled", "off", "none"}:
        raise ValueError("Benchmark refresh refused: the configured price provider is disabled.")
    if str(cfg.net_provider).strip().lower() == "disabled":
        raise ValueError("Benchmark refresh refused: network access is disabled in configuration.")
    requested_dates = sorted(dates)
    for requested in requested_dates:
        if date.fromisoformat(requested).isoformat() != requested:
            raise ValueError("Benchmark anchor dates must use YYYY-MM-DD.")
    result = BenchmarkRefresh()
    if not requested_dates:
        return result

    # The factory constructs disk caches. Give those constructors a disposable
    # directory, then use memory caches for the actual quotes. Redirect db_path
    # defensively as well; the supported provider paths do not open a database.
    with TemporaryDirectory(prefix="ivi-picks-benchmarks-") as temporary:
        directory = Path(temporary)
        provider_cfg = cfg.model_copy(
            update={"cache_dir": directory, "db_path": directory / "unused.sqlite"}
        )
        provider = get_default_provider(provider_cfg, max_retries=0)
        providers = _leaves(provider)
        for leaf in providers:
            if isinstance(leaf, StooqProvider):
                leaf.cache = _MemoryPriceCache(directory)  # type: ignore[assignment]
            if isinstance(leaf, YahooFinanceProvider):
                leaf.auto_adjust = False

        for ticker in ("IWM", "SPY"):
            for requested in requested_dates:
                reasons = []
                for leaf in providers:
                    name = leaf.provider_name
                    if isinstance(leaf, StooqProvider) and not isinstance(leaf, EODHDProvider):
                        reasons.append(f"{name}: RAW_BASIS_UNAVAILABLE")
                        continue
                    result.request_count += 1
                    try:
                        snapshot = leaf.get_price_asof(ticker, requested)
                    except Exception as exc:  # noqa: BLE001
                        # Never include provider exception text, which may contain credentials.
                        reasons.append(f"{name}: {type(exc).__name__}")
                        continue
                    if snapshot is None:
                        reasons.append(f"{name}: NO_DATA")
                        continue
                    try:
                        used_date = date.fromisoformat(snapshot.as_of_date)
                    except ValueError:
                        reasons.append(f"{name}: INVALID_DATE")
                        continue
                    age = (date.fromisoformat(requested) - used_date).days
                    if snapshot.ticker.upper() != ticker or not 0 <= age <= cfg.price_fallback_days:
                        reasons.append(f"{name}: INVALID_ANCHOR")
                        continue
                    if snapshot.currency.upper() != "USD":
                        reasons.append(f"{name}: NON_USD_QUOTE")
                        continue
                    raw = _raw_price(leaf, snapshot)
                    if raw is None:
                        reasons.append(f"{name}: RAW_BASIS_UNAVAILABLE")
                        continue
                    result.prices[(ticker, requested)] = BenchmarkPrice(
                        ticker=ticker,
                        requested_date=requested,
                        as_of_date=snapshot.as_of_date,
                        price=raw,
                        provider=name,
                    )
                    break
                else:
                    detail = "; ".join(reasons) or "NO_CONFIGURED_PROVIDER"
                    result.failures.append(f"{ticker} on {requested}: {detail}")
    return result
