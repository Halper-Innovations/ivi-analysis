"""Commodity context for sector scans.

For sectors whose profitability is driven by commodity prices (energy, metals &
mining, chemicals), the AI needs to know the current commodity price and where
it sits relative to recent history. A scan of oil producers is meaningless
without context on whether WTI is at $50 or $95.

This module:
1. Maps each commodity-sensitive sector to the commodities that matter.
2. Fetches recent futures prices via yfinance.
3. Computes 30/90/180-day averages, ranges, and position-in-range.
4. Formats a text block suitable for injection into scan prompts.
5. Caches to disk (24h TTL) so we don't hit yfinance on every scan.

The AI's world knowledge interprets the data — we give it numbers and ranges,
it figures out what "elevated crude" means for upstream vs midstream vs refiners.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# yfinance symbol, display name, unit
# Futures contracts:
#   CL=F  — WTI crude (NYMEX, $/bbl)
#   BZ=F  — Brent crude (ICE, $/bbl)
#   NG=F  — Natural gas (NYMEX Henry Hub, $/MMBtu)
#   GC=F  — Gold ($/oz)
#   SI=F  — Silver ($/oz)
#   HG=F  — Copper ($/lb)
COMMODITY_CATALOG: dict[str, tuple[str, str]] = {
    "CL=F": ("WTI Crude", "$/bbl"),
    "BZ=F": ("Brent Crude", "$/bbl"),
    "NG=F": ("Natural Gas", "$/MMBtu"),
    "GC=F": ("Gold", "$/oz"),
    "SI=F": ("Silver", "$/oz"),
    "HG=F": ("Copper", "$/lb"),
}


# Which commodities matter for each sector
SECTOR_COMMODITIES: dict[str, list[str]] = {
    "energy": ["CL=F", "BZ=F", "NG=F"],
    "metals_mining": ["GC=F", "SI=F", "HG=F"],
    "chemicals": ["CL=F", "NG=F"],  # oil is feedstock for petrochemicals; nat gas for fertilizer/ammonia
}


# Human-readable analyst notes per commodity-sector combination.
# These are non-controversial framings that help the AI interpret the price
# signal in the context of the specific sector.
SECTOR_COMMODITY_NOTES: dict[str, str] = {
    "energy": (
        "Upstream E&P profits scale almost linearly with crude above breakeven; "
        "elevated oil helps them most. Refiners care about the crude-product spread, "
        "not the absolute oil price. Oilfield services lag the commodity cycle by "
        "2-4 quarters. Natural-gas-heavy E&Ps read off NG=F, not crude. Midstream/"
        "pipelines are volume-driven and less commodity-sensitive."
    ),
    "metals_mining": (
        "Pure-play gold miners benefit directly from gold above all-in sustaining cost; "
        "elevated gold compresses the risk-reward at current equity prices. Copper is a "
        "proxy for global industrial demand — elevated copper signals growth, depressed "
        "copper signals recession risk."
    ),
    "chemicals": (
        "Petrochemicals use oil/naphtha as feedstock — higher crude squeezes margins "
        "unless pricing power passes cost through. Fertilizer/ammonia producers use "
        "natural gas as feedstock; depressed NG helps, elevated NG hurts."
    ),
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CommoditySnapshot:
    """Current price plus 30/90/180-day windows for one commodity."""

    symbol: str
    display_name: str
    unit: str
    as_of: str
    current_price: float

    # Rolling-window stats (30/90/180 calendar days)
    avg_30d: float | None = None
    avg_90d: float | None = None
    avg_180d: float | None = None

    range_30d: tuple[float, float] | None = None
    range_90d: tuple[float, float] | None = None
    range_180d: tuple[float, float] | None = None

    pct_change_30d: float | None = None
    pct_change_90d: float | None = None
    pct_change_180d: float | None = None

    # Where current sits in the 180d range: 0.0 = at 180d low, 1.0 = at 180d high
    position_in_180d_range: float | None = None

    # Qualitative: "elevated" | "neutral" | "depressed"
    # Based on position in 180d range: > 0.75 elevated, < 0.25 depressed, else neutral
    interpretation: str = "neutral"

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Tuples need to become lists for JSON serialization
        for k in ("range_30d", "range_90d", "range_180d"):
            if d[k] is not None:
                d[k] = list(d[k])
        return d


# ---------------------------------------------------------------------------
# Fetch + compute
# ---------------------------------------------------------------------------


def _classify_position(pos: float | None) -> str:
    if pos is None:
        return "neutral"
    if pos >= 0.75:
        return "elevated"
    if pos <= 0.25:
        return "depressed"
    return "neutral"


def compute_snapshot_from_prices(
    *,
    symbol: str,
    prices: list[tuple[str, float]],
    as_of: str | None = None,
) -> CommoditySnapshot | None:
    """Pure function: given (date_str, close_price) pairs, compute the snapshot.

    Kept pure so tests can drive it without hitting yfinance.
    Prices must be sorted ASCENDING by date.
    """
    if not prices:
        return None

    display_name, unit = COMMODITY_CATALOG.get(symbol, (symbol, ""))
    current_price = prices[-1][1]
    current_date = prices[-1][0]
    as_of = as_of or current_date

    def _window(days: int) -> list[float]:
        """Return closes from the last `days` calendar days (approximate via trading days)."""
        # Trading days ≈ days * 5/7. Cap by available length.
        n = min(len(prices), int(days * 5 / 7))
        if n <= 0:
            return []
        return [p[1] for p in prices[-n:]]

    def _avg(window: list[float]) -> float | None:
        return sum(window) / len(window) if window else None

    def _range(window: list[float]) -> tuple[float, float] | None:
        return (min(window), max(window)) if window else None

    w30 = _window(30)
    w90 = _window(90)
    w180 = _window(180)

    def _pct_change(window: list[float]) -> float | None:
        """Current vs window's first close, as a percentage."""
        if not window:
            return None
        first = window[0]
        if first == 0:
            return None
        return (current_price - first) / first * 100.0

    range_180 = _range(w180)
    if range_180 is not None and range_180[1] > range_180[0]:
        low, high = range_180
        position = (current_price - low) / (high - low)
        position = max(0.0, min(1.0, position))
    else:
        position = None

    return CommoditySnapshot(
        symbol=symbol,
        display_name=display_name,
        unit=unit,
        as_of=as_of,
        current_price=current_price,
        avg_30d=_avg(w30),
        avg_90d=_avg(w90),
        avg_180d=_avg(w180),
        range_30d=_range(w30),
        range_90d=_range(w90),
        range_180d=range_180,
        pct_change_30d=_pct_change(w30),
        pct_change_90d=_pct_change(w90),
        pct_change_180d=_pct_change(w180),
        position_in_180d_range=position,
        interpretation=_classify_position(position),
    )


def _cache_path(symbol: str, cache_dir: Path) -> Path:
    # Sanitize symbol for filesystem (CL=F → CL_F)
    safe = symbol.replace("=", "_").replace("/", "_")
    return cache_dir / f"{safe}.json"


def _load_from_cache(
    symbol: str,
    cache_dir: Path,
    ttl_seconds: int,
) -> CommoditySnapshot | None:
    p = _cache_path(symbol, cache_dir)
    if not p.exists():
        return None
    age = time.time() - p.stat().st_mtime
    if age > ttl_seconds:
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    # Reconstruct the snapshot (handle tuple fields stored as lists)
    for k in ("range_30d", "range_90d", "range_180d"):
        if data.get(k) is not None:
            data[k] = tuple(data[k])
    try:
        return CommoditySnapshot(**data)
    except TypeError:
        return None


def _save_to_cache(snapshot: CommoditySnapshot, cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = _cache_path(snapshot.symbol, cache_dir)
    try:
        p.write_text(json.dumps(snapshot.as_dict(), indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("commodity_context: failed to cache %s: %s", snapshot.symbol, exc)


def fetch_commodity_snapshot(
    symbol: str,
    *,
    cache_dir: Path | None = None,
    cache_ttl_seconds: int = 86400,  # 24h
    lookback_days: int = 200,
    as_of: str | None = None,
    yf_ticker_factory=None,
) -> CommoditySnapshot | None:
    """Fetch a commodity snapshot, using disk cache.

    Parameters
    ----------
    symbol : yfinance futures symbol (e.g. "CL=F")
    cache_dir : directory for on-disk cache (default data/cache/commodities)
    cache_ttl_seconds : how long a cache entry is valid
    lookback_days : calendar days of history to fetch
    as_of : optional ISO date to tag the snapshot (default: today)
    yf_ticker_factory : injected for tests; defaults to yfinance.Ticker
    """
    if cache_dir is None:
        cache_dir = Path("data/cache/commodities")

    cached = _load_from_cache(symbol, cache_dir, cache_ttl_seconds)
    if cached is not None:
        logger.debug("commodity_context: cache hit %s", symbol)
        return cached

    # Import lazily so tests that mock yf_ticker_factory don't need yfinance installed.
    if yf_ticker_factory is None:
        try:
            import yfinance as yf

            yf_ticker_factory = yf.Ticker
        except ImportError:
            logger.warning("commodity_context: yfinance not installed — returning None for %s", symbol)
            return None

    try:
        t = yf_ticker_factory(symbol)
        end = datetime.now()
        start = end - timedelta(days=lookback_days)
        hist = t.history(start=start, end=end)
    except Exception as exc:
        logger.warning("commodity_context: fetch failed for %s: %s", symbol, exc)
        return None

    if hist is None or len(hist) == 0:
        logger.warning("commodity_context: no history for %s", symbol)
        return None

    # Build (date_str, close) pairs ordered ascending
    try:
        prices: list[tuple[str, float]] = []
        for ts, row in hist.iterrows():
            close = float(row["Close"])
            if close != close:  # NaN check
                continue
            prices.append((ts.strftime("%Y-%m-%d"), close))
    except Exception as exc:
        logger.warning("commodity_context: parse failed for %s: %s", symbol, exc)
        return None

    snapshot = compute_snapshot_from_prices(symbol=symbol, prices=prices, as_of=as_of)
    if snapshot is not None:
        _save_to_cache(snapshot, cache_dir)
    return snapshot


# ---------------------------------------------------------------------------
# Formatting for prompt injection
# ---------------------------------------------------------------------------


def format_commodity_block(
    sector: str,
    snapshots: list[CommoditySnapshot],
) -> str:
    """Render snapshots as a text block for injection into an AI prompt."""
    if not snapshots:
        return ""

    lines: list[str] = [
        "",
        f"=== COMMODITY CONTEXT (relevant to {sector} sector) ===",
        "",
        "Current commodity prices and where they sit in recent history. Company",
        "profitability in this sector is heavily driven by these prices. Interpret",
        "each company's current financials in light of where commodities are now,",
        "not just the multi-year average.",
        "",
    ]

    for s in snapshots:
        lines.append(f"{s.display_name} ({s.symbol}) — {s.interpretation.upper()}")
        lines.append(f"  Current: {s.current_price:.2f} {s.unit}  (as of {s.as_of})")

        if s.avg_30d is not None and s.range_30d is not None:
            lines.append(
                f"  30d:   avg {s.avg_30d:.2f}, "
                f"range {s.range_30d[0]:.2f} – {s.range_30d[1]:.2f}"
            )
        if s.avg_90d is not None and s.range_90d is not None and s.pct_change_90d is not None:
            lines.append(
                f"  90d:   avg {s.avg_90d:.2f}, "
                f"range {s.range_90d[0]:.2f} – {s.range_90d[1]:.2f}, "
                f"current {s.pct_change_90d:+.1f}% vs 90d-ago"
            )
        if s.avg_180d is not None and s.range_180d is not None and s.pct_change_180d is not None:
            lines.append(
                f"  180d:  avg {s.avg_180d:.2f}, "
                f"range {s.range_180d[0]:.2f} – {s.range_180d[1]:.2f}, "
                f"current {s.pct_change_180d:+.1f}% vs 180d-ago"
            )
        if s.position_in_180d_range is not None:
            pct = s.position_in_180d_range * 100
            lines.append(f"  Position in 180d range: {pct:.0f}% ({s.interpretation})")
        lines.append("")

    note = SECTOR_COMMODITY_NOTES.get(sector)
    if note:
        lines.append("INTERPRETIVE NOTES FOR THIS SECTOR:")
        lines.append(note)
        lines.append("")

    lines.append(
        "Use these prices when judging whether current earnings are sustainable, "
        "whether depressed scorecard signals reflect a cyclical trough, or whether "
        "a company's leverage becomes dangerous at other commodity levels."
    )
    lines.append("")
    return "\n".join(lines)


def get_sector_commodity_context(
    sector: str,
    *,
    cache_dir: Path | None = None,
    cache_ttl_seconds: int = 86400,
    as_of: str | None = None,
    yf_ticker_factory=None,
) -> tuple[str, list[CommoditySnapshot]]:
    """Return (formatted_block, snapshots).

    Returns ("", []) when:
    - sector has no mapped commodities, or
    - all commodity fetches failed.

    Callers inject the block into prompts; callers can also persist the snapshots
    to audit artifacts.
    """
    commodities = SECTOR_COMMODITIES.get(sector, [])
    if not commodities:
        return "", []

    snapshots: list[CommoditySnapshot] = []
    for sym in commodities:
        snap = fetch_commodity_snapshot(
            sym,
            cache_dir=cache_dir,
            cache_ttl_seconds=cache_ttl_seconds,
            as_of=as_of,
            yf_ticker_factory=yf_ticker_factory,
        )
        if snap is not None:
            snapshots.append(snap)

    if not snapshots:
        logger.warning(
            "commodity_context: no snapshots fetched for sector %s — omitting block", sector
        )
        return "", []

    block = format_commodity_block(sector, snapshots)
    return block, snapshots
