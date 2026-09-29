from __future__ import annotations

from dataclasses import dataclass

from app.universe.ticker_cik_map import load_ticker_cik_map


class CIKNotFoundError(ValueError):
    pass


@dataclass
class CIKCoverageReport:
    total: int
    resolved: int
    unresolved: list[str]
    resolved_map: dict[str, str]  # ticker -> zero-padded CIK

    @property
    def coverage_pct(self) -> float:
        if self.total == 0:
            return 100.0
        return round(100.0 * self.resolved / self.total, 2)


def resolve(ticker: str) -> str:
    """Return zero-padded 10-digit CIK for ticker, or raise CIKNotFoundError."""
    mapping = load_ticker_cik_map()
    upper = ticker.upper().strip()
    cik = mapping.get(upper)
    if not cik:
        from app.util.http import network_disabled

        if not mapping and network_disabled():
            raise CIKNotFoundError(
                f"No CIK found for ticker '{upper}': offline (VOE_NET_PROVIDER=disabled) "
                "and the SEC ticker list is not cached."
            )
        raise CIKNotFoundError(f"No CIK found for ticker '{upper}'. Run refresh_ticker_cik_cache() to update.")
    return str(cik).zfill(10)


def validate_batch(tickers: list[str]) -> CIKCoverageReport:
    """Resolve all tickers; return coverage report. Never raises — failures go into report."""
    mapping = load_ticker_cik_map()
    resolved_map: dict[str, str] = {}
    unresolved: list[str] = []
    for ticker in tickers:
        upper = ticker.upper().strip()
        cik = mapping.get(upper)
        if cik:
            resolved_map[upper] = str(cik).zfill(10)
        else:
            unresolved.append(upper)
    return CIKCoverageReport(
        total=len(tickers),
        resolved=len(resolved_map),
        unresolved=unresolved,
        resolved_map=resolved_map,
    )
