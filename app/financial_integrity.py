"""Application-level financial-integrity contract.

This neutral import surface lets independent product engines consume the
deterministic unit, quote, split, and formula validator without importing an
autonomous/Classic pipeline module directly.
"""

from app.autonomous.financial_integrity import (
    MARKET_CAP_UNIT_USD_MILLIONS,
    PRICE_BASIS_SPLIT_ADJUSTED,
    PRICE_BASIS_UNADJUSTED,
    PRICE_UNIT_USD_PER_SHARE,
    SHARES_BASIS_ISSUER_REPORTED,
    SHARES_BASIS_UNADJUSTED,
    SHARES_UNIT_MILLIONS,
    FinancialIntegrityScope,
    InvalidFinancialInputError,
    canonical_metric_trace,
    require_financial_integrity_scope,
    require_unchanged_financial_integrity_scope,
    stable_quote_hash,
)

__all__ = [
    "MARKET_CAP_UNIT_USD_MILLIONS",
    "PRICE_BASIS_SPLIT_ADJUSTED",
    "PRICE_BASIS_UNADJUSTED",
    "PRICE_UNIT_USD_PER_SHARE",
    "SHARES_BASIS_ISSUER_REPORTED",
    "SHARES_BASIS_UNADJUSTED",
    "SHARES_UNIT_MILLIONS",
    "FinancialIntegrityScope",
    "InvalidFinancialInputError",
    "canonical_metric_trace",
    "require_financial_integrity_scope",
    "require_unchanged_financial_integrity_scope",
    "stable_quote_hash",
]
