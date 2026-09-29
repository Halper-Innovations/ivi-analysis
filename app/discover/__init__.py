"""Discover funnel package.

Four-stage architecture for scanning thousands of tickers and surfacing
undervalued fundamental-analysis candidates:

- Stage 1: deterministic scorecards (already built in app.valuation)
- Stage 2: Haiku 4.5 triage classifier
- Stage 3: Sonnet 4.6 mid-depth research
- Stage 4: Sonnet 4.6 deep loop with tool use

See docs/superpowers/specs/2026-04-11-discover-funnel-architecture.md
for the design and measured economics.
"""
