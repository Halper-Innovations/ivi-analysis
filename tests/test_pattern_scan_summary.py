from __future__ import annotations

from app.patterns.scanner import count_patterns_with_hits
from app.patterns.schemas import PatternHit, PatternResult, PatternScanReport


def test_count_patterns_with_hits_counts_distinct_pattern_ids() -> None:
    report = PatternScanReport(
        run_id="pattern_summary_test",
        scan_date="2026-03-23T00:00:00Z",
        peer_set_size=3,
        peer_set_tickers=["AAA", "BBB", "CCC"],
        pattern_results=[
            PatternResult(
                pattern_id="pattern_a",
                hypothesis="A",
                hit_count=3,
                confirmed_count=0,
                unconfirmed_count=3,
                hit_rate=None,
                sample_size=0,
                hits=[
                    PatternHit(ticker="AAA", pattern_id="pattern_a", years_detected=[2024], detection_strength=0.8),
                    PatternHit(ticker="BBB", pattern_id="pattern_a", years_detected=[2024], detection_strength=0.7),
                    PatternHit(ticker="CCC", pattern_id="pattern_a", years_detected=[2024], detection_strength=0.6),
                ],
            ),
            PatternResult(
                pattern_id="pattern_b",
                hypothesis="B",
                hit_count=1,
                confirmed_count=0,
                unconfirmed_count=1,
                hit_rate=None,
                sample_size=0,
                hits=[
                    PatternHit(ticker="AAA", pattern_id="pattern_b", years_detected=[2024], detection_strength=0.9),
                ],
            ),
        ],
        patterns_with_signal=[],
    )

    assert count_patterns_with_hits(report) == 2
