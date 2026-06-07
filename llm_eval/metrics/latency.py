"""
Latency metric utilities for LLM evaluation.

Provides percentile computation, SLA violation detection,
and time-to-first-token estimation helpers.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass


@dataclass
class LatencyStats:
    """Descriptive statistics for a set of latency measurements."""

    count: int
    mean_ms: float
    median_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p75_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float


class LatencyMetric:
    """Compute and analyse latency statistics for evaluation runs."""

    def compute_stats(self, latencies_ms: list[float]) -> LatencyStats:
        """Return a full statistical summary of latency measurements."""
        if not latencies_ms:
            return LatencyStats(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        n = len(latencies_ms)
        sorted_lat = sorted(latencies_ms)

        def pct(p: float) -> float:
            idx = int(n * p / 100)
            return sorted_lat[min(idx, n - 1)]

        return LatencyStats(
            count=n,
            mean_ms=statistics.mean(latencies_ms),
            median_ms=statistics.median(latencies_ms),
            std_ms=statistics.stdev(latencies_ms) if n > 1 else 0.0,
            min_ms=sorted_lat[0],
            max_ms=sorted_lat[-1],
            p50_ms=pct(50),
            p75_ms=pct(75),
            p90_ms=pct(90),
            p95_ms=pct(95),
            p99_ms=pct(99),
        )

    def sla_violation_rate(
        self, latencies_ms: list[float], threshold_ms: float = 5000.0
    ) -> float:
        """Return fraction of requests exceeding the SLA threshold."""
        if not latencies_ms:
            return 0.0
        violations = sum(1 for ms in latencies_ms if ms > threshold_ms)
        return violations / len(latencies_ms)

    def classify(self, avg_ms: float) -> str:
        """Classify average latency into a human-readable tier."""
        if avg_ms < 500:
            return "excellent"
        if avg_ms < 1500:
            return "good"
        if avg_ms < 3000:
            return "acceptable"
        if avg_ms < 6000:
            return "slow"
        return "very_slow"
