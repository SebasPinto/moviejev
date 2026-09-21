from __future__ import annotations

import math


def ndcg_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    """Binary-relevance NDCG@k. `ranked` are item ids in reranker order."""
    if k <= 0 or not relevant:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 2) for i, m in enumerate(ranked[:k]) if m in relevant)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0


def hit_rate_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    return 1.0 if any(m in relevant for m in ranked[:k]) else 0.0


def precision_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    if k <= 0:
        return 0.0
    return sum(1 for m in ranked[:k] if m in relevant) / k


def mrr(ranked: list[int], relevant: set[int]) -> float:
    for i, m in enumerate(ranked):
        if m in relevant:
            return 1.0 / (i + 1)
    return 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile, p in [0, 100]. Good enough for latency summaries."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100.0 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]
