import math
from collections.abc import Iterable, Sequence


def hit_rate_at_k(ranked: Sequence[int], relevant: set[int], k: int = 10) -> float:
    return 1.0 if any(item in relevant for item in ranked[:k]) else 0.0


def precision_at_k(ranked: Sequence[int], relevant: set[int], k: int = 10) -> float:
    hits = sum(item in relevant for item in ranked[:k])
    return hits / k if k > 0 else 0.0


def recall_at_k(ranked: Sequence[int], relevant: set[int], k: int = 10) -> float:
    if not relevant:
        raise ValueError("recall is undefined when there are no relevant items")
    hits = sum(item in relevant for item in ranked[:k])
    return hits / len(relevant) if relevant else 0.0


def ndcg_at_k(ranked: Sequence[int], relevant: set[int], k: int = 10) -> float:
    if not relevant:
        raise ValueError("NDCG is undefined when there are no relevant items")
    dcg = sum(
        1.0 / math.log2(position + 2)
        for position, item in enumerate(ranked[:k])
        if item in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(position + 2) for position in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def catalog_coverage(
    all_ranked: Iterable[Sequence[int]], catalog_size: int, k: int = 10
) -> float:
    recommended = {item for ranked in all_ranked for item in ranked[:k]}
    return len(recommended) / catalog_size if catalog_size > 0 else 0.0
