from statistics import mean

from overworld.evaluation import metrics as m
from overworld.evaluation.split import Query


def evaluate(
    recommender, queries: list[Query], catalog_size: int, k: int = 10
) -> dict[str, float]:
    scores: dict[str, list[float]] = {
        "hit_rate": [],
        "precision": [],
        "recall": [],
        "ndcg": [],
    }
    all_ranked = []
    for q in queries:
        ranked = recommender.recommend(q.seed, k)
        all_ranked.append(ranked)
        scores["hit_rate"].append(m.hit_rate_at_k(ranked, q.relevant, k))
        scores["precision"].append(m.precision_at_k(ranked, q.relevant, k))
        scores["recall"].append(m.recall_at_k(ranked, q.relevant, k))
        scores["ndcg"].append(m.ndcg_at_k(ranked, q.relevant, k))
    result = {name: mean(values) for name, values in scores.items()}
    result["coverage"] = m.catalog_coverage(all_ranked, catalog_size, k)
    return result
