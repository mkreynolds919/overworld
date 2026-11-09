from statistics import mean
import pandas as pd

from overworld.evaluation import metrics as m
from overworld.evaluation.split import Query


def evaluate(
    recommender, queries: list[Query], catalog_size: int, k: int = 10, family: pd.Series | None =None
) -> dict[str, float]:
    scores: dict[str, list[float]] = {
        "hit_rate": [],
        "precision": [],
        "recall": [],
        "ndcg": [],
    }
    all_ranked = []
    for q in queries:
        fetc = k * 3 if family is not None else k
        ranked = recommender.recommend(q.seed, k)
        relevant = q.relevant
        if family is not None:
            seed_fam = family[q.seed]
            ranked = [
                f for f in dict.fromkeys(family[g] for g in ranked) if f != seed_fam
            ]
            relevant = {family[g] for g in relevant} - {seed_fam}
            if not relevant:
                continue
        all_ranked.append(ranked)
        scores["hit_rate"].append(m.hit_rate_at_k(ranked, relevant, k))
        scores["precision"].append(m.precision_at_k(ranked, relevant, k))
        scores["recall"].append(m.recall_at_k(ranked, relevant, k))
        scores["ndcg"].append(m.ndcg_at_k(ranked, relevant, k))
    result = {name: mean(values) for name, values in scores.items()}
    result["coverage"] = m.catalog_coverage(all_ranked, catalog_size, k)
    if not scores["hit_rate"]:
        raise ValueError("no queries left to score after family collapsing")
    result["queries"] = len(scores["hit_rate"])
    return result
