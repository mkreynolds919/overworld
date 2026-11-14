import pandas as pd

from overworld.evaluation import metrics as m
from overworld.evaluation.split import Query

METRICS = ["hit_rate", "precision", "recall", "ndcg"]


def score_queries(
    recommender, queries: list[Query], k: int = 10, family: pd.Series | None = None
) -> tuple[pd.DataFrame, list[list[int]]]:
    """One row per scored query (seed, group, four metrics), plus every ranked list."""
    rows, all_ranked = [], []
    for q in queries:
        fetch = k * 3 if family is not None else k
        ranked = recommender.recommend(q.seed, fetch)
        relevant = q.relevant
        if family is not None:
            seed_fam = family[q.seed]
            ranked = [
                f for f in dict.fromkeys(family[g] for g in ranked) if f != seed_fam
            ][:k]
            relevant = {family[g] for g in relevant} - {seed_fam}
            if not relevant:
                continue
        all_ranked.append(ranked)
        rows.append(
            {
                "seed": q.seed,
                "group": q.group,
                "hit_rate": m.hit_rate_at_k(ranked, relevant, k),
                "precision": m.precision_at_k(ranked, relevant, k),
                "recall": m.recall_at_k(ranked, relevant, k),
                "ndcg": m.ndcg_at_k(ranked, relevant, k),
            }
        )
    return pd.DataFrame(rows), all_ranked


def evaluate(
    recommender,
    queries: list[Query],
    catalog_size: int,
    k: int = 10,
    family: pd.Series | None = None,
) -> dict[str, float]:
    per_query, all_ranked = score_queries(recommender, queries, k, family)
    if per_query.empty:
        raise ValueError("no queries left to score after family collapsing")
    result = per_query[METRICS].mean().to_dict()
    result["coverage"] = m.catalog_coverage(all_ranked, catalog_size, k)
    result["queries"] = len(per_query)
    return result
