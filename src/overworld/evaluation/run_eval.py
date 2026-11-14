import time

import pandas as pd

from overworld.config import DATA_DIR
from overworld.evaluation import metrics as m
from overworld.evaluation.bootstrap import bootstrap_ci, format_ci, paired_difference
from overworld.evaluation.harness import score_queries
from overworld.evaluation.split import (
    eligible_groups,
    hold_out,
    make_queries,
    split_groups,
)
from overworld.graph.families import version_families
from overworld.models.baselines import (
    ContentRecommender,
    IGDBSimilarRecommender,
    PopularityRecommender,
)
from overworld.models.ppr import PersonalizedPageRank

SPLIT = "val"  # switch to "test" exactly once, for the final README table


def main() -> None:
    games = pd.read_parquet(DATA_DIR / "processed" / "games.parquet")
    games = games[games["total_rating_count"] >= 10]
    edges = pd.read_parquet(DATA_DIR / "processed" / "edges.parquet")

    groups = eligible_groups(games)
    splits = split_groups(groups)
    queries = make_queries(groups, splits[SPLIT])
    graph = hold_out(edges, groups, splits[SPLIT])  # NEVER the full edges here
    family = version_families(games)

    models = [
        PopularityRecommender(games),
        ContentRecommender(games),
        IGDBSimilarRecommender(games),
        PersonalizedPageRank(games, graph),
    ]
    per_query, table = {}, {}
    for model in models:
        start = time.perf_counter()
        pq, ranked = score_queries(model, queries, family=family)
        per_query[model.name] = pq
        table[model.name] = {
            **format_ci(bootstrap_ci(pq)).to_dict(),
            "coverage": f"{m.catalog_coverage(ranked, family.nunique()):.3f}",
        }
        print(f"{model.name}: {time.perf_counter() - start:.1f}s")

    print(
        f"\n{SPLIT} split: {len(queries)} queries, k = 10, by version family, 95% CI over series\n"
    )
    print(pd.DataFrame(table).T.to_string())

    ppr = per_query["personalized pagerank"]
    for other in ("igdb similar_games", "same-genre"):
        diff = format_ci(paired_difference(ppr, per_query[other]))
        print(f"\nPageRank − {other}:\n{diff.to_string()}")


if __name__ == "__main__":
    main()
