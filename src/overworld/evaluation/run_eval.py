import time

import pandas as pd

from overworld.config import DATA_DIR
from overworld.evaluation.harness import evaluate
from overworld.evaluation.split import (
    eligible_groups,
    hold_out,
    make_queries,
    split_groups,
)
from overworld.models.baselines import (
    ContentRecommender,
    IGDBSimilarRecommender,
    PopularityRecommender,
)
from overworld.models.ppr import PersonalizedPageRank

from overworld.graph.families import version_families

SPLIT = "val"  # switch to "test" exactly once, for the final README table


def main() -> None:
    games = pd.read_parquet(DATA_DIR / "processed" / "games.parquet")
    games = games[games["total_rating_count"] >= 10]
    edges = pd.read_parquet(DATA_DIR / "processed" / "edges.parquet")

    groups = eligible_groups(games)
    splits = split_groups(groups)
    queries = make_queries(groups, splits[SPLIT])
    graph = hold_out(edges, groups, splits[SPLIT])     # NEVER the full edges here
    family = version_families(games)

    models = [
        PopularityRecommender(games),
        ContentRecommender(games),
        IGDBSimilarRecommender(games),
        PersonalizedPageRank(games, graph),
    ]
    rows = {}
    for model in models:
        start = time.perf_counter()
        rows[model.name] = evaluate(model, queries, family.nunique(), family=family)
        print(f"{model.name}: {time.perf_counter() - start:.1f}s")

    print(f"\n{SPLIT} split: {len(queries)} queries, k = 10, scored by version family\n")
    print(pd.DataFrame(rows).T.round(3).to_string())


if __name__ == "__main__":
    main()
