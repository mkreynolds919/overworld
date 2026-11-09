import pandas as pd
import time

from overworld.config import DATA_DIR
from overworld.evaluation.harness import evaluate
from overworld.evaluation.split import eligible_groups, make_queries, split_groups, hold_out
from overworld.models.ppr import PersonalizedPageRank
from overworld.models.baselines import (
    ContentRecommender,
    IGDBSimilarRecommender,
    PopularityRecommender,
)

SPLIT = "val"  # switch to "test" exactly once, for the final README table


def main() -> None:
    games = pd.read_parquet(DATA_DIR / "processed" / "games.parquet")
    games = games[games["total_rating_count"] >= 10]
    groups = eligible_groups(games)
    queries = make_queries(groups, split_groups(groups)[SPLIT])

    edges = pd.read_parquet(DATA_DIR / "processed" / "edges.parquet")
    splits = split_groups(groups)
    queries = make_queries(groups, splits[SPLIT])
    graph = hold_out(edges, groups, splits[SPLIT])

    models = [
        PopularityRecommender(games),
        ContentRecommender(games),
        IGDBSimilarRecommender(games),
        PersonalizedPageRank(games, graph)
    ]
    rows = {model.name: evaluate(model, queries, len(games)) for model in models}
    print(f"{SPLIT} split: {len(queries)} queries, k = 10\n")
    print(pd.DataFrame(rows).T.round(3).to_string())

    rows = {}
    for model in models:
        start = time.perf_counter()
        rows[model.name] = evaluate(model, queries, len(games))
        print(f"{model.name}: {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
