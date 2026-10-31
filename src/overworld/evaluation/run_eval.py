import pandas as pd

from overworld.config import DATA_DIR
from overworld.evaluation.harness import evaluate
from overworld.evaluation.split import eligible_groups, make_queries, split_groups
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

    models = [
        PopularityRecommender(games),
        ContentRecommender(games),
        IGDBSimilarRecommender(games),
    ]
    rows = {model.name: evaluate(model, queries, len(games)) for model in models}
    print(f"{SPLIT} split: {len(queries)} queries, k = 10\n")
    print(pd.DataFrame(rows).T.round(3).to_string())


if __name__ == "__main__":
    main()
