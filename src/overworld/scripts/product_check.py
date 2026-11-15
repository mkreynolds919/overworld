"""Compare product-mode series caps: coverage, how much one series dominates, list length."""

import numpy as np
import pandas as pd

from overworld.config import DATA_DIR
from overworld.evaluation.metrics import catalog_coverage
from overworld.models.ppr import PersonalizedPageRank

games = pd.read_parquet(DATA_DIR / "processed" / "games.parquet")
games = games[games["total_rating_count"] >= 10]
edges = pd.read_parquet(DATA_DIR / "processed" / "edges.parquet")
series = games["series"]
seeds = np.random.default_rng(0).choice(games.index.to_numpy(), size=500, replace=False)


def top_series_share(ranked: list[int]) -> float:
    """Fraction of a list taken by its single most common series."""
    counts: dict[str, int] = {}
    for g in ranked:
        for key in series[g]:
            counts[key] = counts.get(key, 0) + 1
    return max(counts.values(), default=0) / max(len(ranked), 1)


for cap in (None, 3, 2):
    ppr = PersonalizedPageRank(games, edges, exclude_neighbors=True, rec_series_cap=cap)
    lists = [ppr.recommend(s) for s in seeds]
    print(f"cap={cap}: coverage {catalog_coverage(lists, len(games)):.3f}, "
          f"top-series share {np.mean([top_series_share(x) for x in lists]):.2f}, "
          f"avg length {np.mean([len(x) for x in lists]):.1f}")