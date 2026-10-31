import pandas as pd

from overworld.evaluation.harness import evaluate
from overworld.evaluation.split import Query
from overworld.models.baselines import (
    ContentRecommender,
    IGDBSimilarRecommender,
    PopularityRecommender,
)


def make_games():
    return pd.DataFrame(
        {
            "total_rating_count": [5, 50, 20, 8],
            "themes": [["horror"], ["horror"], ["horror"], ["comedy"]],
            "genres": [[], [], [], []],
            "similar_games": [[2, 999], [1], [], [1, 2]],
        },
        index=[1, 2, 3, 4],
    )


def test_popularity_excludes_seed():
    assert PopularityRecommender(make_games()).recommend(2, k=2) == [3, 4]


def test_content_breaks_ties_by_popularity():
    # games 2 and 3 tie with seed 1 on tags; game 2 has more ratings
    assert ContentRecommender(make_games()).recommend(1, k=1) == [2]


def test_content_never_returns_seed():
    assert 1 not in ContentRecommender(make_games()).recommend(1, k=3)


def test_igdb_similar_drops_unknown_games():
    assert IGDBSimilarRecommender(make_games()).recommend(1) == [
        2
    ]  # 999 isn't in the graph


class FixedRecommender:
    name = "fixed"

    def recommend(self, seed, k=10):
        return [2, 3]


def test_harness_averages_over_queries():
    queries = [Query(1, frozenset({2}), "g"), Query(4, frozenset({9}), "g")]
    result = evaluate(FixedRecommender(), queries, catalog_size=10, k=2)
    assert result["hit_rate"] == 0.5  # first query hits, second misses
    assert result["coverage"] == 0.2  # games 2 and 3, out of 10
