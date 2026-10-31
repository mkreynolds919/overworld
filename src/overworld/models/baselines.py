import numpy as np
import pandas as pd

from overworld.graph.edges import tag_matrix


class PopularityRecommender:
    name = "popularity"

    def __init__(self, games: pd.DataFrame) -> None:
        self._ranked = games.sort_values(
            "total_rating_count", ascending=False
        ).index.to_numpy()

    def recommend(self, seed: int, k: int = 10) -> list[int]:
        return [g for g in self._ranked[: k + 1] if g != seed][:k]


class ContentRecommender:
    name = "same-genre"

    def __init__(self, games: pd.DataFrame) -> None:
        self._ids = games.index.to_numpy()
        self._pos = {g: i for i, g in enumerate(self._ids)}
        self._X = tag_matrix(games).tocsr()
        self._pop = games["total_rating_count"].to_numpy()

    def recommend(self, seed: int, k: int = 10) -> list[int]:
        i = self._pos[seed]
        sims = (self._X[i] @ self._X.T).toarray().ravel()
        sims[i] = -1.0
        order = np.lexsort((-self._pop, -sims))
        return self._ids[order[:k]].tolist()


class IGDBSimilarRecommender:
    name = "igdb similar_games"

    def __init__(self, games: pd.DataFrame) -> None:
        valid = set(games.index)
        self._lists = {
            g: [s for s in sims if s in valid and s != g]
            for g, sims in games["similar_games"].items()
        }

    def recommend(self, seed: int, k: int = 10) -> list[int]:
        return self._lists.get(seed, [])[:k]
