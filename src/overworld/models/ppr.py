import numpy as np
import pandas as pd
from scipy import sparse

from overworld.graph.families import version_families

RESTART_PROB = 0.15
TOL = 1e-6
MAX_ITER = 200
REC_SERIES_CAP = 3


def build_transition(
    edges: pd.DataFrame, ids: np.ndarray, type_weights: dict[str, float] | None = None
) -> tuple[sparse.csr_matrix, np.ndarray]:

    pos = pd.Series(np.arange(len(ids)), index=ids)
    w = edges["weight"].to_numpy(dtype=float)
    if type_weights is not None:
        w = w * edges["type"].map(type_weights).fillna(0.0).to_numpy()
    src = edges["src"].map(pos).to_numpy()
    dst = edges["dst"].map(pos).to_numpy()
    n = len(ids)

    A = sparse.coo_matrix(
        (
            np.concatenate([w, w]),
            (np.concatenate([src, dst]), np.concatenate([dst, src])),
        ),
        shape=(n, n),
    ).tocsr()

    out = np.asarray(A.sum(axis=1)).ravel()
    dangling = out == 0
    inv = np.zeros(n)
    inv[~dangling] = 1.0 / out[~dangling]
    return (sparse.diags(inv) @ A).tocsr(), dangling


class PersonalizedPageRank:
    name = "personalized pagerank"

    def __init__(
        self,
        games: pd.DataFrame,
        edges: pd.DataFrame,
        restart_prob: float = RESTART_PROB,
        type_weights: dict[str, float] | None = None,
        exclude_neighbors: bool = False,
        rec_series_cap: int | None = None,
        name="personalized pagerank",
    ) -> None:
        self.name = name
        self._ids = games.index.to_numpy()
        self._pos = {g: i for i, g in enumerate(self._ids)}
        self._P, self._dangling = build_transition(edges, self._ids, type_weights)
        self._PT = self._P.T.tocsr()
        self._alpha = restart_prob
        self._exclude_neighbors = exclude_neighbors
        self._family = version_families(games).to_numpy()
        self._series = games["series"].to_numpy() if "series" in games else None
        self._rec_series_cap = rec_series_cap

    def scores(self, seed: int) -> np.ndarray:
        s = self._pos[seed]
        r = np.zeros(len(self._ids))
        r[s] = 1.0
        for _ in range(MAX_ITER):
            nxt = (1 - self._alpha) * (self._PT @ r)
            nxt[s] += self._alpha + (1 - self._alpha) * r[self._dangling].sum()
            if np.abs(nxt - r).sum() < TOL:
                return nxt
            r = nxt
        return r

    def recommend(self, seed: int, k: int = 10) -> list[int]:
        r = self.scores(seed)
        s = self._pos[seed]
        r[s] = 0.0
        if self._exclude_neighbors:
            r[self._P[s].indices] = 0.0
        top = np.argsort(-r, kind="stable")[:k]
        top = top[r[top] > 0]
        order = np.argsort(-r, kind="stable")
        seen = {self._family[s]}
        used: dict[str, int] = {}
        capping = self._rec_series_cap is not None and self._series is not None
        out = []
        for i in order:
            if r[i] <= 0 or len(out) == k:
                break
            if self._family[i] in seen:
                continue
            keys = self._series[i] if capping else []
            if any(used.get(key, 0) >= self._rec_series_cap for key in keys):
                continue
            seen.add(self._family[i])
            for key in keys:
                used[key] = used.get(key, 0) + 1
            out.append(self._ids[i])
        return out


def product_recommender(games: pd.DataFrame, edges: pd.DataFrame) -> PersonalizedPageRank:
    """The settings the app uses: only games not already connected, at most 3 per series."""
    return PersonalizedPageRank(
        games, edges, exclude_neighbors=True, rec_series_cap=REC_SERIES_CAP, name="ppr (product)"
    )