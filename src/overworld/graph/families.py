# Groups each game with its remakes, remasters, and editions
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components


def version_families(games: pd.DataFrame) -> pd.Series:
    # Map each game id to a family id, which is found through a game's parent_game
    ids = games.index.to_numpy()
    n = len(ids)
    if "parent_game" not in games:
        return pd.Series(np.arange(n), index=ids, name="family")

    pos = pd.Series(np.arange(n), index=ids)
    parents = games["parent_game"].dropna().astype(int)
    parents = parents[parents.isin(pos.index)]
    rows = parents.index.map(pos).to_numpy()
    cols = parents.map(pos).to_numpy()
    links = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    _, labels = connected_components(links, directed=False)
    return pd.Series(labels, index=ids, name="family")
