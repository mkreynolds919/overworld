import argparse
import math
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import sparse

from overworld.config import DATA_DIR
from overworld.graph.families import version_families

GAMES_PATH = DATA_DIR / "processed" / "games.parquet"
EDGES_PATH = DATA_DIR / "processed" / "edges.parquet"

BASE_WEIGHTS = {"version": 1.0, "series": 0.8, "developer": 0.5, "content": 0.15}
DEV_HALF_WEIGHT_GAP = 20

TAG_WEIGHTS = {"themes": 1.0, "genres": 0.5}
CONTENT_TOP_N = 10
CONTENT_REL_THRESHOLD = 0.3
CHUNK_ROWS = 1000
TIEBREAK_SEED = 12

SERIES_CAP = 2  # content links can only go to 2 members of a given other series
POOL_SIZE = 100  # num candidates considered before diversifying


def group_edges(games: pd.DataFrame, column: str, base: float) -> pd.DataFrame:
    groups: dict[str, list[int]] = {}
    for game_id, keys in games[column].items():
        for key in keys:
            groups.setdefault(key, []).append(game_id)

    best: dict[tuple[int, int], float] = {}
    for ids in groups.values():
        if len(ids) < 2:
            continue
        weight = base / math.log(len(ids) + 1)
        for pair in combinations(sorted(ids), 2):
            if weight > best.get(pair, 0.0):
                best[pair] = weight

    return pd.DataFrame(
        [(a, b, w) for (a, b), w in best.items()], columns=["src", "dst", "weight"]
    )


def apply_year_decay(edges: pd.DataFrame, years: pd.Series) -> pd.DataFrame:
    gap = (edges["src"].map(years) - edges["dst"].map(years)).abs()
    factor = (1 / (1 + gap / DEV_HALF_WEIGHT_GAP)).fillna(1.0)
    return edges.assign(weight=edges["weight"] * factor)


def version_edges(games: pd.DataFrame) -> pd.DataFrame:
    parents = games["parent_game"].dropna().astype(int)
    parents = parents[parents.isin(games.index) & (parents != parents.index)]
    pairs = np.sort(np.column_stack([parents.index, parents.to_numpy()]), axis=1)
    edges = pd.DataFrame(pairs, columns=["src", "dst"]).drop_duplicates()
    return edges.assign(weight=BASE_WEIGHTS["version"])


def tag_matrix(games: pd.DataFrame) -> sparse.csr_matrix:
    rows, cols, vals = [], [], []
    vocab: dict[str, int] = {}
    for column, type_weight in TAG_WEIGHTS.items():
        for i, tags in enumerate(games[column]):
            for tag in tags:
                j = vocab.setdefault(f"{column}:{tag}", len(vocab))
                rows.append(i)
                cols.append(j)
                vals.append(type_weight)
    X = sparse.csr_matrix((vals, (rows, cols)), shape=(len(games), len(vocab)))

    games_per_tag = np.bincount(X.indices, minlength=X.shape[1])
    idf = np.log(len(games) / games_per_tag)
    X = X @ sparse.diags(idf)

    norms = np.sqrt(X.multiply(X).sum(axis=1)).A1
    norms[norms == 0] = 1.0
    return sparse.diags(1 / norms) @ X


def diversify(
    candidates: np.ndarray,
    k: int,
    series: np.ndarray,
    family: np.ndarray,
    self_index: int,
    cap: int = SERIES_CAP,
) -> np.ndarray:
    # Pick up to k candidates in order, at most 'cap' per other series and one per version family
    own_series = set(series[self_index])
    seen_families = {family[self_index]}
    used: dict[str, int] = {}
    picked = []
    for j in candidates:
        if len(picked) == k:
            break
        if family[j] in seen_families:
            continue
        others = [s for s in series[j] if s not in own_series]
        if any(used.get(s, 0) >= cap for s in others):
            continue
        picked.append(j)
        seen_families.add(family[j])
        for s in others:
            used[s] = used.get(s, 0) + 1
    return np.array(picked, dtype=int)


def cap_final_edges(
    edges: pd.DataFrame, series: pd.Series, family: pd.Series, cap: int = SERIES_CAP
) -> pd.DataFrame:
    # Apply series cap both ways for content edges
    both_ends = pd.concat([
        edges.assign(game=edges["src"], other=edges["dst"]),
        edges.assign(game=edges["dst"], other=edges["src"]),
    ]).sort_values("weight", ascending=False, kind="stable")

    rejected: set[tuple[int, int]] = set()
    for game, rows in both_ends.groupby("game", sort=False):
        own_series = set(series[game])
        seen_families = {family[game]}
        used: dict[str, int] = {}
        for src, dst, other in zip(rows["src"], rows["dst"], rows["other"]):
            others = [s for s in series[other] if s not in own_series]
            if family[other] in seen_families or any(used.get(s, 0) >= cap for s in others):
                rejected.add((src, dst))
                continue
            seen_families.add(family[other])
            for s in others:
                used[s] = used.get(s, 0) + 1

    keep = [(s, d) not in rejected for s, d in zip(edges["src"], edges["dst"])]
    return edges[keep].reset_index(drop=True)


def select_neighbors(
    row: np.ndarray,
    top_n: int,
    rel_threshold: float,
    tiebreak=None,
    series=None,
    family=None,
    self_index=None,
) -> np.ndarray:
    k = min(top_n, len(row) - 1)
    if k <= 0:
        return np.array([], dtype=int)
    score = row if tiebreak is None else row + 1e-6 * tiebreak
    if series is None:
        top = np.argpartition(score, -k)[-k:]
    else:
        pool = min(POOL_SIZE, len(row) - 1)
        cand = np.argpartition(score, -pool)[-pool:]
        cand = cand[np.argsort(-score[cand], kind="stable")]
        cand = cand[row[cand] > 0]
        top = diversify(cand, k, series, family, self_index)
    top = top[row[top] > 0]
    if len(top) == 0:
        return top
    return top[row[top] >= rel_threshold * row[top].max()]


def content_edges(games: pd.DataFrame, diversify_series: bool = False) -> pd.DataFrame:
    X = tag_matrix(games).tocsr()
    ids = games.index.to_numpy()
    rng = np.random.default_rng(TIEBREAK_SEED)
    series = games["series"].to_numpy() if diversify_series else None
    family = version_families(games).to_numpy() if diversify_series else None
    best: dict[tuple[int, int], float] = {}

    for start in range(0, X.shape[0], CHUNK_ROWS):
        sims = (X[start : start + CHUNK_ROWS] @ X.T).toarray()
        for r, row in enumerate(sims):
            i = start + r
            row[i] = 0.0
            noise = rng.random(len(row))
            picks = select_neighbors(
                row,
                CONTENT_TOP_N,
                CONTENT_REL_THRESHOLD,
                noise,
                series=series,
                family=family,
                self_index=i,
            )
            for j in picks:
                pair = (min(ids[i], ids[j]), max(ids[i], ids[j]))
                best[pair] = max(best.get(pair, 0.0), BASE_WEIGHTS["content"] * row[j])

    content = pd.DataFrame(
        [(a, b, w) for (a, b), w in best.items()], columns=["src", "dst", "weight"]
    )
    if diversify_series:
        content = cap_final_edges(content, games["series"], version_families(games))
    return content


def build_edges(games: pd.DataFrame) -> pd.DataFrame:
    parts = {
        "version": version_edges(games),
        "series": group_edges(games, "series", BASE_WEIGHTS["series"]),
        "developer": apply_year_decay(
            group_edges(games, "developers", BASE_WEIGHTS["developer"]), games["year"]
        ),
        "content": content_edges(games, diversify_series=True),
    }
    edges = pd.concat(
        [df.assign(type=edge_type) for edge_type, df in parts.items()],
        ignore_index=True,
    )
    assert (edges["src"] < edges["dst"]).all()
    return edges


def print_stats(games: pd.DataFrame, edges: pd.DataFrame) -> None:
    print(f"{len(games)} games, {len(edges)} edges")
    print(edges.groupby("type")["weight"].describe()[["count", "mean", "min", "max"]])
    degree = pd.concat([edges["src"], edges["dst"]]).value_counts()
    print(f"\nisolated games (no edges yet): {len(games) - len(degree)}")
    print(f"median degree: {degree.median():.0f}, max: {degree.max()}")
    top = degree.head(5).rename(games["display_name"])
    print(f"most connected:\n{top.to_string()}")
    strength = (
        pd.concat(
            [
                edges[["src", "weight"]].rename(columns={"src": "id"}),
                edges[["dst", "weight"]].rename(columns={"dst": "id"}),
            ]
        )
        .groupby("id")["weight"]
        .sum()
        .sort_values(ascending=False)
    )
    print(
        f"\nhighest total edge weight:\n{strength.head(5).rename(games['display_name']).to_string()}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build typed edges between games.")
    parser.add_argument(
        "--min-ratings",
        type=int,
        default=10,
        help="keep games with at least this many ratings",
    )
    args = parser.parse_args()

    games = pd.read_parquet(GAMES_PATH)
    games = games[games["total_rating_count"] >= args.min_ratings]
    edges = build_edges(games)
    edges.to_parquet(EDGES_PATH, index=False)
    print_stats(games, edges)


if __name__ == "__main__":
    main()
