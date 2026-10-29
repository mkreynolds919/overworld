import argparse
import math
from itertools import combinations

import numpy as np
import pandas as pd

from overworld.config import DATA_DIR

GAMES_PATH = DATA_DIR / "processed" / "games.parquet"
EDGES_PATH = DATA_DIR / "processed" / "edges.parquet"

BASE_WEIGHTS = {"version": 1.0, "series": 0.8, "developer": 0.5, "content": 0.3}
DEV_HALF_WEIGHT_GAP = 20


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


def build_edges(games: pd.DataFrame) -> pd.DataFrame:
    parts = {
        "version": version_edges(games),
        "series": group_edges(games, "series", BASE_WEIGHTS["series"]),
        "developer": apply_year_decay(
            group_edges(games, "developers", BASE_WEIGHTS["developer"]), games["year"]
        ),
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
        pd.concat([edges[["src", "weight"]].rename(columns={"src": "id"}),
                   edges[["dst", "weight"]].rename(columns={"dst": "id"})])
        .groupby("id")["weight"].sum()
        .sort_values(ascending=False)
    )
    print(f"\nhighest total edge weight:\n{strength.head(5).rename(games['display_name']).to_string()}")


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
