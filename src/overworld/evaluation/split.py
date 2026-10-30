from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from overworld.config import DATA_DIR

SPLIT_SEED = 7
MIN_GROUP, MAX_GROUP = 3, 30
HIDDEN_TYPES = ("series", "version")


@dataclass(frozen=True)
class Query:
    seed: int
    relevant: frozenset[int]
    group: str


def eligible_groups(games: pd.DataFrame) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for game_id, keys in games["series"].items():
        for key in keys:
            groups.setdefault(key, []).append(game_id)
    return {k: sorted(v) for k, v in groups.items() if MIN_GROUP <= len(v) < MAX_GROUP}


def split_groups(
    groups: dict[str, list[int]],
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    seed: int = SPLIT_SEED,
) -> dict[str, list[str]]:
    names = sorted(groups)
    np.random.default_rng(seed).shuffle(names)
    n_val, n_test = round(len(names) * val_frac), round(len(names) * test_frac)
    return {
        "val": names[:n_val],
        "test": names[n_val : n_val + n_test],
        "train": names[n_val + n_test :],
    }


def hold_out(
    edges: pd.DataFrame, groups: dict[str, list[int]], held: list[str]
) -> pd.DataFrame:
    hidden = {pair for g in held for pair in combinations(groups[g], 2)}
    in_group = pd.Series(list(zip(edges["src"], edges["dst"])), index=edges.index).isin(
        hidden
    )
    return edges[~(in_group & edges["type"].isin(HIDDEN_TYPES))]


def make_queries(groups: dict[str, list[int]], held: list[str]) -> list[Query]:
    return [
        Query(seed=game, relevant=frozenset(set(groups[g]) - {game}), group=g)
        for g in held
        for game in groups[g]
    ]


def main() -> None:
    games = pd.read_parquet(DATA_DIR / "processed" / "games.parquet")
    games = games[games["total_rating_count"] >= 10]
    edges = pd.read_parquet(DATA_DIR / "processed" / "edges.parquet")

    groups = eligible_groups(games)
    splits = split_groups(groups)
    for name in ("val", "test"):
        kept = hold_out(edges, groups, splits[name])
        queries = make_queries(groups, splits[name])
        removed = edges.groupby("type").size() - kept.groupby("type").size()
        print(
            f"{name}: {len(splits[name])} series, {len(queries)} queries, "
            f"removed {removed.fillna(0).astype(int).to_dict()}"
        )


if __name__ == "__main__":
    main()
