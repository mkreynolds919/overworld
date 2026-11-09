"""Show a game's tags and every edge it has, strongest first."""

import sys

import pandas as pd

from overworld.config import DATA_DIR

games = pd.read_parquet(DATA_DIR / "processed" / "games.parquet")
edges = pd.read_parquet(DATA_DIR / "processed" / "edges.parquet")

gid = games.index[games["name"] == sys.argv[1]][0]
g = games.loc[gid]
print(f"{g['display_name']}\n  genres: {list(g['genres'])}\n  themes: {list(g['themes'])}")
print(f"  developers: {list(g['developers'])}\n  series: {list(g['series'])}\n")

mine = edges[(edges["src"] == gid) | (edges["dst"] == gid)].copy()
mine["other"] = mine["src"].where(mine["dst"] == gid, mine["dst"])
mine["game"] = mine["other"].map(games["display_name"])
print(mine.sort_values("weight", ascending=False)[["type", "weight", "game"]].to_string(index=False))