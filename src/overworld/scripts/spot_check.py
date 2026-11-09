"""Print PageRank's top 10 for a few games, by name, on the full graph."""

import pandas as pd
from overworld.config import DATA_DIR
from overworld.models.ppr import PersonalizedPageRank

SEEDS = ["Kingdom Hearts", "Inscryption", "Stardew Valley", "Dishonored", "Lake"]

games = pd.read_parquet(DATA_DIR / "processed" / "games.parquet")
games = games[games["total_rating_count"] >= 10]
edges = pd.read_parquet(DATA_DIR / "processed" / "edges.parquet")
ppr = PersonalizedPageRank(games, edges, exclude_neighbors=False)

for name in SEEDS:
    matches = games.index[games["name"] == name]
    if len(matches) == 0:
        print(f"\n{name}: not in graph")
        continue
    seed = matches[0]
    print(f"\n{games.loc[seed, 'display_name']}:")
    for gid in ppr.recommend(seed):
        print(f"    {games.loc[gid, 'display_name']}")
