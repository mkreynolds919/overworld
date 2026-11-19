# Load everything the API serves, once: games, edges, and version families

import pandas as pd

from overworld.config import DATA_DIR, MIN_RATINGS
from overworld.graph.families import version_families

COVER_URL = "https://images.igdb.com/igdb/image/upload/t_cover_big/{image_id}.jpg"


def cover_url(image_id) -> str | None:
    return COVER_URL.format(image_id=image_id) if isinstance(image_id, str) else None


class Catalog:
    def __init__(self, games: pd.DataFrame, edges: pd.DataFrame) -> None:
        self.games = games
        self.edges = edges
        self.family = version_families(games)
        by_rating = games.assign(family=self.family).sort_values(
            "total_rating_count", ascending=False, kind="stable"
        )
        self.representatives = by_rating.groupby("family").head(1) # most-rated version of each game's version family


    @classmethod
    def load(cls) -> "Catalog":
        games = pd.read_parquet(DATA_DIR / "processed" / "games.parquet")
        games = games[games["total_rating_count"] >= MIN_RATINGS]
        edges = pd.read_parquet(DATA_DIR / "processed" / "edges.parquet")
        return cls(games, edges)

    def has(self, game_id: int) -> bool:
        return game_id in self.games.index

    def search(self, query: str, limit: int) -> list[int]:
        reps = self.representatives
        hits = reps[reps["name"].str.contains(query, case=False, regex=False)]
        return hits.index[:limit].tolist()

    def summary(self, game_id: int) -> dict:
        g = self.games.loc[game_id]
        return {
            "id": int(game_id),
            "display_name": g["display_name"],
            "year": None if pd.isna(g["year"]) else int(g["year"]),
            "cover_url": cover_url(g["cover_image_id"]),
        }

    def detail(self, game_id: int) -> dict:
        g = self.games.loc[game_id]
        members = self.family.index[self.family == self.family[game_id]]
        versions = self.games.loc[members].sort_values("year", kind="stable")
        summary = g.get("summary")
        return {
            **self.summary(game_id),
            "name": g["name"],
            "platforms": [str(x) for x in g["platforms"]],
            "genres": [str(x) for x in g["genres"]],
            "themes": [str(x) for x in g["themes"]],
            "series": [str(x) for x in g["series"]],
            "developers": [str(x) for x in g["developers"]],
            "versions": [{"id": int(v), "display_name": versions.loc[v, "display_name"]} for v in versions.index],
            "summary": summary if isinstance(summary, str) else None,
        }