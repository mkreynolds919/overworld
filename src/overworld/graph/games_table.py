import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from overworld.config import DATA_DIR

RAW_DIR = DATA_DIR / "raw" / "igdb" / "games"
OUT_PATH = DATA_DIR / "processed" / "games.parquet"


def load_raw_games(raw_dir: Path) -> list[dict]:
    games = []
    for page in sorted(raw_dir.glob("page_*.json")):
        games.extend(json.loads(page.read_text()))
    return games


def names(items: list[dict] | None) -> list[str]:
    return [item["name"] for item in items or [] if "name" in item]


def release_year(timestamp: int | None) -> int | None:
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).year


def normalize_series(name: str) -> str:
    name = unicodedata.normalize("NFKC", name).lower()
    name = re.sub(r"[^\w\s]", " ", name)
    return re.sub(r"\s+", " ", name).strip()


def flatten_game(game: dict) -> dict:
    year = release_year(game.get("first_release_date"))
    series = {
        normalize_series(s)
        for s in names(game.get("franchises")) + names(game.get("collections"))
    }
    developers = sorted(
        {
            c["company"]["name"]
            for c in game.get("involved_companies", [])
            if c.get("developer") and "company" in c
        }
    )
    return {
        "id": game["id"],
        "name": game["name"],
        "display_name": f"{game['name']} ({year})" if year else game["name"],
        "year": year,
        "game_type": game.get("game_type"),
        "parent_game": game.get("parent_game"),
        "total_rating_count": game.get("total_rating_count", 0),
        "platforms": names(game.get("platforms")),
        "genres": names(game.get("genres")),
        "themes": names(game.get("themes")),
        "series": sorted(series),
        "developers": developers,
        "similar_games": game.get("similar_games", []),
        "cover_image_id": (game.get("cover") or {}).get("image_id"),
    }


def build_games_table(raw_games: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame([flatten_game(g) for g in raw_games])
    df = df.drop_duplicates(subset="id")
    return df.set_index("id", verify_integrity=True)


def main() -> None:
    df = build_games_table(load_raw_games(RAW_DIR))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH)
    print(f"{len(df)} games -> {OUT_PATH}")
    print(df[["display_name", "series", "developers"]].head(10).to_string())


if __name__ == "__main__":
    main()
