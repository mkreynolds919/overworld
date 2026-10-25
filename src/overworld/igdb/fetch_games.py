import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from overworld.config import DATA_DIR
from overworld.igdb.client import IGDBClient

OUT_DIR = DATA_DIR / "raw" / "igdb" / "games"
PAGE_SIZE = 500
KEEP_GAME_TYPES = (
    0,
    4,
    8,
    9,
    10,
)  # main game, standalone expansion, remake, remaster, expanded version

FIELDS = [
    "name",
    "slug",
    "first_release_date",
    "total_rating_count",
    "game_type",
    "parent_game",
    "platforms.name",
    "genres.name",
    "themes.name",
    "franchises.name",
    "collections.name",
    "involved_companies.developer",
    "involved_companies.company.name",
    "similar_games",
    "cover.image_id",
]


def build_query(min_ratings: int) -> str:
    types = ",".join(str(t) for t in KEEP_GAME_TYPES)
    return (
        f"fields {', '.join(FIELDS)};"
        f" where total_rating_count >= {min_ratings} & game_type = ({types});"
        " sort id asc;"
    )


def fetch_all(
    client: IGDBClient, out_dir: Path, query: str, page_size: int = PAGE_SIZE
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    page_num, total, fetched, cached = 0, 0, 0, 0
    while True:
        path = out_dir / f"page_{page_num:04d}.json"
        if path.exists():
            page = json.loads(path.read_text())
            cached += 1
        else:
            page = client.query(
                "games", query + f" limit {page_size}; offset {page_num * page_size};"
            )
            path.write_text(json.dumps(page, indent=2))
            fetched += 1
        total += len(page)
        if len(page) < page_size:
            break
        page_num += 1

    manifest = {
        "query": query,
        "pages": page_num + 1,
        "pages_fetched": fetched,
        "pages_from_cache": cached,
        "games": total,
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "IGDB API (igdb.com)",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch games from IGDB API")
    parser.add_argument(
        "--min-ratings",
        type=int,
        default=5,
        help="minimum total_rating_count (default: 5)",
    )
    parser.add_argument(
        "--refresh", action="store_true", help="delete saved pages and redownload"
    )
    args = parser.parse_args()

    if args.refresh and OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    from overworld.igdb.smoke import build_client

    manifest = fetch_all(build_client(), OUT_DIR, build_query(args.min_ratings))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
