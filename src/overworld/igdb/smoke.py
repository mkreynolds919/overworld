import json

from overworld.config import TOKEN_CACHE, load_credentials
from overworld.igdb.auth import TokenProvider
from overworld.igdb.client import IGDBClient


def build_client() -> IGDBClient:
    client_id, client_secret = load_credentials()
    tokens = TokenProvider(client_id, client_secret, cache_path=TOKEN_CACHE)
    return IGDBClient(client_id, tokens)


def main() -> None:
    client = build_client()

    game = client.query("games", 'fields *; where name = "Kingdom Hearts";')
    print(json.dumps(game, indent=2)[:3000])

    print("\n2) The 3 most-rated games, with expanded references:")
    top = client.query(
        "games",
        "fields name, first_release_date, total_rating_count, platforms.name,"
        " genres.name, themes.name, franchises.name, collections.name,"
        " involved_companies.developer, involved_companies.company.name,"
        " similar_games, cover.image_id;"
        " sort total_rating_count desc; limit 3;",
    )
    print(json.dumps(top, indent=2)[:4000])

    print("\n3) How many games the full fetch would pull, by rating threshold:")
    for n in (1, 5, 20, 50):
        result = client.query("games/count", f"where total_rating_count >= {n};")
        print(f"   total_rating_count >= {n:>2}: {result['count']}")


if __name__ == "__main__":
    main()
