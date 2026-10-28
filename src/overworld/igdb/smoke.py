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

    print("1) Every field IGDB has for one well-known game:")
    game = client.query(
        "games",
        'fields *; where name = "Kingdom Hearts"; sort total_rating_count desc; limit 1;',
    )
    print(sorted(game[0].keys()))

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

    print("\n4) Game types IGDB uses:")
    for t in client.query("game_types", "fields id, type; sort id asc; limit 50;"):
        print(f"   {t['id']:>2}: {t['type']}")

    keep = "(0,4,8,9,10)"

    print("\n5) Counts with the game-type filter:")
    for n in (5, 10, 20):
        result = client.query(
            "games/count",
            f"where total_rating_count >= {n} & game_type = {keep};",
        )
        print(f"   total_rating_count >= {n:>2}: {result['count']}")

    print("\n6) How remakes/remasters link to their original:")
    versions = client.query(
        "games",
        "fields name, game_type, parent_game.name, version_parent.name;"
        " where game_type = (8,9,10) & total_rating_count >= 50;"
        " sort total_rating_count desc; limit 8;",
    )
    print(json.dumps(versions, indent=2))

    print("\n7) Keywords on some well-known games:")
    sample = client.query(
        "games",
        "fields name, keywords.name;"
        " where game_type = (0,4,8,9,10) & total_rating_count >= 500;"
        " sort total_rating_count desc; limit 10;",
    )
    for g in sample:
        kws = sorted(k["name"] for k in g.get("keywords", []))
        print(f"\n   {g['name']} ({len(kws)} keywords)")
        print(f"   {', '.join(kws[:40])}")


if __name__ == "__main__":
    main()
