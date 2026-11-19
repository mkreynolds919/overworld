import json

import pandas as pd

from overworld.config import DATA_DIR

GAMES_PATH = DATA_DIR / "processed" / "games.parquet"
OUT_PATH = DATA_DIR / "raw" / "igdb" / "summaries.json"
BATCH = 500


def fetch_summaries(client, ids: list[int], batch: int = BATCH) -> dict[int, str]:
    summaries = {}
    for start in range(0, len(ids), batch):
        chunk = ids[start : start + batch]
        body = f"fields summary; where id = ({','.join(map(str, chunk))}); limit {batch};"
        for game in client.query("games", body):
            if game.get("summary"):
                summaries[game["id"]] = game["summary"]
    return summaries


def main() -> None:
    from overworld.igdb.smoke import build_client

    ids = sorted(int(i) for i in pd.read_parquet(GAMES_PATH).index)
    summaries = fetch_summaries(build_client(), ids)
    OUT_PATH.write_text(json.dumps(summaries))
    print(f"{len(summaries)} of {len(ids)} games have a summary -> {OUT_PATH}")



if __name__ == "__main__":
    main()