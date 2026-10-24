import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
TOKEN_CACHE = DATA_DIR / ".twitch_token.json"


def load_credentials() -> tuple[str, str]:
    load_dotenv(REPO_ROOT / ".env")
    client_id = os.getenv("TWITCH_CLIENT_ID", "").strip()
    client_secret = os.getenv("TWITCH_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        raise SystemExit("Missing TWITCH_CLIENT_ID or TWITCH_CLIENT_SECRET in .env")
    return client_id, client_secret
