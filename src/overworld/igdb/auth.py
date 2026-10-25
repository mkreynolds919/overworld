import json
import time
from collections.abc import Callable
from pathlib import Path

import requests

EXPIRY_MARGIN_S = 300
TOKEN_URL = "https://id.twitch.tv/oauth2/token"


class TokenProvider:
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        cache_path: Path | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._cache_path = cache_path
        self._clock = clock
        self._token: str | None = None
        self._expires_at = 0.0
        self._load_cache()

    def get(self) -> str | None:
        if (
            self._token is None or self._clock() >= self._expires_at - EXPIRY_MARGIN_S
        ):  # if token is close to expiring but not expired, still refresh it
            self._refresh()
        return self._token

    def invalidate(
        self,
    ) -> (
        None
    ):  # resets the token and expiration time, forcing a refresh on the next get() call
        self._token = None
        self._expires_at = 0.0

    def _refresh(self) -> None:
        data = fetch_token(self._client_id, self._client_secret)
        self._token = data["access_token"]
        self._expires_at = self._clock() + data["expires_in"]
        self._save_cache()

    def _load_cache(self) -> None:
        if not self._cache_path or not self._cache_path.exists():
            return
        try:
            cached = json.loads(self._cache_path.read_text())
        except json.JSONDecodeError:
            return
        if cached.get("client_id") == self._client_id:
            self._token = cached["access_token"]
            self._expires_at = cached["expires_at"]

    def _save_cache(self) -> None:
        if not self._cache_path:
            return
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(
            json.dumps(
                {
                    "client_id": self._client_id,
                    "access_token": self._token,
                    "expires_at": self._expires_at,
                }
            )
        )

    @property
    def expires_at(self) -> float:
        return self._expires_at


def fetch_token(client_id: str, client_secret: str) -> dict:
    resp = requests.post(
        TOKEN_URL,
        params={
            "client_id": client_id,
            "client_secret": client_secret,
            "grant_type": "client_credentials",
        },
        timeout=30,
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"Token request failed ({resp.status_code}): {resp.text[:200]}"
        )
    return resp.json()
