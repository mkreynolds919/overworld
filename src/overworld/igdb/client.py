import time
from collections.abc import Callable
from typing import Any

import requests

from overworld.igdb.auth import TokenProvider

BASE_URL = "https://api.igdb.com/v4"
MIN_INTERVAL_S = 0.26  # rate limit: 4 requests per second
RETRY_STATUSES = {429, 500, 502, 503, 504}


# give certain errors a wrapper to designate them as IGDB-specific errors
class IGDBError(RuntimeError):
    pass


class IGDBClient:
    def __init__(
        self,
        client_id: str,
        tokens: TokenProvider,
        session: requests.Session | None = None,
        max_retries: int = 5,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._client_id = client_id
        self._tokens = tokens
        self._session = session or requests.Session()
        self._max_retries = max_retries
        self._sleep = sleep
        self._clock = clock
        self._last_request_time = float("-inf")

    def query(self, endpoint: str, body: str) -> Any:
        refreshed = False
        for attempt in range(self._max_retries + 1):
            self._throttle()
            try:
                resp = self._session.post(
                    f"{BASE_URL}/{endpoint}",
                    headers={
                        "Client-ID": self._client_id,
                        "Authorization": f"Bearer {self._tokens.get()}",
                    },
                    data=body.encode(),
                    timeout=60,
                )
            except (requests.ConnectionError, requests.Timeout):
                if attempt < self._max_retries:
                    self._sleep(0.5 * 2**attempt)
                    continue
                raise
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 401 and not refreshed:
                self._tokens.invalidate()
                refreshed = True
                continue
            if resp.status_code in RETRY_STATUSES and attempt < self._max_retries:
                self._sleep(0.5 * 2**attempt)
                continue
            raise IGDBError(f"{endpoint}: HTTP {resp.status_code}: {resp.text[:300]}")
        raise IGDBError(f"{endpoint}: exceeded max retries ({self._max_retries})")

    def _throttle(self) -> None:
        wait = self._last_request_time + MIN_INTERVAL_S - self._clock()
        if wait > 0:
            self._sleep(wait)
        self._last_request_time = self._clock()
