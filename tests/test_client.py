import json

import requests

import pytest

from overworld.igdb.client import IGDBClient, IGDBError


class FakeResponse:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload) if payload is not None else ""

    def json(self):
        return self._payload


class FakeSession:
    """Replays scripted responses in order and records every request."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeTokens:
    """Stands in for TokenProvider: hands out t1, then t2 after invalidate()."""

    def __init__(self):
        self.tokens = ["t1", "t2"]

    def get(self):
        return self.tokens[0]

    def invalidate(self):
        self.tokens.pop(0)


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.slept = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(round(seconds, 3))
        self.now += seconds


def make_client(responses):
    clock = FakeClock()
    session = FakeSession(responses)
    client = IGDBClient(
        "cid", FakeTokens(), session=session, sleep=clock.sleep, clock=clock
    )
    return client, session, clock


def test_sends_auth_headers_and_body():
    client, session, _ = make_client([FakeResponse(200, [{"id": 1}])])
    assert client.query("games", "fields name;") == [{"id": 1}]
    url, kwargs = session.calls[0]
    assert url == "https://api.igdb.com/v4/games"
    assert kwargs["headers"] == {"Client-ID": "cid", "Authorization": "Bearer t1"}
    assert kwargs["data"] == b"fields name;"


def test_spaces_requests_to_respect_rate_limit():
    client, _, clock = make_client([FakeResponse(200, [])] * 3)
    for _ in range(3):
        client.query("games", "fields name;")
    assert clock.slept == [0.26, 0.26]  # first request immediate, then spaced


def test_retries_busy_server_with_backoff():
    client, _, clock = make_client(
        [FakeResponse(429), FakeResponse(503), FakeResponse(200, [])]
    )
    assert client.query("games", "x;") == []
    assert [s for s in clock.slept if s != 0.26] == [0.5, 1.0]


def test_401_gets_new_token_once():
    client, session, _ = make_client(
        [FakeResponse(401), FakeResponse(200, [{"id": 2}])]
    )
    assert client.query("games", "x;") == [{"id": 2}]
    assert session.calls[1][1]["headers"]["Authorization"] == "Bearer t2"


def test_bad_query_fails_fast_with_igdb_message():
    client, session, _ = make_client([FakeResponse(400, [{"title": "Invalid Field"}])])
    with pytest.raises(IGDBError, match="Invalid Field"):
        client.query("games", "fields nope;")
    assert len(session.calls) == 1  # no pointless retries on own bug


def test_retries_network_errors():
    client, session, _ = make_client(
        [
            requests.ConnectionError("wifi dropped"),
            FakeResponse(200, [{"id": 1}]),
        ]
    )
    assert client.query("games", "x;") == [{"id": 1}]
    assert len(session.calls) == 2


def test_network_error_raises_after_last_retry():
    client, _, _ = make_client([requests.Timeout("slow")] * 6)
    with pytest.raises(requests.Timeout):
        client.query("games", "x;")
