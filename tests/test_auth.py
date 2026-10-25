from overworld.igdb import auth
from overworld.igdb.auth import TokenProvider


class FakeClock:
    def __init__(self, now=1_000_000.0):
        self.now = now

    def __call__(self):
        return self.now


def fake_fetch(responses):
    calls = []

    def _fetch(client_id, client_secret):
        calls.append(client_id)
        return responses[len(calls) - 1]

    return _fetch, calls


def test_reuses_token_until_near_expiry(monkeypatch):
    fetch, calls = fake_fetch(
        [
            {"access_token": "first", "expires_in": 1000},
            {"access_token": "second", "expires_in": 1000},
        ]
    )
    monkeypatch.setattr(auth, "fetch_token", fetch)
    clock = FakeClock()
    tp = TokenProvider("id", "secret", clock=clock)

    assert tp.get() == "first"
    clock.now += 600
    assert tp.get() == "first"
    clock.now += 200
    assert tp.get() == "second"
    assert len(calls) == 2


def test_ignores_cache_from_another_app(monkeypatch, tmp_path):
    fetch, _calls = fake_fetch(
        [
            {"access_token": "app1", "expires_in": 5_000_000},
            {"access_token": "app2", "expires_in": 5_000_000},
        ]
    )
    monkeypatch.setattr(auth, "fetch_token", fetch)
    cache = tmp_path / "token.json"

    TokenProvider("app-one", "s", cache_path=cache).get()
    assert TokenProvider("app-two", "s", cache_path=cache).get() == "app2"


def test_invalidate_forces_refresh(monkeypatch):
    fetch, _calls = fake_fetch(
        [
            {"access_token": "old", "expires_in": 5_000_000},
            {"access_token": "new", "expires_in": 5_000_000},
        ]
    )
    monkeypatch.setattr(auth, "fetch_token", fetch)
    tp = TokenProvider("id", "secret")
    tp.get()
    tp.invalidate()
    assert tp.get() == "new"


def test_cache_file_survives_restart(monkeypatch, tmp_path):
    fetch, calls = fake_fetch([{"access_token": "cached", "expires_in": 5_000_000}])
    monkeypatch.setattr(auth, "fetch_token", fetch)
    cache = tmp_path / "token.json"

    TokenProvider("id", "secret", cache_path=cache).get()
    second_run = TokenProvider("id", "secret", cache_path=cache)
    assert second_run.get() == "cached"
    assert len(calls) == 1
