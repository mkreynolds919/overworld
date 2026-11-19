import pandas as pd
import pytest
from fastapi.testclient import TestClient

from overworld.api.app import create_app
from overworld.api.catalog import Catalog


@pytest.fixture
def client():
    games = pd.DataFrame(
        {
            "name": ["Bloodborne", "Dark Souls", "Dark Souls Remastered", "Forza"],
            "display_name": ["Bloodborne (2015)", "Dark Souls (2011)",
                             "Dark Souls Remastered (2018)", "Forza"],
            "year": [2015, 2011, 2018, None],
            "total_rating_count": [3000, 2500, 900, 50],
            "parent_game": [None, None, 2, None],
            "platforms": [["PS4"], ["PS3"], ["PS4"], ["Xbox"]],
            "genres": [["RPG"], ["RPG"], ["RPG"], ["Racing"]],
            "themes": [["Horror"], ["Fantasy"], ["Fantasy"], []],
            "series": [[], ["dark souls"], ["dark souls"], []],
            "developers": [["FromSoftware"]] * 3 + [["Turn 10"]],
            "cover_image_id": ["co1", "co2", None, "co4"],
        },
        index=pd.Index([1, 2, 3, 4], name="id"),
    )
    edges = pd.DataFrame(columns=["src", "dst", "type", "weight"])
    return TestClient(create_app(Catalog(games, edges)))


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "games": 4}


def test_search_returns_one_game_per_family(client):
    names = [g["display_name"] for g in client.get("/games/search", params={"q": "dark"}).json()]
    assert names == ["Dark Souls (2011)"]                   # the remaster is the same family


def test_search_is_case_insensitive(client):
    assert len(client.get("/games/search", params={"q": "BLOOD"}).json()) == 1


def test_detail_lists_all_versions_oldest_first(client):
    body = client.get("/games/2").json()
    assert [v["id"] for v in body["versions"]] == [2, 3]
    assert body["cover_url"].endswith("/co2.jpg")


def test_missing_year_and_cover_become_null(client):
    assert client.get("/games/4").json()["year"] is None
    assert client.get("/games/3").json()["cover_url"] is None


def test_unknown_game_is_404(client):
    assert client.get("/games/999").status_code == 404


def test_bad_inputs_are_422(client):
    assert client.get("/games/search", params={"q": ""}).status_code == 422
    assert client.get("/games/search", params={"q": "a", "limit": 0}).status_code == 422
    assert client.get("/games/abc").status_code == 422