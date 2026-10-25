import json

from overworld.igdb.fetch_games import build_query, fetch_all


class FakeClient:
    """Serves pre-made pages and records each query it receives."""

    def __init__(self, pages):
        self.pages = list(pages)
        self.queries = []

    def query(self, endpoint, body):
        self.queries.append(body)
        return self.pages.pop(0)


def test_query_filters_and_sorts():
    q = build_query(min_ratings=10)
    assert "total_rating_count >= 10" in q
    assert "game_type = (0,4,8,9,10)" in q
    assert "sort id asc;" in q


def test_fetches_until_short_page(tmp_path):
    pages = [[{"id": 1}, {"id": 2}], [{"id": 3}, {"id": 4}], [{"id": 5}]]
    client = FakeClient(pages)
    manifest = fetch_all(client, tmp_path, "q;", page_size=2)

    assert manifest["games"] == 5
    assert manifest["pages"] == 3
    assert "offset 2;" in client.queries[1]
    assert (tmp_path / "page_0002.json").exists()


def test_resumes_from_saved_pages(tmp_path):
    (tmp_path / "page_0000.json").write_text(json.dumps([{"id": 1}, {"id": 2}]))
    client = FakeClient([[{"id": 3}]])  # only the missing page is requested
    manifest = fetch_all(client, tmp_path, "q;", page_size=2)

    assert manifest["pages_from_cache"] == 1
    assert manifest["pages_fetched"] == 1
    assert len(client.queries) == 1
    assert "offset 2;" in client.queries[0]
