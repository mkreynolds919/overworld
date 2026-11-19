from overworld.igdb.fetch_summaries import fetch_summaries


class FakeClient:
    def __init__(self, pages):
        self.pages = list(pages)
        self.bodies = []

    def query(self, endpoint, body):
        self.bodies.append(body)
        return self.pages.pop(0)


def test_batches_ids_and_skips_missing_summaries():
    client = FakeClient([
        [{"id": 1, "summary": "A gothic hunt."}, {"id": 2}],    # game 2 has no summary
        [{"id": 3, "summary": ""}],                              # empty counts as missing
    ])
    result = fetch_summaries(client, [1, 2, 3], batch=2)
    assert result == {1: "A gothic hunt."}
    assert client.bodies[0] == "fields summary; where id = (1,2); limit 2;"
    assert "where id = (3)" in client.bodies[1]