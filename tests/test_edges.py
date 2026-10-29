import math

import pandas as pd
import pytest

from overworld.graph.edges import apply_year_decay, group_edges, version_edges


@pytest.fixture
def games():
    return pd.DataFrame(
        {
            "series": [["thief"], ["thief"], ["thief", "looking glass"], [], ["solo"]],
            "developers": [["A"], ["A"], ["B"], ["B"], []],
            "year": [1998, 2000, 2004, None, 2014],
            "parent_game": [None, None, None, 1, 999],
        },
        index=pd.Index([1, 2, 3, 4, 5], name="id"),
    )


def test_group_edges_connects_all_pairs_with_size_decay(games):
    edges = group_edges(games, "series", base=0.8)
    assert set(zip(edges.src, edges.dst)) == {(1, 2), (1, 3), (2, 3)}
    assert edges.weight.iloc[0] == pytest.approx(0.8 / math.log(4))


def test_singleton_groups_make_no_edges(games):
    edges = group_edges(games, "series", base=0.8)
    assert 5 not in set(edges.src) | set(edges.dst)


def test_pair_in_several_groups_keeps_strongest():
    df = pd.DataFrame(
        {"series": [["big", "small"], ["big", "small"], ["big"], ["big"]]},
        index=[1, 2, 3, 4],
    )
    edges = group_edges(df, "series", base=1.0)
    pair = edges[(edges.src == 1) & (edges.dst == 2)]
    assert len(pair) == 1
    assert pair.weight.iloc[0] == pytest.approx(1.0 / math.log(3))


def test_year_decay(games):
    edges = pd.DataFrame({"src": [1, 3], "dst": [2, 4], "weight": [1.0, 1.0]})
    out = apply_year_decay(edges, games["year"])
    assert out.weight.iloc[0] == pytest.approx(1 / (1 + 2 / 20))
    assert out.weight.iloc[1] == 1.0


def test_version_edges_skip_missing_parents(games):
    edges = version_edges(games)
    assert list(zip(edges.src, edges.dst)) == [(1, 4)]


from overworld.graph.edges import content_edges


def tag_games(themes, genres):
    return pd.DataFrame(
        {"themes": themes, "genres": genres}, index=range(1, len(themes) + 1)
    )


def test_content_links_games_with_shared_tags():
    games = tag_games(
        themes=[["horror"], ["horror"], ["sci-fi"], []],
        genres=[["shooter"], ["shooter"], ["racing"], []],
    )
    edges = content_edges(games)
    pairs = set(zip(edges.src, edges.dst))
    assert (1, 2) in pairs
    assert all(4 not in p for p in pairs)
    assert all(a != b for a, b in pairs)


from overworld.graph.edges import BASE_WEIGHTS


def test_identical_games_get_full_content_weight():
    games = tag_games(themes=[["horror"], ["horror"], ["comedy"]], genres=[[], [], []])
    edges = content_edges(games)
    w = edges[(edges.src == 1) & (edges.dst == 2)].weight.iloc[0]
    assert w == pytest.approx(BASE_WEIGHTS["content"])


import numpy as np

from overworld.graph.edges import select_neighbors


def test_select_keeps_top_n():
    row = np.array([0.0, 0.9, 0.8, 0.7, 0.6])
    assert set(select_neighbors(row, top_n=2, rel_threshold=0.0)) == {1, 2}


def test_select_drops_weak_tail():
    row = np.array([0.0, 0.9, 0.5, 0.1])  # 0.1 < 30% of 0.9
    assert set(select_neighbors(row, top_n=10, rel_threshold=0.3)) == {1, 2}


def test_select_handles_rows_smaller_than_n():
    row = np.array([0.0, 0.4])  # only one other game
    assert list(select_neighbors(row, top_n=10, rel_threshold=0.3)) == [1]


def test_select_ignores_zero_similarity():
    row = np.array([0.0, 0.0, 0.0])
    assert len(select_neighbors(row, top_n=10, rel_threshold=0.3)) == 0
    games = tag_games(
        themes=[
            ["horror", "space"],
            ["horror", "space"],
            ["horror", "a", "b", "c", "d", "e"],
        ],
        genres=[[], [], []],
    )
    edges = content_edges(games)
    pairs = set(zip(edges.src, edges.dst))
    assert (1, 2) in pairs
    assert (1, 3) not in pairs


def test_content_edges_are_reproducible():
    games = tag_games(
        themes=[["horror"]] * 6 + [["comedy"]],
        genres=[["shooter"]] * 6 + [[]],
    )
    a = content_edges(games)
    b = content_edges(games)
    pd.testing.assert_frame_equal(a, b)
