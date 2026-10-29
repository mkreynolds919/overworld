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
    assert edges.weight.iloc[0] == pytest.approx(0.8 / math.log(4))  # group of 3


def test_singleton_groups_make_no_edges(games):
    edges = group_edges(games, "series", base=0.8)
    assert 5 not in set(edges.src) | set(edges.dst)  # "solo" has one game


def test_pair_in_several_groups_keeps_strongest():
    df = pd.DataFrame(
        {"series": [["big", "small"], ["big", "small"], ["big"], ["big"]]},
        index=[1, 2, 3, 4],
    )
    edges = group_edges(df, "series", base=1.0)
    pair = edges[(edges.src == 1) & (edges.dst == 2)]
    assert len(pair) == 1
    assert pair.weight.iloc[0] == pytest.approx(
        1.0 / math.log(3)
    )  # "small" (size 2) wins


def test_year_decay(games):
    edges = pd.DataFrame({"src": [1, 3], "dst": [2, 4], "weight": [1.0, 1.0]})
    out = apply_year_decay(edges, games["year"])
    assert out.weight.iloc[0] == pytest.approx(1 / (1 + 2 / 20))  # 2 years apart
    assert out.weight.iloc[1] == 1.0  # unknown year: no decay


def test_version_edges_skip_missing_parents(games):
    edges = version_edges(games)
    assert list(zip(edges.src, edges.dst)) == [(1, 4)]  # 999 isn't in the graph
