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



from overworld.graph.edges import diversify

SERIES = np.array([[], ["a"], ["a"], ["a"], ["b"], ["a", "c"]], dtype=object)
FAMILY = np.array([0, 1, 2, 3, 4, 5])


def test_caps_each_other_series():
    picked = diversify(np.array([1, 2, 3, 4]), k=3, series=SERIES, family=FAMILY, self_index=0, cap=2)
    assert list(picked) == [1, 2, 4]               # the third "a" game is skipped


def test_own_series_is_exempt():
    picked = diversify(np.array([2, 3, 5]), k=3, series=SERIES, family=FAMILY, self_index=1, cap=2)
    assert list(picked) == [2, 3, 5]               # all share game 1's series "a"


def test_one_per_family_and_never_own_family():
    family = np.array([0, 0, 1, 1, 2, 3])          # 0 and 1 are one family; 2 and 3 another
    picked = diversify(np.array([1, 2, 3, 4]), k=4, series=np.array([[]] * 6, dtype=object),
                       family=family, self_index=0)
    assert list(picked) == [2, 4]                  # 1 is own family; 3 duplicates 2's family


def test_multi_series_candidate_counts_toward_each():
    series = np.array([[], ["c"], ["c"], ["a", "c"]], dtype=object)
    picked = diversify(np.array([1, 2, 3]), k=3, series=series, family=np.arange(4), self_index=0, cap=2)
    assert list(picked) == [1, 2]                  # game 3 blocked: "c" is full


def test_plain_content_edges_unchanged_by_switch():
    games = tag_games(themes=[["horror"]] * 6 + [["comedy"]], genres=[["shooter"]] * 6 + [[]])
    games = games.assign(series=[[]] * 7)
    pd.testing.assert_frame_equal(content_edges(games), content_edges(games, diversify_series=False))


from overworld.graph.edges import cap_final_edges


def test_final_cap_drops_edges_picked_from_the_other_side():
    # Game 0 has no series; games 1-4 are all in series "s" and each linked to 0.
    edges = pd.DataFrame({"src": [0, 0, 0, 0], "dst": [1, 2, 3, 4], "weight": [0.9, 0.8, 0.7, 0.6]})
    series = pd.Series({0: [], 1: ["s"], 2: ["s"], 3: ["s"], 4: ["s"]})
    family = pd.Series({i: i for i in range(5)})
    kept = cap_final_edges(edges, series, family, cap=2)
    assert list(zip(kept.src, kept.dst)) == [(0, 1), (0, 2)]      # strongest two "s" links survive
