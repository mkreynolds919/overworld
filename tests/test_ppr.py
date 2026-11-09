import numpy as np
import pandas as pd
import pytest

from overworld.models.ppr import PersonalizedPageRank, build_transition


def games(n):
    return pd.DataFrame(index=pd.Index(range(1, n + 1), name="id"))


def edges(rows):
    return pd.DataFrame(rows, columns=["src", "dst", "type", "weight"])


PATH = edges([(1, 2, "series", 1.0), (2, 3, "series", 1.0), (3, 4, "series", 1.0)])


def test_scores_sum_to_one():
    ppr = PersonalizedPageRank(games(5), PATH)  # game 5 is isolated
    assert ppr.scores(1).sum() == pytest.approx(1.0)


def test_closer_games_score_higher():
    r = PersonalizedPageRank(games(4), PATH).scores(1)
    assert r[1] > r[2] > r[3]  # positions of games 2, 3, 4


def test_heavier_edges_win():
    e = edges([(1, 2, "series", 1.0), (1, 3, "content", 0.1)])
    assert PersonalizedPageRank(games(3), e).recommend(1, k=2) == [2, 3]


def test_types_between_same_pair_are_summed():
    e = edges([(1, 2, "series", 0.5), (1, 2, "developer", 0.3), (1, 3, "content", 0.7)])
    P, _ = build_transition(e, np.array([1, 2, 3]))
    assert P[0, 1] == pytest.approx(0.8 / 1.5)  # (0.5 + 0.3) / total 1.5


def test_type_weight_zero_turns_type_off():
    e = edges([(1, 2, "series", 1.0), (1, 3, "developer", 1.0)])
    ppr = PersonalizedPageRank(
        games(3), e, type_weights={"series": 1.0, "developer": 0.0}
    )
    assert ppr.recommend(1) == [2]


def test_isolated_seed_returns_nothing():
    ppr = PersonalizedPageRank(games(5), PATH)
    assert ppr.recommend(5) == []


def test_never_recommends_seed():
    assert 1 not in PersonalizedPageRank(games(4), PATH).recommend(1)


def test_exclude_neighbors_is_opt_in():
    default = PersonalizedPageRank(games(4), PATH).recommend(1)
    product = PersonalizedPageRank(games(4), PATH, exclude_neighbors=True).recommend(1)
    assert 2 in default and 2 not in product
