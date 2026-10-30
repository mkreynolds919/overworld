import pandas as pd

from overworld.evaluation.split import (
    eligible_groups,
    hold_out,
    make_queries,
    split_groups,
)


def test_eligible_groups_respects_size_limits():
    games = pd.DataFrame(
        {"series": [["a", "b"], ["a", "b"], ["a"], ["c"]]}, index=[1, 2, 3, 4]
    )
    groups = eligible_groups(games)
    assert groups == {"a": [1, 2, 3]}  # "b" has 2 games, "c" has 1


def test_split_is_deterministic_and_disjoint():
    groups = {f"g{i}": [i, i + 100, i + 200] for i in range(50)}
    s1, s2 = split_groups(groups), split_groups(groups)
    assert s1 == s2
    assert not set(s1["val"]) & set(s1["test"])
    assert sorted(s1["val"] + s1["test"] + s1["train"]) == sorted(groups)


def test_hold_out_removes_only_inside_series_and_version():
    groups = {"thief": [1, 2, 3]}
    edges = pd.DataFrame(
        [
            (1, 2, "series"),  # inside group, hidden type -> removed
            (2, 3, "version"),  # inside group, hidden type -> removed
            (1, 3, "developer"),  # inside group, kept type -> stays
            (1, 9, "series"),  # leaves the group -> stays
        ],
        columns=["src", "dst", "type"],
    ).assign(weight=1.0)
    kept = hold_out(edges, groups, ["thief"])
    assert list(zip(kept.src, kept.dst, kept.type)) == [
        (1, 3, "developer"),
        (1, 9, "series"),
    ]


def test_queries_exclude_the_seed():
    queries = make_queries({"thief": [1, 2, 3]}, ["thief"])
    assert len(queries) == 3
    q = next(q for q in queries if q.seed == 2)
    assert q.relevant == {1, 3}
