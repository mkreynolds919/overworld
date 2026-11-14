import pandas as pd
import pytest

from overworld.evaluation.bootstrap import bootstrap_ci, paired_difference


def frame(rows):
    """rows: (seed, group, value). The value is used for all four metrics."""
    return pd.DataFrame(
        [
            {
                "seed": s,
                "group": g,
                "hit_rate": v,
                "precision": v,
                "recall": v,
                "ndcg": v,
            }
            for s, g, v in rows
        ]
    )


def test_identical_groups_give_zero_width_interval():
    pq = frame([(1, "a", 0.5), (2, "b", 0.5), (3, "c", 0.5)])
    ci = bootstrap_ci(pq, n_boot=200)
    assert ci.loc["ndcg", "low"] == pytest.approx(0.5)
    assert ci.loc["ndcg", "high"] == pytest.approx(0.5)


def test_mean_weights_queries_not_groups():
    pq = frame([(1, "big", 1.0), (2, "big", 1.0), (3, "big", 1.0), (4, "small", 0.0)])
    assert bootstrap_ci(pq).loc["recall", "mean"] == pytest.approx(
        0.75
    )  # 3 of 4 queries


def test_interval_contains_mean_and_is_reproducible():
    pq = frame([(i, f"g{i % 7}", (i % 5) / 4) for i in range(40)])
    a, b = bootstrap_ci(pq, seed=3), bootstrap_ci(pq, seed=3)
    pd.testing.assert_frame_equal(a, b)
    assert (a["low"] <= a["mean"]).all() and (a["mean"] <= a["high"]).all()


def test_model_minus_itself_is_exactly_zero():
    pq = frame([(i, f"g{i % 5}", (i % 3) / 2) for i in range(20)])
    d = paired_difference(pq, pq)
    assert (d[["mean", "low", "high"]] == 0).all().all()


def test_paired_difference_rejects_misaligned_queries():
    a = frame([(1, "a", 0.5), (2, "b", 0.5)])
    b = frame([(2, "b", 0.5), (1, "a", 0.5)])
    with pytest.raises(ValueError):
        paired_difference(a, b)
