# Bootstrap over multiple series groups to confirm model performance, returns confidence intervals for both metrics and model differences

import numpy as np
import pandas as pd

from overworld.evaluation.harness import METRICS

N_BOOT = 1000
BOOT_SEED = 0


def bootstrap_ci(
    per_query: pd.DataFrame,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
    level: float = 0.95,
) -> pd.DataFrame:
    # Returns mean and confidence interval per metric, resamples the whole group with replacement
    by_group = per_query.groupby("group")
    sums = by_group[METRICS].sum().to_numpy()
    counts = by_group.size().to_numpy()

    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(counts), size=(n_boot, len(counts)))
    means = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)[:, None]

    tail = (1 - level) / 2
    low, high = np.quantile(means, [tail, 1 - tail], axis=0)
    return pd.DataFrame(
        {"mean": per_query[METRICS].mean(), "low": low, "high": high}, index=METRICS
    )


def paired_difference(a: pd.DataFrame, b: pd.DataFrame, **kwargs) -> pd.DataFrame:
    # Makes an interval for (model a - model b) using the same resampled groups for both
    if not a["seed"].equals(b["seed"]):
        raise ValueError(
            "both models must be scored on the same queries, in the same order"
        )
    diff = a[METRICS] - b[METRICS]
    return bootstrap_ci(diff.assign(group=a["group"]), **kwargs)


def format_ci(ci: pd.DataFrame) -> pd.Series:
    return ci.apply(
        lambda r: f"{r['mean']:.3f} [{r['low']:.3f}, {r['high']:.3f}]", axis=1
    )
