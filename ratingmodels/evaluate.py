"""Metrics, sparse-data buckets and a paired bootstrap."""

from __future__ import annotations

import numpy as np
import pandas as pd

BUCKET_EDGES = [-0.5, 0.5, 4.5, 19.5, 99.5, np.inf]
BUCKET_LABELS = ["unseen", "1-4", "5-19", "20-99", "100+"]


def rmse(y, pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(pred)) ** 2)))


def mae(y, pred) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(pred))))


def movie_count_bucket(fit_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.Series:
    """How many ratings did each target row's movie have in the data we trained on?"""
    counts = fit_df["movieId"].value_counts()
    c = target_df["movieId"].map(counts).fillna(0).to_numpy()
    return pd.Series(pd.cut(c, bins=BUCKET_EDGES, labels=BUCKET_LABELS), index=target_df.index)


def bootstrap_rmse_ci(sq_err: np.ndarray, n_boot: int = 1000, seed: int = 0, level: float = 0.95):
    """Confidence interval for RMSE by resampling test rows."""
    rng = np.random.default_rng(seed)
    n = len(sq_err)
    stats = np.array([np.sqrt(sq_err[rng.integers(0, n, n)].mean()) for _ in range(n_boot)])
    lo, hi = np.quantile(stats, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def paired_rmse_diff_ci(sq_a: np.ndarray, sq_b: np.ndarray, n_boot: int = 1000,
                        seed: int = 0, level: float = 0.95):
    """CI for RMSE(a) - RMSE(b), resampling the same rows for both models.

    If the interval excludes 0, the difference is unlikely to be noise.
    """
    rng = np.random.default_rng(seed)
    n = len(sq_a)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[b] = np.sqrt(sq_a[idx].mean()) - np.sqrt(sq_b[idx].mean())
    lo, hi = np.quantile(diffs, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(np.sqrt(sq_a.mean()) - np.sqrt(sq_b.mean())), float(lo), float(hi)
