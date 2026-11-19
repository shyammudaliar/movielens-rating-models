from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PRETTY = {
    "global_mean": "Global average",
    "movie_mean": "Movie's own average",
    "bias_model": "Average + user/movie effects (shrunk)",
    "matrix_factorisation": "+ taste factors (matrix factorisation)",
    "gradient_boosting": "Gradient boosting",
}


def plot_overall(overall: pd.DataFrame, path: str) -> None:
    df = overall.sort_values("rmse", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    y = np.arange(len(df))
    err = np.vstack([df["rmse"] - df["ci_lo"], df["ci_hi"] - df["rmse"]])
    ax.barh(y, df["rmse"], xerr=err, color="#2874a6", alpha=0.85, capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels([PRETTY[m] for m in df["model"]], fontsize=9)
    ax.set_xlabel("Test RMSE (lower is better), with 95% bootstrap interval")
    lo = max(0.0, df["ci_lo"].min() - 0.05)
    ax.set_xlim(lo, df["ci_hi"].max() + 0.03)
    for yi, v in zip(y, df["rmse"]):
        ax.text(v + 0.003, yi + 0.28, f"{v:.3f}", fontsize=8)
    ax.set_title("Which model predicts newer ratings best?")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def plot_by_bucket(by_bucket: pd.DataFrame, path: str) -> None:
    labels = ["unseen", "1-4", "5-19", "20-99", "100+"]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for m, g in by_bucket.groupby("model"):
        g = g.set_index("bucket").reindex(labels)
        ax.plot(labels, g["rmse"], marker="o", label=PRETTY[m])
    ax.set_xlabel("How many ratings the movie had in the training data")
    ax.set_ylabel("Test RMSE (lower is better)")
    ax.set_title("Error by how much data we have about the movie")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
