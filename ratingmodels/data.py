"""Download and split the MovieLens "latest-small" dataset.

Data: https://grouplens.org/datasets/movielens/ (research/education use; it is
downloaded on demand and never committed to this repo).
"""

from __future__ import annotations

import io
import re
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


def download(data_dir: str = "data") -> Path:
    """Download and unzip the dataset if it isn't already there."""
    root = Path(data_dir) / "ml-latest-small"
    if (root / "ratings.csv").exists():
        return root
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=60) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        zf.extractall(data_dir)
    return root


def load(data_dir: str = "data") -> tuple[pd.DataFrame, pd.DataFrame]:
    root = download(data_dir)
    ratings = pd.read_csv(root / "ratings.csv")
    movies = pd.read_csv(root / "movies.csv")
    return ratings, movies


def movie_features(movies: pd.DataFrame) -> pd.DataFrame:
    """Static movie metadata (release year and genre flags), indexed by movieId."""
    year = movies["title"].str.extract(r"\((\d{4})\)\s*$")[0].astype(float)
    genres = movies["genres"].str.get_dummies(sep="|")
    genres = genres.drop(columns=[c for c in genres.columns if c.startswith("(no genres")], errors="ignore")
    genres.columns = [re.sub(r"\W+", "_", c) for c in genres.columns]
    out = pd.concat([movies[["movieId"]], year.rename("year"), genres], axis=1)
    return out.set_index("movieId")


def time_split(ratings: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15):
    """Split by time: train on the oldest ratings, test on the newest.

    A random split would let the model "see the future". Predicting newer
    ratings from older ones is how a real system would be used.
    """
    ordered = ratings.sort_values("timestamp", kind="stable").reset_index(drop=True)
    n = len(ordered)
    a, b = int(n * train_frac), int(n * (train_frac + val_frac))
    return ordered.iloc[:a].copy(), ordered.iloc[a:b].copy(), ordered.iloc[b:].copy()


def per_user_time_split(ratings: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15):
    """Hide each user's *newest* ratings.

    Every user keeps their oldest 70% for training, the next 15% for tuning and
    their newest 15% for testing. Unlike the global split, nearly every test
    user has history, which is closer to "predict what this user rates next".
    """
    ordered = ratings.sort_values(["userId", "timestamp"], kind="stable").reset_index(drop=True)
    pos = ordered.groupby("userId").cumcount() / ordered.groupby("userId")["rating"].transform("size")
    train = ordered[pos < train_frac].copy()
    val = ordered[(pos >= train_frac) & (pos < train_frac + val_frac)].copy()
    test = ordered[pos >= train_frac + val_frac].copy()
    return train, val, test


SPLITS = {"global_time": time_split, "per_user_time": per_user_time_split}
