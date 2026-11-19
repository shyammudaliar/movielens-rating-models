"""Five ways to predict a rating, from simplest to most flexible.

1. GlobalMean      : always predict the overall average.
2. MovieMean       : each movie's raw average (unseen movies get the global average).
3. BiasModel       : overall average + how generous the user is + how good the
                     movie is, with the movie/user effects *shrunk* toward zero
                     when they rest on few ratings.
4. ALSFactorization: BiasModel plus learned "taste dimensions" (matrix factorisation).
5. GBMModel        : gradient-boosted trees on user/movie statistics, genres and year.

All models share fit(train) / predict(df). Predictions are clipped to the
valid rating range of 0.5 to 5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

LO, HI = 0.5, 5.0


class IdMap:
    """Maps raw ids to 0..n-1; unseen ids map to -1."""

    def __init__(self, ids):
        self.index = {v: i for i, v in enumerate(pd.unique(pd.Series(ids)))}

    def __len__(self) -> int:
        return len(self.index)

    def get(self, ids) -> np.ndarray:
        return pd.Series(ids).map(self.index).fillna(-1).astype(int).to_numpy()


def _safe(arr: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """arr[idx] where idx == -1 gives 0 instead of wrapping around."""
    return np.where(idx >= 0, arr[np.maximum(idx, 0)], 0.0)


class GlobalMean:
    name = "global_mean"

    def fit(self, train: pd.DataFrame):
        self.mu = float(train["rating"].mean())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self.mu)


class MovieMean:
    name = "movie_mean"

    def fit(self, train: pd.DataFrame):
        self.mu = float(train["rating"].mean())
        self.means = train.groupby("movieId")["rating"].mean()
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return df["movieId"].map(self.means).fillna(self.mu).to_numpy(float)


class BiasModel:
    """rating ~ mu + user_bias + movie_bias, with shrinkage.

    movie_bias = (sum of the movie's residuals) / (lam_item + number of ratings).
    A larger lam_item trusts a movie's average less when it has few ratings.
    """

    name = "bias_model"

    def __init__(self, lam_item: float = 10.0, lam_user: float = 10.0, n_iter: int = 15):
        self.lam_item, self.lam_user, self.n_iter = lam_item, lam_user, n_iter

    def fit(self, train: pd.DataFrame):
        self.mu = float(train["rating"].mean())
        self.umap, self.imap = IdMap(train["userId"]), IdMap(train["movieId"])
        u, i = self.umap.get(train["userId"]), self.imap.get(train["movieId"])
        r = train["rating"].to_numpy(float)
        nu, ni = len(self.umap), len(self.imap)
        cnt_u, cnt_i = np.bincount(u, minlength=nu), np.bincount(i, minlength=ni)
        bu, bi = np.zeros(nu), np.zeros(ni)
        for _ in range(self.n_iter):
            bi = np.bincount(i, weights=r - self.mu - bu[u], minlength=ni) / (self.lam_item + cnt_i)
            bu = np.bincount(u, weights=r - self.mu - bi[i], minlength=nu) / (self.lam_user + cnt_u)
        self.bu, self.bi = bu, bi
        return self

    def raw_predict(self, df: pd.DataFrame) -> np.ndarray:
        u, i = self.umap.get(df["userId"]), self.imap.get(df["movieId"])
        return self.mu + _safe(self.bu, u) + _safe(self.bi, i)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.clip(self.raw_predict(df), LO, HI)


class ALSFactorization:
    """Bias model + low-rank "taste" factors learned by alternating least squares."""

    name = "matrix_factorisation"

    def __init__(self, k: int = 10, reg: float = 0.1, n_iter: int = 12,
                 lam_item: float = 10.0, lam_user: float = 10.0, seed: int = 0):
        self.k, self.reg, self.n_iter = k, reg, n_iter
        self.lam_item, self.lam_user, self.seed = lam_item, lam_user, seed

    @staticmethod
    def _group(idx: np.ndarray, n: int):
        order = np.argsort(idx, kind="stable")
        ptr = np.concatenate([[0], np.cumsum(np.bincount(idx, minlength=n))])
        return order, ptr

    def fit(self, train: pd.DataFrame):
        self.base = BiasModel(self.lam_item, self.lam_user).fit(train)
        u = self.base.umap.get(train["userId"])
        i = self.base.imap.get(train["movieId"])
        res = train["rating"].to_numpy(float) - self.base.raw_predict(train)
        nu, ni, k = len(self.base.umap), len(self.base.imap), self.k
        rng = np.random.default_rng(self.seed)
        self.P = np.zeros((nu, k))
        self.Q = rng.normal(0, 0.3, (ni, k))
        u_order, u_ptr = self._group(u, nu)
        i_order, i_ptr = self._group(i, ni)
        eye = np.eye(k)
        for _ in range(self.n_iter):
            for a in range(nu):
                rows = u_order[u_ptr[a]:u_ptr[a + 1]]
                if len(rows) == 0:
                    continue
                Qs = self.Q[i[rows]]
                self.P[a] = np.linalg.solve(Qs.T @ Qs + self.reg * len(rows) * eye, Qs.T @ res[rows])
            for b in range(ni):
                rows = i_order[i_ptr[b]:i_ptr[b + 1]]
                if len(rows) == 0:
                    continue
                Ps = self.P[u[rows]]
                self.Q[b] = np.linalg.solve(Ps.T @ Ps + self.reg * len(rows) * eye, Ps.T @ res[rows])
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        u = self.base.umap.get(df["userId"])
        i = self.base.imap.get(df["movieId"])
        both = (u >= 0) & (i >= 0)
        interaction = np.zeros(len(df))
        interaction[both] = np.einsum("ij,ij->i", self.P[u[both]], self.Q[i[both]])
        return np.clip(self.base.raw_predict(df) + interaction, LO, HI)


class GBMModel:
    """Gradient-boosted trees on user/movie statistics, genres and release year.

    Training-row statistics are computed *out of fold*: a row's own rating never
    contributes to its own features. Otherwise the model would just memorise the answer.
    """

    name = "gradient_boosting"

    def __init__(self, movie_feats: pd.DataFrame, max_depth: int | None = 3,
                 max_iter: int = 300, learning_rate: float = 0.05,
                 prior_strength: float = 5.0, n_folds: int = 5, seed: int = 0):
        self.movie_feats = movie_feats
        self.max_depth, self.max_iter, self.lr = max_depth, max_iter, learning_rate
        self.m, self.n_folds, self.seed = prior_strength, n_folds, seed

    @staticmethod
    def _stats(df: pd.DataFrame):
        return (df.groupby("userId")["rating"].agg(["sum", "count"]),
                df.groupby("movieId")["rating"].agg(["sum", "count"]),
                float(df["rating"].mean()))

    def _features(self, df: pd.DataFrame, stats, count_scale: float = 1.0) -> pd.DataFrame:
        ustat, istat, mu = stats
        us = ustat.reindex(df["userId"]).fillna(0.0)
        its = istat.reindex(df["movieId"]).fillna(0.0)
        feats = pd.DataFrame({
            "user_mean": ((us["sum"] + self.m * mu) / (us["count"] + self.m)).to_numpy(),
            "user_count": us["count"].to_numpy() * count_scale,
            "movie_mean": ((its["sum"] + self.m * mu) / (its["count"] + self.m)).to_numpy(),
            "movie_count": its["count"].to_numpy() * count_scale,
        }, index=df.index)
        meta = self.movie_feats.reindex(df["movieId"]).set_index(df.index)
        return pd.concat([feats, meta], axis=1)

    def oof_features(self, train: pd.DataFrame) -> pd.DataFrame:
        parts = []
        kf = KFold(self.n_folds, shuffle=True, random_state=self.seed)
        scale = self.n_folds / (self.n_folds - 1)
        for fit_idx, hold_idx in kf.split(train):
            stats = self._stats(train.iloc[fit_idx])
            parts.append(self._features(train.iloc[hold_idx], stats, count_scale=scale))
        return pd.concat(parts).loc[train.index]

    def fit(self, train: pd.DataFrame):
        X = self.oof_features(train)
        self.columns = list(X.columns)
        self.stats = self._stats(train)
        self.model = HistGradientBoostingRegressor(
            max_depth=self.max_depth, max_iter=self.max_iter,
            learning_rate=self.lr, random_state=self.seed,
        ).fit(X, train["rating"].to_numpy(float))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = self._features(df, self.stats)[self.columns]
        return np.clip(self.model.predict(X), LO, HI)
