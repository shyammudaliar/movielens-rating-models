import numpy as np
import pandas as pd
import pytest

from ratingmodels.data import movie_features, per_user_time_split, time_split
from ratingmodels.evaluate import (bootstrap_rmse_ci, movie_count_bucket,
                                   paired_rmse_diff_ci, rmse)
from ratingmodels.models import (ALSFactorization, BiasModel, GBMModel,
                                 GlobalMean, MovieMean)


def make_ratings(n_users=40, n_movies=30, density=0.5, seed=0, rank=2):
    """Synthetic ratings with planted structure: user/movie effects + low-rank taste."""
    rng = np.random.default_rng(seed)
    ub, mb = rng.normal(0, 0.5, n_users), rng.normal(0, 0.6, n_movies)
    P, Q = rng.normal(0, 0.6, (n_users, rank)), rng.normal(0, 0.6, (n_movies, rank))
    rows = []
    for u in range(n_users):
        for m in range(n_movies):
            if rng.random() < density:
                r = 3.5 + ub[u] + mb[m] + P[u] @ Q[m] + rng.normal(0, 0.2)
                rows.append((u, m, float(np.clip(np.round(r * 2) / 2, 0.5, 5)), len(rows)))
    df = pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"])
    df["timestamp"] = rng.permutation(len(df))  # mix users/movies across time
    return df


def test_time_split_is_chronological_and_complete():
    df = make_ratings()
    tr, va, te = time_split(df)
    assert len(tr) + len(va) + len(te) == len(df)
    assert tr["timestamp"].max() <= va["timestamp"].min()
    assert va["timestamp"].max() <= te["timestamp"].min()


def test_global_mean_predicts_the_average():
    df = make_ratings()
    m = GlobalMean().fit(df)
    assert np.allclose(m.predict(df), df["rating"].mean())


def test_movie_mean_falls_back_for_unseen_movie():
    df = make_ratings()
    m = MovieMean().fit(df)
    new = pd.DataFrame({"userId": [0], "movieId": [999], "rating": [3.0], "timestamp": [0]})
    assert m.predict(new)[0] == pytest.approx(df["rating"].mean())


def test_bias_model_shrinks_movies_with_few_ratings():
    # Movie 1 has one 5-star rating; movie 2 has 50 ratings averaging 5.0.
    rows = [(u, 0, 3.0, u) for u in range(60)]
    rows += [(0, 1, 5.0, 100)]
    rows += [(u, 2, 5.0, 200 + u) for u in range(50)]
    df = pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"])
    m = BiasModel(lam_item=5.0, lam_user=1e6).fit(df)  # ignore user effects
    one = m.predict(pd.DataFrame({"userId": [0], "movieId": [1]}))[0]
    many = m.predict(pd.DataFrame({"userId": [0], "movieId": [2]}))[0]
    assert one < many <= 5.0  # one rating is trusted less than fifty


def test_bias_model_with_no_shrinkage_recovers_movie_means():
    df = pd.DataFrame({"userId": [0, 1, 0, 1], "movieId": [0, 0, 1, 1],
                       "rating": [4.0, 4.0, 2.0, 2.0], "timestamp": range(4)})
    m = BiasModel(lam_item=0.0, lam_user=1e9).fit(df)
    assert m.predict(df) == pytest.approx([4.0, 4.0, 2.0, 2.0], abs=1e-6)


def test_predictions_are_clipped_to_valid_range():
    df = make_ratings()
    for model in [BiasModel(lam_item=0, lam_user=0), ALSFactorization(k=2, reg=0.01)]:
        pred = model.fit(df).predict(df)
        assert pred.min() >= 0.5 and pred.max() <= 5.0


def test_unseen_user_and_movie_get_a_finite_prediction():
    df = make_ratings()
    new = pd.DataFrame({"userId": [9999], "movieId": [9999], "rating": [3.0], "timestamp": [0]})
    for model in [BiasModel(), ALSFactorization(k=2)]:
        assert np.isfinite(model.fit(df).predict(new)).all()


def test_factorisation_beats_bias_model_when_taste_structure_exists():
    df = make_ratings(n_users=80, n_movies=60, density=0.6, seed=1, rank=2)
    tr, va, te = time_split(df)
    base = rmse(te["rating"], BiasModel(lam_item=1, lam_user=1).fit(tr).predict(te))
    als = rmse(te["rating"], ALSFactorization(k=3, reg=0.02, lam_item=1, lam_user=1).fit(tr).predict(te))
    assert als < base


def test_out_of_fold_features_never_include_the_row_itself():
    # Movie 7 has exactly one rating: out of fold, its count must be 0, not 1.
    rows = [(u, m, 3.0, u * 10 + m) for u in range(20) for m in range(3)]
    rows.append((0, 7, 5.0, 999))
    df = pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"])
    movies = pd.DataFrame({"movieId": list(range(3)) + [7],
                           "title": ["a (2000)", "b (2001)", "c (2002)", "d (2003)"],
                           "genres": ["Drama"] * 4})
    gbm = GBMModel(movie_features(movies))
    X = gbm.oof_features(df)
    lone = df.index[df["movieId"] == 7][0]
    assert X.loc[lone, "movie_count"] == 0


def test_gbm_trains_and_predicts():
    df = make_ratings(n_users=40, n_movies=25, density=0.6)
    movies = pd.DataFrame({"movieId": range(25), "title": [f"m{i} (199{i % 10})" for i in range(25)],
                           "genres": ["Drama|Comedy" if i % 2 else "Action" for i in range(25)]})
    gbm = GBMModel(movie_features(movies), max_iter=30).fit(df)
    pred = gbm.predict(df)
    assert len(pred) == len(df) and np.isfinite(pred).all()


def test_movie_count_buckets():
    fit = pd.DataFrame({"movieId": [1] * 150 + [2] * 3 + [3] * 10})
    target = pd.DataFrame({"movieId": [1, 2, 3, 99]})
    b = movie_count_bucket(fit, target).astype(str).tolist()
    assert b == ["100+", "1-4", "5-19", "unseen"]


def test_bootstrap_interval_contains_point_estimate_and_paired_diff_is_zero_for_same_model():
    sq = np.random.default_rng(0).exponential(1.0, 2000)
    lo, hi = bootstrap_rmse_ci(sq, n_boot=300)
    assert lo <= np.sqrt(sq.mean()) <= hi
    d, dlo, dhi = paired_rmse_diff_ci(sq, sq, n_boot=200)
    assert d == 0 and dlo == 0 and dhi == 0


def test_per_user_split_hides_each_users_newest_ratings():
    df = make_ratings(n_users=30, n_movies=40, density=0.7, seed=3)
    tr, va, te = per_user_time_split(df)
    assert len(tr) + len(va) + len(te) == len(df)
    assert set(te["userId"]) <= set(tr["userId"])  # every test user has history
    for u in te["userId"].unique()[:10]:
        assert tr.loc[tr.userId == u, "timestamp"].max() < te.loc[te.userId == u, "timestamp"].min()
