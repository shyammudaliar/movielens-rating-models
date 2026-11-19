"""python -m ratingmodels.run --split per_user_time   (downloads the data on first run, ~1 MB)"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ratingmodels import data
from ratingmodels.evaluate import (BUCKET_LABELS, bootstrap_rmse_ci, mae,
                                   movie_count_bucket, paired_rmse_diff_ci, rmse)
from ratingmodels.models import (ALSFactorization, BiasModel, GBMModel,
                                 GlobalMean, MovieMean)
from ratingmodels.plots import plot_by_bucket, plot_overall


def tune(factory, grid: list[dict], train, val):
    """Pick the setting with the lowest validation RMSE."""
    best, best_score = None, np.inf
    for params in grid:
        score = rmse(val["rating"], factory(**params).fit(train).predict(val))
        if score < best_score:
            best, best_score = params, score
    return best, best_score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--split", choices=list(data.SPLITS), default="per_user_time",
                    help="global_time: newest ratings overall (mostly new users); "
                         "per_user_time: each user's newest ratings")
    args = ap.parse_args()
    out = Path(args.out) / args.split
    out.mkdir(parents=True, exist_ok=True)

    ratings, movies = data.load(args.data_dir)
    movie_feats = data.movie_features(movies)
    train, val, test = data.SPLITS[args.split](ratings)
    trainval = pd.concat([train, val])
    print(f"split={args.split} ratings: train={len(train)} val={len(val)} test={len(test)}")
    unseen_movie = (~test["movieId"].isin(trainval["movieId"])).mean()
    unseen_user = (~test["userId"].isin(trainval["userId"])).mean()
    print(f"test rows with a movie never seen in training: {unseen_movie:.1%}; user: {unseen_user:.1%}")

    chosen = {}

    # Bias model: how much to distrust averages built on few ratings
    bias_grid = [dict(lam_item=a, lam_user=b) for a, b in itertools.product([0, 1, 3, 5, 10, 20, 50], [0, 5, 10, 25])]
    chosen["bias_model"], s = tune(BiasModel, bias_grid, train, val)
    print("bias_model", chosen["bias_model"], f"val RMSE {s:.4f}")

    # Matrix factorisation: number of taste dimensions and regularisation
    bp = chosen["bias_model"]
    als_grid = [dict(k=k, reg=r, **bp) for k, r in itertools.product([5, 10, 20], [0.01, 0.03, 0.1])]
    chosen["matrix_factorisation"], s = tune(ALSFactorization, als_grid, train, val)
    print("matrix_factorisation", chosen["matrix_factorisation"], f"val RMSE {s:.4f}")

    # Gradient boosting
    gbm_grid = [dict(movie_feats=movie_feats, max_depth=d, max_iter=it) for d, it in itertools.product([3, 6], [200, 400])]
    chosen["gradient_boosting"], s = tune(GBMModel, gbm_grid, train, val)
    print("gradient_boosting", {k: v for k, v in chosen["gradient_boosting"].items() if k != "movie_feats"}, f"val RMSE {s:.4f}")

    factories = {
        "global_mean": lambda: GlobalMean(),
        "movie_mean": lambda: MovieMean(),
        "bias_model": lambda: BiasModel(**chosen["bias_model"]),
        "matrix_factorisation": lambda: ALSFactorization(**chosen["matrix_factorisation"]),
        "gradient_boosting": lambda: GBMModel(**chosen["gradient_boosting"]),
    }

    # Final models are refit on train+val and scored once on the untouched, newest ratings
    buckets = movie_count_bucket(trainval, test)
    rows, bucket_rows, sq_errs = [], [], {}
    for name, make in factories.items():
        pred = make().fit(trainval).predict(test)
        y = test["rating"].to_numpy(float)
        sq = (y - pred) ** 2
        sq_errs[name] = sq
        lo, hi = bootstrap_rmse_ci(sq)
        rows.append({"model": name, "rmse": rmse(y, pred), "mae": mae(y, pred), "ci_lo": lo, "ci_hi": hi})
        for b in BUCKET_LABELS:
            mask = (buckets == b).to_numpy()
            if mask.sum() > 0:
                bucket_rows.append({"model": name, "bucket": b, "rmse": rmse(y[mask], pred[mask]), "n": int(mask.sum())})
        print(f"{name:22s} RMSE {rows[-1]['rmse']:.4f}  MAE {rows[-1]['mae']:.4f}")

    overall = pd.DataFrame(rows)
    by_bucket = pd.DataFrame(bucket_rows)
    overall.to_csv(out / "overall.csv", index=False)
    by_bucket.to_csv(out / "by_bucket.csv", index=False)
    plot_overall(overall, str(out / "overall_rmse.png"))
    plot_by_bucket(by_bucket, str(out / "rmse_by_movie_data.png"))

    ref = "bias_model"
    comparisons = {}
    for other in ["movie_mean", "matrix_factorisation", "gradient_boosting"]:
        d, lo, hi = paired_rmse_diff_ci(sq_errs[other], sq_errs[ref])
        comparisons[f"{other}_minus_{ref}"] = {"diff": round(d, 4), "ci": [round(lo, 4), round(hi, 4)]}
        print(f"RMSE({other}) - RMSE({ref}) = {d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")

    report = {
        "split": args.split, "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "test_unseen_movie_share": round(float(unseen_movie), 4),
        "test_unseen_user_share": round(float(unseen_user), 4),
        "chosen": {k: {kk: vv for kk, vv in v.items() if kk != "movie_feats"} for k, v in chosen.items()},
        "paired_comparisons": comparisons,
        "bucket_sizes": by_bucket.drop_duplicates("bucket").set_index("bucket")["n"].to_dict(),
    }
    with open(out / "report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Saved to {out}/")


if __name__ == "__main__":
    main()
