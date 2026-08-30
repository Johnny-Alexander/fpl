"""
Honest evaluation of the points model.

Two things are being corrected here.

First, the split. A random `train_test_split` over a player-gameweek panel puts
adjacent gameweeks of the same player on both sides of the boundary, and their
rolling windows overlap, so the reported error is optimistic. Evaluation is
walk-forward: for each test gameweek, train only on rows whose target was already
known before that gameweek kicked off.

Second, the baselines. Roughly 60% of rows are players who did not play, so a model
that predicts zero for everyone scores a deceptively good MAE. A score is only
meaningful next to the naive predictors it claims to beat, and on the population
the optimizer actually chooses between -- players likely to start.
"""

import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import features
import ml_model


def naive_baselines(test_df):
    """Predictors any real model must beat."""
    return {
        "predict zero": np.zeros(len(test_df)),
        "last gw points": test_df["total_points"].to_numpy(dtype=float),
        "mean of last 3": test_df["total_points_rolling_3"].to_numpy(dtype=float),
        "season pts/game": np.where(
            test_df["GW"] > 1,
            test_df["total_points_cumsum"] / test_df["GW"].clip(lower=1),
            0.0,
        ),
    }


def score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    if len(np.unique(y_pred)) > 1:
        rho = float(spearmanr(y_true, y_pred).statistic)
    else:
        rho = float("nan")
    return {"mae": mae, "rmse": rmse, "spearman": rho}


def top_n_realised(test_df, y_pred, n=50):
    """
    Mean points actually scored by the n highest-predicted players.

    This is the metric closest to what the optimizer does: it only ever acts on
    the top of the ranking, so accuracy in the tail is irrelevant.
    """
    if len(test_df) < n:
        return float("nan")
    order = np.argsort(-np.asarray(y_pred, dtype=float))
    picked = test_df.iloc[order[:n]]
    return float(picked["target_points"].mean())


def walk_forward(panel, feature_cols, season, first_gw, last_gw, kind, step=1):
    """
    Roll an expanding training window forward across a season.

    A row's target is the following gameweek's points, so training on rows up to
    GW `t-2` uses only outcomes observed by GW `t-1` -- nothing from the gameweek
    being predicted.
    """
    labelled = panel.dropna(subset=["target_points"] + feature_cols)
    season_index = panel.loc[panel["season"] == season, "season_index"].iloc[0]

    collected = []
    for test_gw in range(first_gw, last_gw + 1, step):
        train = labelled[
            (labelled["season_index"] < season_index)
            | (
                (labelled["season_index"] == season_index)
                & (labelled["GW"] <= test_gw - 2)
            )
        ]
        test = labelled[
            (labelled["season_index"] == season_index) & (labelled["GW"] == test_gw - 1)
        ]
        if len(train) < 500 or len(test) == 0:
            continue

        model = ml_model.train_model(train, feature_cols, kind=kind)
        block = test.copy()
        block["predicted_points"] = model.predict(test[feature_cols])
        block["test_gw"] = test_gw
        collected.append(block)

    if not collected:
        raise RuntimeError("no evaluation folds produced -- check season and gw range")
    return pd.concat(collected, ignore_index=True), model


def report(results, label, n_top=50):
    """Print a model-vs-baselines table for one slice of the results."""
    y_true = results["target_points"]
    rows = []

    model_scores = score(y_true, results["predicted_points"])
    model_scores["name"] = "MODEL"
    model_scores["top50"] = top_n_realised(results, results["predicted_points"], n_top)
    rows.append(model_scores)

    for name, preds in naive_baselines(results).items():
        entry = score(y_true, preds)
        entry["name"] = name
        entry["top50"] = top_n_realised(results, preds, n_top)
        rows.append(entry)

    print(f"\n  {label}  (n={len(results):,})")
    print(f"    {'predictor':<18}{'MAE':>7}{'RMSE':>8}{'rank ρ':>9}{f'top{n_top} pts':>11}")
    print(f"    {'-' * 51}")
    for row in rows:
        marker = "*" if row["name"] == "MODEL" else " "
        rho = "  n/a" if np.isnan(row["spearman"]) else f"{row['spearman']:.3f}"
        top = "  n/a" if np.isnan(row["top50"]) else f"{row['top50']:.2f}"
        print(
            f"  {marker} {row['name']:<18}{row['mae']:>7.3f}{row['rmse']:>8.3f}"
            f"{rho:>9}{top:>11}"
        )

    best_naive = min(
        (r for r in rows if r["name"] != "MODEL"), key=lambda r: r["mae"]
    )
    delta = (best_naive["mae"] - model_scores["mae"]) / best_naive["mae"] * 100
    verdict = "beats" if delta > 0 else "LOSES TO"
    print(
        f"    -> model {verdict} best naive ({best_naive['name']}) "
        f"by {abs(delta):.1f}% MAE"
    )
    return model_scores


def main():
    parser = argparse.ArgumentParser(description="Walk-forward evaluation of the FPL model.")
    parser.add_argument("--seasons", nargs="+", default=["2024-25", "2025-26"])
    parser.add_argument("--test-season", default="2025-26")
    parser.add_argument("--first-gw", type=int, default=12)
    parser.add_argument("--last-gw", type=int, default=38)
    parser.add_argument("--step", type=int, default=2, help="evaluate every Nth gameweek")
    parser.add_argument("--model", default=ml_model.DEFAULT_MODEL, choices=sorted(ml_model.MODELS))
    parser.add_argument("--no-shrink", action="store_true",
                        help="disable form shrinkage (for A/B comparison)")
    parser.add_argument("--k-career", type=float, default=features.DEFAULT_K_CAREER)
    parser.add_argument("--k-position", type=float, default=features.DEFAULT_K_POSITION)
    args = parser.parse_args()

    print("=" * 60)
    print("  FPL model evaluation (walk-forward)")
    print("=" * 60)
    print(f"  seasons     : {', '.join(args.seasons)}")
    print(f"  test season : {args.test_season} GW{args.first_gw}-{args.last_gw} (step {args.step})")
    print(f"  model       : {args.model}")

    shrink = not args.no_shrink
    print(f"  shrinkage   : {'off' if args.no_shrink else f'on (k_career={args.k_career}, k_position={args.k_position})'}")

    _, feature_cols, panel = features.prepare(
        args.seasons, shrink=shrink,
        k_career=args.k_career, k_position=args.k_position,
    )
    results, last_model = walk_forward(
        panel, feature_cols, args.test_season,
        args.first_gw, args.last_gw, args.model, args.step,
    )

    print(f"\n  folds       : {results['test_gw'].nunique()}")

    report(results, "ALL PLAYERS")

    # The optimizer only picks among plausible starters, so this is the slice
    # that actually determines squad quality.
    starters = results[results["start_rate_5"] >= 0.5]
    report(starters, "LIKELY STARTERS (started >=50% of last 5)")

    played = results[results["minutes"] >= 60]
    report(played, "PLAYED 60+ MINS LAST GAMEWEEK")

    print("\n  top features:")
    for name, importance in ml_model.feature_importance(last_model, feature_cols, top=10):
        print(f"    {name:<34}{importance:.4f}")


if __name__ == "__main__":
    main()
