from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.data import build_data_spec, build_preprocessor, load_csv, split_xy
from src.features import add_gotham_features
from src.models import build_pipeline, get_candidate_models
from src.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Feature ablation study for Gotham Cabs."
    )
    parser.add_argument("--train-path", type=str, default="data/gotham/Train.csv")
    parser.add_argument("--target", type=str, default="duration")
    parser.add_argument("--task-type", type=str, default="regression")
    parser.add_argument("--model", type=str, default="xgboost")
    parser.add_argument("--valid-size", type=float, default=0.1)
    parser.add_argument("--max-train-rows", type=int, default=300000)
    parser.add_argument("--output-dir", type=str, default="gotham/outputs/ablation")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def regression_metrics(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def parse_pickup_datetime(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, format="%Y-%m-%d %H:%M:%S", errors="coerce")

    if dt.isna().any():
        fallback = pd.to_datetime(series, format="%m/%d/%y %H:%M", errors="coerce")
        dt = dt.fillna(fallback)

    if dt.isna().any():
        fallback = pd.to_datetime(series, errors="coerce")
        dt = dt.fillna(fallback)

    return dt


def build_raw_coordinates(df: pd.DataFrame, target: str) -> pd.DataFrame:
    cols = [
        "NumberOfPassengers",
        "pickup_x",
        "pickup_y",
        "dropoff_x",
        "dropoff_y",
        target,
    ]
    return df[cols].copy()


def build_basic_time_coordinates(df: pd.DataFrame, target: str) -> pd.DataFrame:
    out = build_raw_coordinates(df, target)

    dt = parse_pickup_datetime(df["pickup_datetime"])

    out["pickup_hour"] = dt.dt.hour
    out["pickup_dayofweek"] = dt.dt.dayofweek
    out["pickup_month"] = dt.dt.month
    out["pickup_day"] = dt.dt.day
    out["is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int)

    return out


def build_full_engineered(df: pd.DataFrame, target: str) -> pd.DataFrame:
    return add_gotham_features(df)


def plot_ablation_results(results: pd.DataFrame, output_path: Path) -> None:
    plot_df = results.sort_values("rmse", ascending=False)

    plt.figure(figsize=(9, 5.5))
    bars = plt.barh(plot_df["feature_set"], plot_df["rmse"])

    plt.title("Feature Engineering Ablation Study")
    plt.xlabel("Validation RMSE (seconds)")
    plt.ylabel("Feature Set")
    plt.grid(True, axis="x", alpha=0.3)

    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + 2,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.2f}",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def run_one_feature_set(
    name: str,
    df_original: pd.DataFrame,
    args: argparse.Namespace,
) -> dict:
    builders = {
        "raw_coordinates": build_raw_coordinates,
        "basic_time_coordinates": build_basic_time_coordinates,
        "full_engineered": build_full_engineered,
    }

    if name not in builders:
        raise ValueError(f"Unknown feature set: {name}")

    print(f"\n[Ablation] Building feature set: {name}")
    df = builders[name](df_original, args.target)

    spec = build_data_spec(df, target=args.target, task_type=args.task_type)
    X, y = split_xy(df, spec)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=args.valid_size,
        random_state=args.random_state,
    )

    candidates = get_candidate_models(spec.task_type)
    if args.model not in candidates:
        raise ValueError(f"Unknown model: {args.model}. Available: {list(candidates)}")

    cfg = candidates[args.model]
    preprocessor = build_preprocessor(spec, scale_numeric=cfg["scale_numeric"])
    pipeline = build_pipeline(preprocessor, cfg["model"])

    print(f"[Ablation] Training {args.model} on {name}")
    start = time.time()
    pipeline.fit(X_train, y_train)
    elapsed = time.time() - start

    print(f"[Ablation] Predicting validation set for {name}")
    pred = pipeline.predict(X_valid)
    pred = np.maximum(pred, 0)

    metrics = regression_metrics(y_valid, pred)

    result = {
        "feature_set": name,
        "model": args.model,
        "num_features": len(spec.feature_columns),
        "numeric_features": len(spec.numeric_columns),
        "categorical_features": len(spec.categorical_columns),
        "train_rows": len(X_train),
        "valid_rows": len(X_valid),
        "elapsed_sec": round(elapsed, 2),
        **metrics,
    }

    print(
        f"[Ablation] {name}: "
        f"RMSE={metrics['rmse']:.4f}, "
        f"MAE={metrics['mae']:.4f}, "
        f"R2={metrics['r2']:.4f}, "
        f"features={len(spec.feature_columns)}"
    )

    return result


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)

    print(f"[Ablation] Loading data: {args.train_path}")
    df = load_csv(args.train_path)

    if args.max_train_rows and len(df) > args.max_train_rows:
        df = df.sample(n=args.max_train_rows, random_state=args.random_state)
        print(f"[Ablation] Sampled rows: {len(df):,}")

    feature_sets = [
        "raw_coordinates",
        "basic_time_coordinates",
        "full_engineered",
    ]

    results = []
    for feature_set in feature_sets:
        result = run_one_feature_set(feature_set, df, args)
        results.append(result)

    results_df = pd.DataFrame(results).sort_values("rmse", ascending=True)

    csv_path = output_dir / "feature_ablation_results.csv"
    results_df.to_csv(csv_path, index=False)

    fig_path = output_dir / "feature_ablation_rmse.png"
    plot_ablation_results(results_df, fig_path)

    print("\n[Ablation] Final results:")
    print(results_df.round(4))
    print(f"\n[Ablation] Saved results to: {csv_path}")
    print(f"[Ablation] Saved figure to: {fig_path}")


if __name__ == "__main__":
    main()
