from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.data import build_data_spec, build_preprocessor, load_csv, split_xy
from src.features import add_gotham_features
from src.models import build_pipeline
from src.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune XGBoost for Gotham Cabs.")
    parser.add_argument("--train-path", type=str, default="data/gotham/Train.csv")
    parser.add_argument("--target", type=str, default="duration")
    parser.add_argument("--task-type", type=str, default="regression")
    parser.add_argument("--valid-size", type=float, default=0.1)
    parser.add_argument("--max-train-rows", type=int, default=300000)
    parser.add_argument("--output-dir", type=str, default="gotham/outputs/tuning")
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


def get_xgboost_configs(random_state: int) -> list[dict]:
    base = {
        "objective": "reg:squarederror",
        "random_state": random_state,
        "n_jobs": -1,
        "tree_method": "hist",
    }

    configs = [
        {
            "name": "baseline",
            **base,
            "n_estimators": 800,
            "learning_rate": 0.04,
            "max_depth": 8,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 1,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        },
        {
            "name": "more_trees_lower_lr",
            **base,
            "n_estimators": 1200,
            "learning_rate": 0.03,
            "max_depth": 8,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 1,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        },
        {
            "name": "shallower_depth",
            **base,
            "n_estimators": 1000,
            "learning_rate": 0.04,
            "max_depth": 6,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 1,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        },
        {
            "name": "deeper_depth",
            **base,
            "n_estimators": 1000,
            "learning_rate": 0.03,
            "max_depth": 10,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 1,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        },
        {
            "name": "child_weight_5",
            **base,
            "n_estimators": 1000,
            "learning_rate": 0.04,
            "max_depth": 8,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 5,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        },
        {
            "name": "regularized",
            **base,
            "n_estimators": 1000,
            "learning_rate": 0.04,
            "max_depth": 8,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "reg_alpha": 0.1,
            "reg_lambda": 2.0,
        },
        {
            "name": "high_capacity_regularized",
            **base,
            "n_estimators": 1200,
            "learning_rate": 0.03,
            "max_depth": 10,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_weight": 5,
            "reg_alpha": 0.1,
            "reg_lambda": 2.0,
        },
    ]

    return configs


def make_xgboost_model(config: dict):
    from xgboost import XGBRegressor

    params = {k: v for k, v in config.items() if k != "name"}
    return XGBRegressor(**params)


def save_feature_importance(pipeline, feature_columns: list[str], output_path: Path) -> None:
    model = pipeline.named_steps["model"]

    if not hasattr(model, "feature_importances_"):
        return

    importances = model.feature_importances_

    if len(importances) != len(feature_columns):
        names = [f"feature_{i}" for i in range(len(importances))]
    else:
        names = feature_columns

    fi = pd.DataFrame(
        {
            "feature": names,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)

    fi.to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)

    print(f"[Tune XGBoost] Loading data: {args.train_path}")
    df = load_csv(args.train_path)
    df = add_gotham_features(df)

    if args.max_train_rows and len(df) > args.max_train_rows:
        df = df.sample(n=args.max_train_rows, random_state=args.random_state)
        print(f"[Tune XGBoost] Sampled rows: {len(df):,}")

    spec = build_data_spec(df, target=args.target, task_type=args.task_type)
    X, y = split_xy(df, spec)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=args.valid_size,
        random_state=args.random_state,
    )

    configs = get_xgboost_configs(args.random_state)
    rows = []

    best_rmse = float("inf")
    best_pipeline = None
    best_config = None

    for config in configs:
        name = config["name"]
        print(f"\n[Tune XGBoost] Training config: {name}")
        start = time.time()

        model = make_xgboost_model(config)
        preprocessor = build_preprocessor(spec, scale_numeric=False)
        pipeline = build_pipeline(preprocessor, model)

        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_valid)

        metrics = regression_metrics(y_valid, pred)
        elapsed = time.time() - start

        row = {
            "name": name,
            **metrics,
            "elapsed_sec": round(elapsed, 2),
            "params": json.dumps({k: v for k, v in config.items() if k != "name"}),
        }
        rows.append(row)

        print(
            f"[Tune XGBoost] {name}: "
            f"RMSE={metrics['rmse']:.4f}, "
            f"MAE={metrics['mae']:.4f}, "
            f"R2={metrics['r2']:.4f}, "
            f"time={elapsed:.2f}s"
        )

        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            best_pipeline = pipeline
            best_config = config

    results = pd.DataFrame(rows).sort_values("rmse", ascending=True)
    results_path = output_dir / "xgboost_tuning_results.csv"
    results.to_csv(results_path, index=False)

    if best_pipeline is None or best_config is None:
        raise RuntimeError("No XGBoost config succeeded.")

    best_name = best_config["name"]
    print(f"\n[Tune XGBoost] Best config: {best_name}")
    print(f"[Tune XGBoost] Best RMSE: {best_rmse:.4f}")

    joblib.dump(best_pipeline, output_dir / "best_xgboost_validation_model.joblib")

    with (output_dir / "best_xgboost_config.json").open("w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=2)

    save_feature_importance(
        best_pipeline,
        spec.feature_columns,
        output_dir / "best_xgboost_feature_importance.csv",
    )

    print(f"[Tune XGBoost] Saved results: {results_path}")
    print(f"[Tune XGBoost] Saved best config: {output_dir / 'best_xgboost_config.json'}")
    print("[Tune XGBoost] Done.")


if __name__ == "__main__":
    main()