from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.data import build_data_spec, build_preprocessor, load_csv, split_xy
from src.features import add_gotham_features
from src.models import build_pipeline, get_candidate_models
from src.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Holdout validation for Gotham Cabs.")
    parser.add_argument("--train-path", type=str, default="data/gotham/Train.csv")
    parser.add_argument("--target", type=str, default="duration")
    parser.add_argument("--task-type", type=str, default="regression")
    parser.add_argument("--model", type=str, default="xgboost")
    parser.add_argument("--valid-size", type=float, default=0.1)
    parser.add_argument("--max-train-rows", type=int, default=300000)
    parser.add_argument("--output-dir", type=str, default="gotham/outputs/validation")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--log-target",
        action="store_true",
        help="Train on log1p(duration), then inverse-transform predictions with expm1.",
    )
    return parser.parse_args()


def regression_metrics(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def save_feature_importance(pipeline, feature_columns: list[str], output_path: Path) -> None:
    model = pipeline.named_steps["model"]

    if not hasattr(model, "feature_importances_"):
        return

    importances = model.feature_importances_

    # Since all Gotham features are numeric and no one-hot expansion is used,
    # feature names align with feature_columns.
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

    print(f"[Validate] Loading data: {args.train_path}")
    df = load_csv(args.train_path)
    df = add_gotham_features(df)

    if args.max_train_rows and len(df) > args.max_train_rows:
        df = df.sample(n=args.max_train_rows, random_state=args.random_state)
        print(f"[Validate] Sampled rows: {len(df):,}")

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

    print(f"[Validate] Training model: {args.model}")
    if args.log_target:
        y_train_fit = np.log1p(y_train)
    else:
        y_train_fit = y_train

    pipeline.fit(X_train, y_train_fit)

    print("[Validate] Predicting validation set...")
    pred = pipeline.predict(X_valid)

    if args.log_target:
        pred = np.expm1(pred)
        pred = np.maximum(pred, 0)

    metrics = regression_metrics(y_valid, pred)
    print("[Validate] Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    pd.DataFrame([{"model": args.model, **metrics}]).to_csv(
        output_dir / f"{args.model}_metrics.csv",
        index=False,
    )

    pred_df = pd.DataFrame(
        {
            "y_true": y_valid.values,
            "y_pred": pred,
            "error": y_valid.values - pred,
            "abs_error": np.abs(y_valid.values - pred),
        }
    )
    pred_df.to_csv(output_dir / f"{args.model}_validation_predictions.csv", index=False)

    save_feature_importance(
        pipeline,
        spec.feature_columns,
        output_dir / f"{args.model}_feature_importance.csv",
    )

    joblib.dump(pipeline, output_dir / f"{args.model}_validation_model.joblib")

    print(f"[Validate] Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()