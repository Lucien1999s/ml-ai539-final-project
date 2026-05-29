from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split

from src.features import add_gotham_features
from src.data import build_data_spec, build_preprocessor, load_csv, split_xy
from src.metrics import get_cv_scoring, get_human_score_name, score_sort_note
from src.models import build_pipeline, get_candidate_models
from src.utils import ensure_dir, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and compare Gotham Cabs models.")

    parser.add_argument("--train-path", type=str, default="data/gotham/Train.csv")
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--task-type", type=str, default=None, choices=[None, "regression", "classification"])
    parser.add_argument("--output-dir", type=str, default="gotham/outputs")

    parser.add_argument("--sample-frac", type=float, default=0.15)
    parser.add_argument("--max-sample-rows", type=int, default=150_000)
    parser.add_argument("--cv", type=int, default=5)

    parser.add_argument("--final-model", type=str, default="auto")
    parser.add_argument("--random-state", type=int, default=42)

    return parser.parse_args()


def make_sample(
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str,
    sample_frac: float,
    max_sample_rows: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    if sample_frac <= 0 or sample_frac > 1:
        raise ValueError("--sample-frac must be in (0, 1].")

    desired_rows = int(len(X) * sample_frac)
    desired_rows = min(desired_rows, max_sample_rows)
    desired_rows = max(desired_rows, min(len(X), 1000))

    if desired_rows >= len(X):
        return X, y

    train_size = desired_rows / len(X)
    stratify = y if task_type == "classification" and y.nunique() > 1 else None

    X_sample, _, y_sample, _ = train_test_split(
        X,
        y,
        train_size=train_size,
        random_state=random_state,
        stratify=stratify,
    )

    return X_sample, y_sample


def make_cv(task_type: str, cv: int, random_state: int):
    if task_type == "classification":
        return StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    return KFold(n_splits=cv, shuffle=True, random_state=random_state)


def main() -> None:
    args = parse_args()
    set_seed(args.random_state)
    output_dir = ensure_dir(args.output_dir)

    print(f"[Gotham] Loading train data: {args.train_path}")
    df = load_csv(args.train_path)
    df = add_gotham_features(df)

    spec = build_data_spec(df, target=args.target, task_type=args.task_type)
    X, y = split_xy(df, spec)

    print(f"[Gotham] Target column: {spec.target}")
    print(f"[Gotham] Task type: {spec.task_type}")
    print(f"[Gotham] Rows: {len(df):,}")
    print(f"[Gotham] Features: {len(spec.feature_columns)}")
    print(f"[Gotham] Numeric features: {len(spec.numeric_columns)}")
    print(f"[Gotham] Categorical features: {len(spec.categorical_columns)}")

    X_sample, y_sample = make_sample(
        X=X,
        y=y,
        task_type=spec.task_type,
        sample_frac=args.sample_frac,
        max_sample_rows=args.max_sample_rows,
        random_state=args.random_state,
    )

    print(f"[Gotham] Model comparison rows: {len(X_sample):,}")
    print(f"[Gotham] CV scoring: {get_human_score_name(spec.task_type)}")
    print(f"[Gotham] Note: {score_sort_note(spec.task_type)}")

    candidates = get_candidate_models(spec.task_type)
    scoring = get_cv_scoring(spec.task_type)
    cv = make_cv(spec.task_type, args.cv, args.random_state)

    rows = []

    for name, cfg in candidates.items():
        print(f"[Gotham] Running CV: {name}")
        start = time.time()

        preprocessor = build_preprocessor(spec, scale_numeric=cfg["scale_numeric"])
        pipeline = build_pipeline(preprocessor, cfg["model"])

        try:
            scores = cross_val_score(
                pipeline,
                X_sample,
                y_sample,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                error_score="raise",
            )

            elapsed = time.time() - start

            rows.append(
                {
                    "model": name,
                    "mean_cv_score": float(np.mean(scores)),
                    "std_cv_score": float(np.std(scores)),
                    "raw_scores": "|".join(f"{s:.6f}" for s in scores),
                    "elapsed_sec": round(elapsed, 2),
                    "status": "ok",
                    "error": "",
                }
            )
        except Exception as exc:
            elapsed = time.time() - start
            rows.append(
                {
                    "model": name,
                    "mean_cv_score": np.nan,
                    "std_cv_score": np.nan,
                    "raw_scores": "",
                    "elapsed_sec": round(elapsed, 2),
                    "status": "failed",
                    "error": str(exc),
                }
            )
            print(f"[Gotham] Failed: {name} | {exc}")

    leaderboard = pd.DataFrame(rows).sort_values("mean_cv_score", ascending=False)
    leaderboard_path = output_dir / "leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    print(f"[Gotham] Saved leaderboard: {leaderboard_path}")

    valid = leaderboard[leaderboard["status"] == "ok"].dropna(subset=["mean_cv_score"])

    if valid.empty:
        raise RuntimeError("No candidate model completed successfully. Check dependencies and data format.")

    if args.final_model == "auto":
        best_model_name = str(valid.iloc[0]["model"])
    else:
        best_model_name = args.final_model
        if best_model_name not in candidates:
            raise ValueError(f"Unknown --final-model '{best_model_name}'. Available: {list(candidates)}")

    print(f"[Gotham] Final model selected: {best_model_name}")
    final_cfg = candidates[best_model_name]

    final_preprocessor = build_preprocessor(spec, scale_numeric=final_cfg["scale_numeric"])
    final_pipeline = build_pipeline(final_preprocessor, final_cfg["model"])

    print("[Gotham] Fitting final model on the FULL training data...")
    final_pipeline.fit(X, y)

    model_path = output_dir / "model.joblib"
    metadata_path = output_dir / "metadata.json"

    joblib.dump(final_pipeline, model_path)

    save_json(
        {
            "target": spec.target,
            "task_type": spec.task_type,
            "feature_columns": spec.feature_columns,
            "numeric_columns": spec.numeric_columns,
            "categorical_columns": spec.categorical_columns,
            "best_model": best_model_name,
            "train_path": args.train_path,
            "sample_frac_for_model_comparison": args.sample_frac,
            "max_sample_rows_for_model_comparison": args.max_sample_rows,
            "cv": args.cv,
            "scoring": scoring,
            "leaderboard_path": str(leaderboard_path),
        },
        metadata_path,
    )

    print(f"[Gotham] Saved final model: {model_path}")
    print(f"[Gotham] Saved metadata: {metadata_path}")
    print("[Gotham] Done.")


if __name__ == "__main__":
    main()