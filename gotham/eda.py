from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.features import add_gotham_features
from src.data import build_data_spec, load_csv
from src.utils import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run basic EDA for Gotham Cabs.")
    parser.add_argument("--train-path", type=str, default="data/gotham/Train.csv")
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--task-type", type=str, default=None, choices=[None, "regression", "classification"])
    parser.add_argument("--output-dir", type=str, default="gotham/outputs/eda")
    parser.add_argument("--max-rows", type=int, default=200_000)
    return parser.parse_args()


def maybe_sample(df: pd.DataFrame, max_rows: int, random_state: int = 42) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=random_state)


def save_target_distribution(df: pd.DataFrame, target: str, output_dir: Path) -> None:
    plt.figure(figsize=(8, 5))

    if pd.api.types.is_numeric_dtype(df[target]) and df[target].nunique() > 20:
        df[target].hist(bins=50)
        plt.title(f"Target Distribution: {target}")
        plt.xlabel(target)
        plt.ylabel("Count")
    else:
        df[target].value_counts().head(50).plot(kind="bar")
        plt.title(f"Target Class Distribution: {target}")
        plt.xlabel(target)
        plt.ylabel("Count")

    plt.tight_layout()
    plt.savefig(output_dir / "target_distribution.png", dpi=200)
    plt.close()


def save_missing_plot(df: pd.DataFrame, output_dir: Path) -> None:
    missing_rate = df.isna().mean().sort_values(ascending=False)
    missing_rate.to_csv(output_dir / "missing_rate.csv", header=["missing_rate"])

    top_missing = missing_rate.head(20)
    plt.figure(figsize=(10, 5))
    top_missing.plot(kind="bar")
    plt.title("Top Missing Value Rates")
    plt.ylabel("Missing Rate")
    plt.tight_layout()
    plt.savefig(output_dir / "missing_values_top20.png", dpi=200)
    plt.close()


def save_numeric_correlations(df: pd.DataFrame, target: str, output_dir: Path) -> None:
    numeric_df = df.select_dtypes(include="number")

    if target not in numeric_df.columns:
        return

    corr = numeric_df.corr(numeric_only=True)[target].sort_values(key=lambda s: s.abs(), ascending=False)
    corr.to_csv(output_dir / "numeric_correlation_with_target.csv", header=["correlation_with_target"])

    top_corr = corr.drop(labels=[target], errors="ignore").head(20)

    plt.figure(figsize=(10, 5))
    top_corr.plot(kind="bar")
    plt.title("Top Numeric Correlations with Target")
    plt.ylabel("Correlation")
    plt.tight_layout()
    plt.savefig(output_dir / "top_numeric_correlations.png", dpi=200)
    plt.close()


def save_numeric_histograms(df: pd.DataFrame, target: str, output_dir: Path) -> None:
    numeric_cols = [c for c in df.select_dtypes(include="number").columns if c != target]
    selected = numeric_cols[:12]

    for col in selected:
        plt.figure(figsize=(7, 4))
        df[col].hist(bins=40)
        plt.title(f"Histogram: {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(output_dir / f"hist_{col}.png", dpi=200)
        plt.close()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)

    print(f"[Gotham EDA] Loading: {args.train_path}")
    df = load_csv(args.train_path)
    df = add_gotham_features(df)
    df_sample = maybe_sample(df, args.max_rows)

    spec = build_data_spec(df_sample, target=args.target, task_type=args.task_type)

    summary = {
        "rows_full": len(df),
        "rows_used_for_eda": len(df_sample),
        "columns": list(df.columns),
        "target": spec.target,
        "task_type": spec.task_type,
        "num_features": len(spec.feature_columns),
        "num_numeric_features": len(spec.numeric_columns),
        "num_categorical_features": len(spec.categorical_columns),
        "numeric_columns": spec.numeric_columns,
        "categorical_columns": spec.categorical_columns,
    }

    save_json(summary, output_dir / "eda_summary.json")
    df_sample.head(20).to_csv(output_dir / "head_20_rows.csv", index=False)

    save_missing_plot(df_sample, output_dir)
    save_target_distribution(df_sample, spec.target, output_dir)
    save_numeric_correlations(df_sample, spec.target, output_dir)
    save_numeric_histograms(df_sample, spec.target, output_dir)

    print(f"[Gotham EDA] Saved EDA outputs to: {output_dir}")


if __name__ == "__main__":
    main()