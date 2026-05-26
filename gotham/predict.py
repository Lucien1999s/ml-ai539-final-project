from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.data import align_test_features, load_csv
from src.utils import ensure_dir, read_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Gotham Cabs predictions.")

    parser.add_argument("--test-path", type=str, default="data/gotham/TestFileTemplate.csv")
    parser.add_argument("--model-path", type=str, default="gotham/outputs/model.joblib")
    parser.add_argument("--metadata-path", type=str, default="gotham/outputs/metadata.json")
    parser.add_argument("--output-path", type=str, default="submissions/gotham_predictions.csv")

    parser.add_argument(
        "--template-path",
        type=str,
        default=None,
        help="Optional official/template CSV. If provided, the script fills a prediction column in this file.",
    )
    parser.add_argument(
        "--id-column",
        type=str,
        default=None,
        help="Optional ID column to keep in the output.",
    )
    parser.add_argument(
        "--prediction-column",
        type=str,
        default=None,
        help="Prediction column name. Defaults to the training target name.",
    )
    parser.add_argument(
        "--keep-test-columns",
        action="store_true",
        help="If set, output keeps all test columns and appends predictions.",
    )

    return parser.parse_args()


def build_submission(
    test_df: pd.DataFrame,
    predictions,
    target_column: str,
    output_column: str,
    template_path: str | None,
    id_column: str | None,
    keep_test_columns: bool,
) -> pd.DataFrame:
    if template_path is not None:
        submission = load_csv(template_path)

        if len(submission) != len(predictions):
            raise ValueError(
                f"Template rows ({len(submission)}) do not match predictions ({len(predictions)})."
            )

        if output_column in submission.columns:
            submission[output_column] = predictions
            return submission

        if target_column in submission.columns:
            submission[target_column] = predictions
            return submission

        # Fallback: fill the last column if no obvious prediction column exists.
        submission[submission.columns[-1]] = predictions
        return submission

    if id_column is not None:
        if id_column not in test_df.columns:
            raise ValueError(f"ID column '{id_column}' not found in test file.")

        return pd.DataFrame(
            {
                id_column: test_df[id_column].values,
                output_column: predictions,
            }
        )

    if keep_test_columns:
        submission = test_df.copy()
        submission[output_column] = predictions
        return submission

    return pd.DataFrame({output_column: predictions})


def main() -> None:
    args = parse_args()

    print(f"[Gotham] Loading metadata: {args.metadata_path}")
    metadata = read_json(args.metadata_path)

    target_column = metadata["target"]
    feature_columns = metadata["feature_columns"]
    output_column = args.prediction_column or target_column

    print(f"[Gotham] Loading model: {args.model_path}")
    model = joblib.load(args.model_path)

    print(f"[Gotham] Loading test data: {args.test_path}")
    test_df = load_csv(args.test_path)
    X_test = align_test_features(test_df, feature_columns)

    print(f"[Gotham] Predicting {len(test_df):,} rows...")
    predictions = model.predict(X_test)

    submission = build_submission(
        test_df=test_df,
        predictions=predictions,
        target_column=target_column,
        output_column=output_column,
        template_path=args.template_path,
        id_column=args.id_column,
        keep_test_columns=args.keep_test_columns,
    )

    output_path = Path(args.output_path)
    ensure_dir(output_path.parent)
    submission.to_csv(output_path, index=False)

    print(f"[Gotham] Saved predictions: {output_path}")
    print(f"[Gotham] Output rows: {len(submission):,}")
    print(f"[Gotham] Output columns: {list(submission.columns)}")
    print("[Gotham] Done.")


if __name__ == "__main__":
    main()