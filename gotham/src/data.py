from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class DataSpec:
    target: str
    task_type: str
    feature_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(obj: dict) -> "DataSpec":
        return DataSpec(
            target=obj["target"],
            task_type=obj["task_type"],
            feature_columns=list(obj["feature_columns"]),
            numeric_columns=list(obj["numeric_columns"]),
            categorical_columns=list(obj["categorical_columns"]),
        )


def load_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"CSV file is empty: {path}")

    return df


def detect_target_column(df: pd.DataFrame, target: str | None = None) -> str:
    if target is not None:
        if target not in df.columns:
            raise ValueError(
                f"Target column '{target}' not found. Available columns: {list(df.columns)}"
            )
        return target

    # Safe default for many class projects: target is the final column.
    # Still, explicit --target is preferred after reading Project Description.pdf.
    return str(df.columns[-1])


def infer_task_type(y: pd.Series, task_type: str | None = None) -> str:
    if task_type is not None:
        if task_type not in {"regression", "classification"}:
            raise ValueError("--task-type must be either 'regression' or 'classification'.")
        return task_type

    y_non_null = y.dropna()
    unique_count = y_non_null.nunique()

    if pd.api.types.is_numeric_dtype(y_non_null) and unique_count > 20:
        return "regression"

    return "classification"


def build_data_spec(
    df: pd.DataFrame,
    target: str | None = None,
    task_type: str | None = None,
) -> DataSpec:
    target_col = detect_target_column(df, target)
    feature_columns = [str(c) for c in df.columns if str(c) != target_col]

    y = df[target_col]
    detected_task_type = infer_task_type(y, task_type)

    X = df[feature_columns]
    numeric_columns = X.select_dtypes(include=[np.number]).columns.astype(str).tolist()
    categorical_columns = [c for c in feature_columns if c not in numeric_columns]

    return DataSpec(
        target=target_col,
        task_type=detected_task_type,
        feature_columns=feature_columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )


def split_xy(df: pd.DataFrame, spec: DataSpec) -> tuple[pd.DataFrame, pd.Series]:
    missing = [c for c in spec.feature_columns + [spec.target] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in training data: {missing}")

    X = df[spec.feature_columns].copy()
    y = df[spec.target].copy()
    return X, y


def align_test_features(test_df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    aligned = test_df.copy()

    for col in feature_columns:
        if col not in aligned.columns:
            aligned[col] = np.nan

    return aligned[feature_columns].copy()


def build_preprocessor(spec: DataSpec, scale_numeric: bool = False) -> ColumnTransformer:
    if scale_numeric:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
    else:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, spec.numeric_columns),
            ("cat", categorical_pipeline, spec.categorical_columns),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )