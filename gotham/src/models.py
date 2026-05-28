from __future__ import annotations

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline


def _try_add_lightgbm(models: dict, task_type: str) -> None:
    try:
        if task_type == "regression":
            from lightgbm import LGBMRegressor

            models["lightgbm"] = {
                "model": LGBMRegressor(
                    n_estimators=1000,
                    learning_rate=0.04,
                    num_leaves=64,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1,
                ),
                "scale_numeric": False,
            }
        else:
            from lightgbm import LGBMClassifier

            models["lightgbm"] = {
                "model": LGBMClassifier(
                    n_estimators=1000,
                    learning_rate=0.04,
                    num_leaves=64,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1,
                ),
                "scale_numeric": False,
            }
    except Exception:
        # Keep the pipeline usable even if LightGBM is not installed correctly.
        pass


def _try_add_xgboost(models: dict, task_type: str) -> None:
    try:
        if task_type == "regression":
            from xgboost import XGBRegressor

            models["xgboost"] = {
                "model": XGBRegressor(
                    n_estimators=1200,
                    learning_rate=0.03,
                    max_depth=10,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    min_child_weight=5,
                    reg_alpha=0.1,
                    reg_lambda=2.0,
                    objective="reg:squarederror",
                    random_state=42,
                    n_jobs=-1,
                    tree_method="hist",
                ),
                "scale_numeric": False,
            }
        else:
            from xgboost import XGBClassifier

            models["xgboost"] = {
                "model": XGBClassifier(
                    n_estimators=800,
                    learning_rate=0.04,
                    max_depth=8,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=-1,
                    tree_method="hist",
                ),
                "scale_numeric": False,
            }
    except Exception:
        pass


def get_candidate_models(task_type: str) -> dict:
    if task_type == "regression":
        models = {
            "dummy_mean": {
                "model": DummyRegressor(strategy="mean"),
                "scale_numeric": False,
            },
            "ridge": {
                "model": Ridge(alpha=1.0),
                "scale_numeric": True,
            },
            "random_forest": {
                "model": RandomForestRegressor(
                    n_estimators=250,
                    random_state=42,
                    n_jobs=-1,
                    max_features="sqrt",
                ),
                "scale_numeric": False,
            },
            "extra_trees": {
                "model": ExtraTreesRegressor(
                    n_estimators=350,
                    random_state=42,
                    n_jobs=-1,
                    max_features="sqrt",
                ),
                "scale_numeric": False,
            },
        }
    elif task_type == "classification":
        models = {
            "dummy_most_frequent": {
                "model": DummyClassifier(strategy="most_frequent"),
                "scale_numeric": False,
            },
            "logistic": {
                "model": LogisticRegression(
                    max_iter=1000,
                    n_jobs=-1,
                    class_weight="balanced",
                ),
                "scale_numeric": True,
            },
            "random_forest": {
                "model": RandomForestClassifier(
                    n_estimators=250,
                    random_state=42,
                    n_jobs=-1,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                ),
                "scale_numeric": False,
            },
            "extra_trees": {
                "model": ExtraTreesClassifier(
                    n_estimators=350,
                    random_state=42,
                    n_jobs=-1,
                    max_features="sqrt",
                    class_weight="balanced",
                ),
                "scale_numeric": False,
            },
        }
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    _try_add_lightgbm(models, task_type)
    _try_add_xgboost(models, task_type)

    return models


def build_pipeline(preprocessor, model) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )