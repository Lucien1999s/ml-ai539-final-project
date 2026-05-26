from __future__ import annotations


def get_cv_scoring(task_type: str) -> str:
    if task_type == "regression":
        return "neg_root_mean_squared_error"
    if task_type == "classification":
        return "accuracy"
    raise ValueError(f"Unsupported task type: {task_type}")


def get_human_score_name(task_type: str) -> str:
    if task_type == "regression":
        return "RMSE"
    if task_type == "classification":
        return "accuracy"
    raise ValueError(f"Unsupported task type: {task_type}")


def score_sort_note(task_type: str) -> str:
    if task_type == "regression":
        return "scikit-learn uses negative RMSE, so higher CV score is better."
    if task_type == "classification":
        return "Higher accuracy is better."
    raise ValueError(f"Unsupported task type: {task_type}")