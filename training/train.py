"""
Train an XGBoost classifier on the credit default dataset.

This is phase 1: plain training + a saved artifact. No MLflow yet --
that gets wired in during phase 2, on top of this exact training logic.
"""

import pickle
from importlib import import_module
from pathlib import Path
from typing import Any

XGBClassifier: Any = getattr(import_module("xgboost"), "XGBClassifier")

from data import get_train_test_split, class_balance_report

MODEL_OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "model.joblib"


def compute_scale_pos_weight(y_train) -> float:
    """
    XGBoost's scale_pos_weight helps with class imbalance by upweighting
    the minority (positive/default) class during training.
    Rule of thumb: negative_count / positive_count.
    """
    negative = (y_train == 0).sum()
    positive = (y_train == 1).sum()
    return negative / positive


def train_model(X_train, y_train) -> Any:
    scale_pos_weight = compute_scale_pos_weight(y_train)
    print(f"Using scale_pos_weight={scale_pos_weight:.3f} to counter class imbalance")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def save_model(model: Any, path: Path = MODEL_OUT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as model_file:
        pickle.dump(model, model_file)
    print(f"Model saved to {path}")


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_train_test_split()
    print("Train class balance:", class_balance_report(y_train))

    model = train_model(X_train, y_train)
    save_model(model)
