"""
Phase 2: MLflow experiment tracking + model registry.

Reuses the exact same training/evaluation logic from train.py / evaluate.py --
this script doesn't change *how* the model is trained, only adds tracking
and registration around it.

Key MLflow concepts used here:
- Experiment: a named group of runs, so you can compare training attempts
  over time (e.g. different hyperparameters) in the UI.
- Run: one execution of training. Every run logs its own params, metrics,
  and artifacts (including the model itself).
- Model Registry: separate from a run -- registering a model creates a
  versioned entry (v1, v2, v3...) under a shared model name, which is what
  later phases (drift-triggered retraining, candidate validation, promotion)
  will read from and write to.
"""

import mlflow
import mlflow.xgboost
from pathlib import Path
from xgboost import XGBClassifier

from data import get_train_test_split, class_balance_report
from train import compute_scale_pos_weight
from evaluate import evaluate_model

# Pin the tracking store to an absolute SQLite database path under the
# project root, so it doesn't matter whether you run this script from
# training/ or the project root -- and so `mlflow ui` can be pointed at
# the exact same database. (MLflow 3.x deprecates the plain file-based
# ./mlruns store in favor of a database backend like SQLite.)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH.as_posix()}")

# Pin where actual model artifacts get saved to an explicit, absolute
# location under the project root -- NOT the default, which is wherever
# the current working directory happens to be when the experiment is
# first created. Without this, running the script from different folders
# (training/ vs project root) scatters artifacts inconsistently.
ARTIFACT_LOCATION = (PROJECT_ROOT / "mlartifacts").as_uri()

EXPERIMENT_NAME = "credit-default-drift-platform"
REGISTERED_MODEL_NAME = "credit-default-classifier"


def get_or_create_experiment():
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        client.create_experiment(EXPERIMENT_NAME, artifact_location=ARTIFACT_LOCATION)
    mlflow.set_experiment(EXPERIMENT_NAME)


def train_and_log(split_random_state: int = 42):
    """
    split_random_state lets a retrain use a different train/test split --
    representing a refreshed data window -- without changing the model's
    hyperparameters. Defaults to 42 to keep existing callers (like
    retrain_pipeline.py) behaving exactly as before.
    """
    get_or_create_experiment()

    X_train, X_test, y_train, y_test = get_train_test_split(random_state=split_random_state)
    scale_pos_weight = compute_scale_pos_weight(y_train)

    params = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": round(float(scale_pos_weight), 4),
        "eval_metric": "logloss",
        "random_state": 42,
    }

    with mlflow.start_run() as run:
        # 1. Log everything needed to reproduce this exact run
        mlflow.log_params(params)
        mlflow.log_dict(class_balance_report(y_train), "train_class_balance.json")

        # 2. Train (identical logic to train.py, just inlined here)
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)

        # 3. Evaluate and log metrics (reusing evaluate.py's logic)
        metrics, _ = evaluate_model(model, X_test, y_test)
        mlflow.log_metrics(metrics)

        # 4. Log + register the model artifact using XGBoost's native MLflow
        # flavor (not mlflow.sklearn) -- this avoids a stricter serialization
        # path in newer MLflow versions that doesn't trust XGBoost's classes
        # by default. `name` replaces the deprecated `artifact_path` param.
        mlflow.xgboost.log_model(
            xgb_model=model,
            name="model",
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        print(f"Run ID: {run.info.run_id}")
        print(f"Experiment: {EXPERIMENT_NAME}")
        print("Params logged:", params)
        print("Metrics logged:", metrics)

    return run


if __name__ == "__main__":
    train_and_log()
