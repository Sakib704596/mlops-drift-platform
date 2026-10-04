"""
Candidate validation and promotion.

Compares a newly trained "candidate" model version against whichever
version currently holds the "production" alias in the MLflow registry.
Promotion only happens if the candidate does not regress on any metric
and improves the primary metric -- a candidate that's merely different,
or worse, leaves production untouched.

This is the governance step the project spec calls out as the key
design decision: never deploy a retrained model automatically just
because drift occurred.
"""

import mlflow
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH.as_posix()}")

MODEL_NAME = "credit-default-classifier"
PRODUCTION_ALIAS = "production"
PRIMARY_METRIC = "roc_auc"  # the metric that must improve for promotion

client = mlflow.MlflowClient()


def get_production_version():
    try:
        return client.get_model_version_by_alias(MODEL_NAME, PRODUCTION_ALIAS)
    except mlflow.exceptions.MlflowException:
        return None


def get_run_metrics(run_id: str) -> dict:
    return client.get_run(run_id).data.metrics


def bootstrap_production(version: str = "1"):
    """One-time setup: tag an existing version as production if none is tagged yet."""
    existing = get_production_version()
    if existing is None:
        client.set_registered_model_alias(MODEL_NAME, PRODUCTION_ALIAS, version)
        print(f"Bootstrapped: version {version} tagged as '{PRODUCTION_ALIAS}'")
    else:
        print(f"Production alias already set to version {existing.version}; no change made.")


def evaluate_candidate(candidate_version: str) -> dict:
    """
    Compares candidate_version against whatever is currently tagged
    'production'. Returns a decision dict only -- does NOT change the
    registry. Call promote() separately to act on a 'promote' decision.
    """
    production_mv = get_production_version()
    if production_mv is None:
        return {"decision": "no_production_model", "reason": "No production model tagged yet. Run bootstrap_production() first."}

    candidate_mv = client.get_model_version(MODEL_NAME, candidate_version)

    prod_metrics = get_run_metrics(production_mv.run_id)
    cand_metrics = get_run_metrics(candidate_mv.run_id)

    comparison = {}
    regressions = []
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        prod_val = prod_metrics.get(metric)
        cand_val = cand_metrics.get(metric)
        comparison[metric] = {"production": prod_val, "candidate": cand_val}
        if cand_val is not None and prod_val is not None and cand_val < prod_val:
            regressions.append(metric)

    primary_improved = cand_metrics.get(PRIMARY_METRIC, 0) > prod_metrics.get(PRIMARY_METRIC, 0)

    if regressions:
        decision = "reject"
        reason = f"Candidate regressed on: {', '.join(regressions)}"
    elif not primary_improved:
        decision = "reject"
        reason = f"Candidate did not improve primary metric ({PRIMARY_METRIC})"
    else:
        decision = "promote"
        reason = f"Candidate improved {PRIMARY_METRIC} with no regressions on any metric"

    return {
        "decision": decision,
        "reason": reason,
        "comparison": comparison,
        "production_version": production_mv.version,
        "candidate_version": candidate_version,
    }


def promote(candidate_version: str):
    client.set_registered_model_alias(MODEL_NAME, PRODUCTION_ALIAS, candidate_version)
    print(f"Promoted version {candidate_version} to '{PRODUCTION_ALIAS}'")


if __name__ == "__main__":
    # One-time bootstrap -- run this once before using retrain_pipeline.py,
    # so there's an existing "production" model to compare candidates against.
    bootstrap_production(version="1")
