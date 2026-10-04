"""
FastAPI serving layer for the credit-default classifier.

Loads the model from the MLflow Model Registry (not a local .joblib file)
so that this service is always serving whatever version we point it at --
this is the piece that later phases (candidate validation, promotion,
GitOps deployment) will update by changing MODEL_VERSION, not by
re-copying files around.
"""

import json
import os
import sys
import time
from pathlib import Path

import mlflow
import mlflow.xgboost
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from schemas import PredictionRequest, PredictionResponse, HealthResponse, ModelInfoResponse

# detector.py / aggregator.py / simulate_drift.py live in app/drift/, a
# sibling folder -- add it to the import path (same pattern used in
# training/retrain_pipeline.py).
sys.path.append(str(Path(__file__).resolve().parent.parent / "drift"))
from detector import detect_drift_report
from aggregator import confirm_drift

# --- MLflow config: must match training/train_with_mlflow.py exactly ---
# Reads from MLFLOW_TRACKING_URI env var if set (used inside Docker);
# otherwise falls back to the same local path used during local dev.
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
if not MLFLOW_TRACKING_URI:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
    MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

MODEL_NAME = "credit-default-classifier"
# Which registry version to serve. Hardcoded here for now; Phase 6's
# promotion step is what will make this dynamic (e.g. always serve
# whichever version is tagged "production").
MODEL_VERSION = os.environ.get("MODEL_VERSION", "1")

# For containerized deployment: if MODEL_ARTIFACT_PATH is set, load the
# model from a local, self-contained folder instead of querying the
# MLflow registry. This avoids baking the whole tracking database (with
# its host-specific absolute artifact paths) into the image -- the
# container serves one fixed, already-exported model version instead.
MODEL_ARTIFACT_PATH = os.environ.get("MODEL_ARTIFACT_PATH")

# --- Prometheus metrics ---
# Application-level (request behavior)
PREDICTION_COUNT = Counter(
    "predictions_total", "Total number of predictions made", ["predicted_class"]
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds", "Time spent processing a prediction request"
)

# ML-specific (model + drift behavior) -- this is what distinguishes
# "infra is healthy" from "the model is actually behaving well."
MODEL_VERSION_INFO = Gauge(
    "model_version_info", "Currently served model version (value is always 1; version is a label)",
    ["model_version"],
)
DRIFT_PCT_FEATURES_CONFIRMED = Gauge(
    "drift_pct_features_confirmed", "Fraction of features confirmed-drifted on the last /drift/check call"
)
DRIFT_FEATURES_CONFIRMED_COUNT = Gauge(
    "drift_features_confirmed_count", "Count of features confirmed-drifted on the last /drift/check call"
)
DRIFT_CONFIRMED = Gauge(
    "drift_confirmed", "1 if the last /drift/check call confirmed drift, else 0"
)
RETRAIN_LAST_TIMESTAMP = Gauge(
    "retrain_last_timestamp_seconds", "Unix timestamp of the last retraining event (from the shared retrain_state.json)"
)

# Paths shared with app/drift/aggregator.py and training/build_reference.py
PROJECT_ROOT_FOR_DATA = Path(__file__).resolve().parent.parent.parent
REFERENCE_PATH = PROJECT_ROOT_FOR_DATA / "data" / "reference" / "reference.csv"
KNOWN_NORMAL_PATH = PROJECT_ROOT_FOR_DATA / "data" / "reference" / "known_normal_sample.csv"
RETRAIN_STATE_PATH = PROJECT_ROOT_FOR_DATA / "data" / "retrain_state.json"

# Tracks the most recent drift check result, returned by GET /drift/status
# without re-running detection every time that endpoint is hit.
last_drift_check = {"status": "no drift check has been run yet"}

app = FastAPI(title="Credit Default Prediction API")

# Loaded once at startup, not per-request -- reloading the model on every
# call would be needlessly slow and is the classic FastAPI+ML mistake.
model = None


@app.on_event("startup")
def load_model():
    global model
    if MODEL_ARTIFACT_PATH:
        model = mlflow.xgboost.load_model(MODEL_ARTIFACT_PATH)
        print(f"Loaded model from local exported path: {MODEL_ARTIFACT_PATH}")
    else:
        model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
        model = mlflow.xgboost.load_model(model_uri)
        print(f"Loaded model '{MODEL_NAME}' version '{MODEL_VERSION}' from {model_uri}")

    MODEL_VERSION_INFO.labels(model_version=MODEL_VERSION).set(1)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok" if model is not None else "model not loaded")


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return ModelInfoResponse(
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        model_stage="not yet tracked (added in Phase 6)",
    )


@app.get("/drift/status")
def drift_status():
    """Returns the result of the most recent /drift/check call, if any."""
    return last_drift_check


@app.get("/drift/check")
def drift_check(simulate: bool = False):
    """
    Runs real KS + PSI drift detection and updates the Prometheus gauges
    so this shows up on a Grafana dashboard, not just in a JSON response.

    simulate=false (default): compares the reference distribution against
      a known-normal held-out sample -- expect little/no drift.
    simulate=true: injects the same synthetic drift used in Phase 5
      testing (halved credit limits, +15 age, +2 payment delay) --
      expect confirmed drift. Useful for demonstrating the dashboard
      actually reacting to a drift event without needing a real one.
    """
    if not REFERENCE_PATH.exists() or not KNOWN_NORMAL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Reference data not found. Run training/build_reference.py first.",
        )

    reference_df = pd.read_csv(REFERENCE_PATH)
    production_df = pd.read_csv(KNOWN_NORMAL_PATH)

    if simulate:
        production_df = production_df.copy()
        production_df["LIMIT_BAL"] = production_df["LIMIT_BAL"] * 0.5
        production_df["AGE"] = production_df["AGE"] + 15
        production_df["PAY_0"] = production_df["PAY_0"] + 2

    report = detect_drift_report(reference_df, production_df)
    drift_decision = confirm_drift(report, min_pct_features=0.1)

    DRIFT_PCT_FEATURES_CONFIRMED.set(drift_decision["pct_features_confirmed"])
    DRIFT_FEATURES_CONFIRMED_COUNT.set(len(drift_decision["confirmed_features"]))
    DRIFT_CONFIRMED.set(1 if drift_decision["confirmed"] else 0)

    global last_drift_check
    last_drift_check = {
        "simulated": simulate,
        "confirmed": drift_decision["confirmed"],
        "pct_features_confirmed": drift_decision["pct_features_confirmed"],
        "confirmed_features": drift_decision["confirmed_features"],
    }
    return last_drift_check


@app.get("/metrics")
def metrics():
    # Refresh the retrain-event gauge from the shared state file on every
    # scrape -- training/retrain_pipeline.py (a separate process) writes
    # to this file, so we re-read it fresh rather than caching in memory.
    if RETRAIN_STATE_PATH.exists():
        state = json.loads(RETRAIN_STATE_PATH.read_text())
        if state.get("last_retrain_time"):
            RETRAIN_LAST_TIMESTAMP.set(state["last_retrain_time"])

    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()

    # Convert to a single-row DataFrame with columns in the exact order
    # the model was trained on.
    input_df = pd.DataFrame([request.model_dump()])

    proba = float(model.predict_proba(input_df)[0][1])
    prediction = int(proba >= 0.5)

    PREDICTION_LATENCY.observe(time.time() - start)
    PREDICTION_COUNT.labels(predicted_class=str(prediction)).inc()

    return PredictionResponse(
        prediction=prediction,
        probability=round(proba, 4),
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
    )