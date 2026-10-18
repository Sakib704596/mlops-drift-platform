"""
End-to-end automated retraining pipeline -- the "closed loop" this whole
project is building toward:

  1. Compare incoming production data against the reference distribution.
  2. Aggregate detector results into a confirmed-drift decision
     (both KS and PSI must agree per feature; 30%+ of features confirmed).
  3. Respect the retraining cooldown (don't retrain repeatedly).
  4. If warranted: retrain, which registers a new candidate model version.
  5. Compare the candidate against the current production model.
  6. Promote only if the candidate is strictly better; otherwise reject
     and leave production exactly as it was.
"""

import sys
from pathlib import Path

# detector.py / aggregator.py / simulate_drift.py live in app/drift/, not
# alongside this script -- add that folder to the import path.
sys.path.append(str(Path(__file__).resolve().parent.parent / "app" / "drift"))

import pandas as pd
import mlflow

from detector import detect_drift_report
from aggregator import confirm_drift, cooldown_active, record_retrain
from train_with_mlflow import train_and_log
from promotion import evaluate_candidate, promote

REFERENCE_DIR = Path(__file__).resolve().parent.parent / "data" / "reference"


def run_pipeline(production_df: pd.DataFrame):
    reference_df = pd.read_csv(REFERENCE_DIR / "reference.csv")

    print("Step 1: Checking for drift...")
    report = detect_drift_report(reference_df, production_df)
    # With only 23 total features, the default 30% threshold needs 7+
    # features drifting together -- too strict for a 3-feature shift
    # like our simulation. Lowered to 10% here; in a real deployment
    # this threshold should be tuned based on how many features tend
    # to move together for genuine drift events in your domain.
    drift_decision = confirm_drift(report, min_pct_features=0.1)
    print(f"  Confirmed drift: {drift_decision['confirmed']} "
          f"({drift_decision['pct_features_confirmed'] * 100:.1f}% of features "
          f"confirmed by both detectors, threshold {drift_decision['threshold'] * 100:.0f}%)")
    if drift_decision["confirmed_features"]:
        print(f"  Confirmed-drifted features: {drift_decision['confirmed_features']}")

    if not drift_decision["confirmed"]:
        print("\nNo confirmed drift. Pipeline stops here -- no retrain needed.")
        return

    if cooldown_active():
        print("\nDrift confirmed, but retrain cooldown is still active. Skipping retrain.")
        return

    print("\nStep 2: Drift confirmed and cooldown clear. Retraining...")
    run = train_and_log()
    record_retrain()

    client = mlflow.MlflowClient()
    candidate_version = None
    for mv in client.search_model_versions(f"run_id='{run.info.run_id}'"):
        candidate_version = mv.version
    if candidate_version is None:
        print("Could not locate the newly registered candidate version. Aborting.")
        return

    print(f"\nStep 3: Evaluating candidate (version {candidate_version}) against production...")
    decision = evaluate_candidate(candidate_version)
    print(f"  Decision: {decision['decision']}")
    print(f"  Reason: {decision['reason']}")
    for metric, vals in decision.get("comparison", {}).items():
        print(f"    {metric}: production={vals['production']:.4f}  candidate={vals['candidate']:.4f}")

    if decision["decision"] == "promote":
        promote(candidate_version)
    else:
        print(f"  Candidate version {candidate_version} rejected. Production unchanged.")


if __name__ == "__main__":
    # Demo: reuse Phase 5's synthetic drift injection as a stand-in for a
    # real incoming production batch.
    from simulate_drift import load_known_normal_sample, inject_synthetic_drift

    production_batch = inject_synthetic_drift(load_known_normal_sample())
    run_pipeline(production_batch)
