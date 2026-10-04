"""
Validates that the drift detectors work correctly by testing them against
two scenarios:
  1. A "known normal" production batch (X_test -- same distribution as
     training, just held out). Expect: little to no drift detected.
  2. A deliberately shifted production batch (synthetic drift injected).
     Expect: drift detected on the features we actually perturbed.

This is exactly the "deliberate failure & validation experiment" pattern
from the project spec (section 18) -- you don't trust a detector until
you've proven it fires on real drift and stays quiet on normal data.
"""

from pathlib import Path
import pandas as pd

from detector import detect_drift_report

REFERENCE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "reference"


def load_reference() -> pd.DataFrame:
    return pd.read_csv(REFERENCE_DIR / "reference.csv")


def load_known_normal_sample() -> pd.DataFrame:
    return pd.read_csv(REFERENCE_DIR / "known_normal_sample.csv")


def inject_synthetic_drift(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulates a real-world shift: a economic downturn lowering credit
    limits, an aging customer base, and a rise in recent payment delays.
    These are plausible, explainable shifts -- not random noise -- which
    matters if you're asked "why did you choose this perturbation" in
    an interview.
    """
    drifted = df.copy()
    drifted["LIMIT_BAL"] = drifted["LIMIT_BAL"] * 0.5       # credit limits cut in half
    drifted["AGE"] = drifted["AGE"] + 15                     # customer base aged significantly
    drifted["PAY_0"] = drifted["PAY_0"] + 2                  # more recent payment delays
    return drifted


def print_summary(label: str, report: dict):
    print(f"\n=== {label} ===")
    print(f"Features checked: {report['n_features_checked']}")
    print(f"KS drift:  {report['n_features_ks_drifted']} features "
          f"({report['pct_features_ks_drifted'] * 100:.1f}%)")
    print(f"PSI drift: {report['n_features_psi_drifted']} features "
          f"({report['pct_features_psi_drifted'] * 100:.1f}%)")

    drifted_features = [
        f for f, r in report["per_feature"].items() if r["ks_drift"] or r["psi_drift"]
    ]
    if drifted_features:
        print("Flagged features:")
        for f in drifted_features:
            r = report["per_feature"][f]
            print(f"  {f}: ks_p={r['ks_p_value']:.4f} (drift={r['ks_drift']}), "
                  f"psi={r['psi']:.4f} (drift={r['psi_drift']})")
    else:
        print("No features flagged.")


if __name__ == "__main__":
    reference = load_reference()
    normal_production = load_known_normal_sample()
    drifted_production = inject_synthetic_drift(normal_production)

    normal_report = detect_drift_report(reference, normal_production)
    drifted_report = detect_drift_report(reference, drifted_production)

    print_summary("Scenario 1: Known-normal production (expect little/no drift)", normal_report)
    print_summary("Scenario 2: Synthetically drifted production (expect drift)", drifted_report)
