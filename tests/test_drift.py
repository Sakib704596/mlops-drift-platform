"""
Unit tests for the drift detection logic (KS + PSI).

Deliberately uses synthetic data, not the real credit-default dataset --
this is what lets these tests run in CI without needing the dataset
checked into Git or downloaded at build time. They test the DETECTION
LOGIC itself (does it correctly distinguish "same distribution" from
"shifted distribution"), which is a property that holds regardless of
what the real data looks like.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent / "app" / "drift"))
from detector import compute_ks, compute_psi, detect_drift_for_feature, detect_drift_report
from aggregator import confirm_drift


def test_ks_no_drift_on_identical_distribution():
    np.random.seed(0)
    reference = pd.Series(np.random.normal(0, 1, 1000))
    production = pd.Series(np.random.normal(0, 1, 1000))

    result = compute_ks(reference, production)
    assert result["ks_p_value"] > 0.05  # should NOT reject the null hypothesis


def test_ks_detects_drift_on_shifted_distribution():
    np.random.seed(0)
    reference = pd.Series(np.random.normal(0, 1, 1000))
    production = pd.Series(np.random.normal(5, 1, 1000))  # shifted mean

    result = compute_ks(reference, production)
    assert result["ks_p_value"] < 0.05  # should reject -- distributions differ


def test_psi_near_zero_for_identical_distribution():
    np.random.seed(1)
    reference = pd.Series(np.random.normal(0, 1, 1000))
    production = pd.Series(np.random.normal(0, 1, 1000))

    psi = compute_psi(reference, production)
    assert psi < 0.1  # "no significant shift" heuristic threshold


def test_psi_high_for_shifted_distribution():
    np.random.seed(1)
    reference = pd.Series(np.random.normal(0, 1, 1000))
    production = pd.Series(np.random.normal(3, 1, 1000))

    psi = compute_psi(reference, production)
    assert psi > 0.2  # "significant shift" heuristic threshold


def test_detect_drift_report_flags_shifted_feature():
    np.random.seed(2)
    reference_df = pd.DataFrame({
        "stable_feature": np.random.normal(0, 1, 500),
        "drifted_feature": np.random.normal(0, 1, 500),
    })
    production_df = pd.DataFrame({
        "stable_feature": np.random.normal(0, 1, 500),       # unchanged
        "drifted_feature": np.random.normal(4, 1, 500),       # shifted
    })

    report = detect_drift_report(reference_df, production_df)
    assert report["per_feature"]["drifted_feature"]["ks_drift"] is True
    assert report["per_feature"]["drifted_feature"]["psi_drift"] is True
    assert report["per_feature"]["stable_feature"]["ks_drift"] is False


def test_confirm_drift_requires_both_detectors_to_agree():
    # Simulates the Phase 5 false-positive case: KS flags it, PSI doesn't.
    report = {
        "n_features_checked": 2,
        "per_feature": {
            "feature_a": {"ks_drift": True, "psi_drift": False},
            "feature_b": {"ks_drift": False, "psi_drift": False},
        },
    }
    decision = confirm_drift(report, min_pct_features=0.3)
    assert decision["confirmed"] is False  # neither feature has BOTH flags true
    assert decision["confirmed_features"] == []


def test_confirm_drift_fires_when_both_detectors_agree():
    report = {
        "n_features_checked": 2,
        "per_feature": {
            "feature_a": {"ks_drift": True, "psi_drift": True},
            "feature_b": {"ks_drift": False, "psi_drift": False},
        },
    }
    decision = confirm_drift(report, min_pct_features=0.3)
    assert decision["confirmed"] is True
    assert decision["confirmed_features"] == ["feature_a"]
