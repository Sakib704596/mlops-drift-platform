"""
Drift detection: KS test and PSI, computed per feature.

These are pure functions -- they work on any two pandas Series/DataFrames
passed in, with no dependency on where the data came from. This keeps
drift detection decoupled from training, serving, or simulation code,
which matters once Phase 6 wires this into an automated retraining loop.

Note what this does NOT do yet: decide whether drift is "confirmed" and
should trigger retraining. That aggregation/cooldown policy is Phase 6.
This module only answers "did this feature's distribution shift" --
one detector's opinion, not the final decision.
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def compute_ks(reference: pd.Series, production: pd.Series) -> dict:
    """
    KS test: compares the two samples' cumulative distributions.
    Null hypothesis: both samples come from the same distribution.
    A small p-value means we reject that -- the distributions likely differ.
    """
    statistic, p_value = ks_2samp(reference.dropna(), production.dropna())
    return {"ks_statistic": float(statistic), "ks_p_value": float(p_value)}


def compute_psi(reference: pd.Series, production: pd.Series, buckets: int = 10) -> float:
    """
    Population Stability Index.

    Bins are built from the REFERENCE distribution's quantiles (not
    production's) -- production is then measured against those same
    fixed bins. This is important: PSI measures how much production has
    shifted *relative to what training considered normal*, not just
    "are these two samples different from each other" in general.

    Heuristic reading (not a hard law):
      < 0.1  -> no significant shift
      0.1-0.2 -> moderate shift, worth watching
      > 0.2  -> significant shift
    """
    reference = reference.dropna()
    production = production.dropna()

    # Build bucket edges from reference quantiles. duplicates="drop" handles
    # low-cardinality features (e.g. SEX: 1/2) where many quantiles collide.
    quantiles = np.linspace(0, 1, buckets + 1)
    bin_edges = np.quantile(reference, quantiles)
    bin_edges = np.unique(bin_edges)  # drop duplicate edges
    if len(bin_edges) < 2:
        # Reference has (near-)zero variance -- PSI isn't meaningful here.
        return 0.0
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    prod_counts, _ = np.histogram(production, bins=bin_edges)

    ref_pct = ref_counts / max(len(reference), 1)
    prod_pct = prod_counts / max(len(production), 1)

    # Small epsilon avoids log(0) / division by zero for empty buckets.
    eps = 1e-6
    ref_pct = np.clip(ref_pct, eps, None)
    prod_pct = np.clip(prod_pct, eps, None)

    psi = float(np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct)))
    return psi


def detect_drift_for_feature(
    reference: pd.Series,
    production: pd.Series,
    alpha: float = 0.05,
    psi_threshold: float = 0.1,
) -> dict:
    """Run both detectors on a single feature and flag drift per-detector."""
    ks_result = compute_ks(reference, production)
    psi_value = compute_psi(reference, production)

    return {
        **ks_result,
        "ks_drift": ks_result["ks_p_value"] < alpha,
        "psi": psi_value,
        "psi_drift": psi_value > psi_threshold,
    }


def detect_drift_report(
    reference_df: pd.DataFrame,
    production_df: pd.DataFrame,
    features: list[str] | None = None,
    alpha: float = 0.05,
    psi_threshold: float = 0.1,
) -> dict:
    """
    Runs drift detection across all (or selected) features and summarizes
    results. This is a REPORT, not a retraining decision -- Phase 6 will
    apply its own policy (e.g. "2+ detectors agree AND 30%+ features
    drifted") on top of this report's output.
    """
    if features is None:
        features = list(reference_df.columns)

    per_feature = {}
    for feature in features:
        per_feature[feature] = detect_drift_for_feature(
            reference_df[feature], production_df[feature], alpha, psi_threshold
        )

    n_features = len(features)
    n_ks_drifted = sum(1 for r in per_feature.values() if r["ks_drift"])
    n_psi_drifted = sum(1 for r in per_feature.values() if r["psi_drift"])

    return {
        "per_feature": per_feature,
        "n_features_checked": n_features,
        "n_features_ks_drifted": n_ks_drifted,
        "n_features_psi_drifted": n_psi_drifted,
        "pct_features_ks_drifted": round(n_ks_drifted / n_features, 4),
        "pct_features_psi_drifted": round(n_psi_drifted / n_features, 4),
    }
