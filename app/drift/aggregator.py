"""
Drift aggregation policy + retraining cooldown.

Turns the raw per-feature detector report (detector.py) into a single
"is retraining actually warranted" decision, and prevents retraining
from firing repeatedly in quick succession.

Policy:
  - A feature only counts as "confirmed drifted" if BOTH KS and PSI
    agree on it. This directly guards against the false positive seen
    in Phase 5 testing (KS alone flagged AGE on pure sampling noise;
    PSI correctly disagreed, so it would NOT count here).
  - Overall drift is "confirmed" if at least `min_pct_features` of
    checked features are confirmed-drifted (default 30%).
  - Even confirmed drift won't trigger retraining again within
    `cooldown_seconds` of the last retrain.
"""

import json
import time
from pathlib import Path

STATE_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "retrain_state.json"
DEFAULT_COOLDOWN_SECONDS = 60 * 60  # 1 hour -- tune for a real deployment


def confirm_drift(report: dict, min_pct_features: float = 0.3) -> dict:
    per_feature = report["per_feature"]
    confirmed_features = [
        f for f, r in per_feature.items() if r["ks_drift"] and r["psi_drift"]
    ]
    pct_confirmed = len(confirmed_features) / report["n_features_checked"]

    return {
        "confirmed": pct_confirmed >= min_pct_features,
        "pct_features_confirmed": round(pct_confirmed, 4),
        "confirmed_features": confirmed_features,
        "threshold": min_pct_features,
    }


def _load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"last_retrain_time": None}


def _save_state(state: dict):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state))


def cooldown_active(cooldown_seconds: int = DEFAULT_COOLDOWN_SECONDS) -> bool:
    state = _load_state()
    last = state.get("last_retrain_time")
    if last is None:
        return False
    return (time.time() - last) < cooldown_seconds


def record_retrain():
    _save_state({"last_retrain_time": time.time()})
