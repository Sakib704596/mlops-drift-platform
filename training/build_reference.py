"""
Builds the reference distribution used for drift detection.

The "reference" is simply the training feature set (X_train) -- it
represents what "normal" input data looked like when the model was
trained. Later, production data gets compared against this reference
using KS and PSI (app/drift/detector.py).

We also save X_test separately as a convenient "known normal" production
sample for the simulation script (training/simulate_drift.py), since it
came from the same distribution as the reference but wasn't seen during
training -- a good sanity check that normal data does NOT trigger drift.
"""

from pathlib import Path
from data import get_train_test_split

REFERENCE_DIR = Path(__file__).resolve().parent.parent / "data" / "reference"


def build_and_save_reference():
    X_train, X_test, y_train, y_test = get_train_test_split()

    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(REFERENCE_DIR / "reference.csv", index=False)
    X_test.to_csv(REFERENCE_DIR / "known_normal_sample.csv", index=False)

    print(f"Reference distribution saved: {len(X_train)} rows -> {REFERENCE_DIR / 'reference.csv'}")
    print(f"Known-normal sample saved: {len(X_test)} rows -> {REFERENCE_DIR / 'known_normal_sample.csv'}")


if __name__ == "__main__":
    build_and_save_reference()
