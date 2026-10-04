"""
Data loading and preprocessing for the UCI "Default of Credit Card Clients" dataset.

Dataset: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
Also mirrored on Kaggle as "UCI_Credit_Card.csv"

How to get the data (do this once, manually):
  1. Download the CSV from Kaggle: "default-of-credit-card-clients-dataset"
     OR download the .xls from the UCI page and convert to CSV.
  2. Save it as: data/raw/UCI_Credit_Card.csv
     Expected columns include: ID, LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE,
     PAY_0..PAY_6, BILL_AMT1..BILL_AMT6, PAY_AMT1..PAY_AMT6,
     default.payment.next.month  (this is the target)

Why this dataset: it's a realistic tabular binary classification problem
(will this customer default next month?) with a meaningful class imbalance
(~22% positive class), which is exactly the kind of problem where "just
look at accuracy" gives a false sense of security.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

RAW_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "UCI_Credit_Card.csv"
TARGET_COL = "default.payment.next.month"


RENAME_MAP = {
    "Unnamed: 0": "ID",
    "X1": "LIMIT_BAL",
    "X2": "SEX",
    "X3": "EDUCATION",
    "X4": "MARRIAGE",
    "X5": "AGE",
    "X6": "PAY_0",
    "X7": "PAY_2",
    "X8": "PAY_3",
    "X9": "PAY_4",
    "X10": "PAY_5",
    "X11": "PAY_6",
    "X12": "BILL_AMT1",
    "X13": "BILL_AMT2",
    "X14": "BILL_AMT3",
    "X15": "BILL_AMT4",
    "X16": "BILL_AMT5",
    "X17": "BILL_AMT6",
    "X18": "PAY_AMT1",
    "X19": "PAY_AMT2",
    "X20": "PAY_AMT3",
    "X21": "PAY_AMT4",
    "X22": "PAY_AMT5",
    "X23": "PAY_AMT6",
    "Y": TARGET_COL,
}


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw CSV. Raises a clear error if the file hasn't been downloaded yet.

    Handles both the Kaggle mirror (already has readable column names) and the
    original UCI export (generic X1..X23, Y column names) by renaming the
    latter to match, so the rest of the pipeline works either way.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Download 'UCI_Credit_Card.csv' (Kaggle mirror of the UCI Default of "
            "Credit Card Clients dataset) and place it at that path."
        )
    df = pd.read_csv(path)

    # If this is the raw UCI export (generic X1..X23, Y headers), rename them.
    if "Y" in df.columns and TARGET_COL not in df.columns:
        df = df.rename(columns=RENAME_MAP)

    # Defensive cleanup: the raw UCI export sometimes carries a stray duplicate
    # header row as actual data (e.g. "ID,LIMIT_BAL,SEX,..." showing up as a
    # literal data row). That single non-numeric row forces every column to
    # be parsed as dtype 'object', even after we later drop the bad row.
    # Fix: coerce every column to numeric, drop any row that fails, then
    # dtypes will be correctly numeric again.
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna()
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    df = df.reset_index(drop=True)

    return df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split into features (X) and target (y).
    Drops the ID column (not predictive, just a row identifier).
    """
    df = df.copy()
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])
    return X, y


def get_train_test_split(
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Returns X_train, X_test, y_train, y_test.

    stratify=y keeps the same class ratio (~22% default) in both splits --
    important on an imbalanced dataset, otherwise a random split could
    accidentally starve the test set of positive examples.
    """
    df = load_raw_data()
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def class_balance_report(y: pd.Series) -> dict:
    """Quick sanity check on class imbalance -- log/print this before training."""
    counts = y.value_counts(normalize=True)
    return {
        "n_samples": len(y),
        "positive_rate": round(float(counts.get(1, 0.0)), 4),
        "negative_rate": round(float(counts.get(0, 0.0)), 4),
    }


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_train_test_split()
    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
    print("Train class balance:", class_balance_report(y_train))
    print("Test class balance:", class_balance_report(y_test))
