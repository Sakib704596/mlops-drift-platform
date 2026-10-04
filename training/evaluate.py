"""
Evaluate a trained model with metrics appropriate for an imbalanced
binary classification problem. Accuracy alone is intentionally NOT
treated as the headline metric here.
"""

from importlib import import_module
from pathlib import Path
metrics_module = import_module("sklearn.metrics")
accuracy_score = metrics_module.accuracy_score
precision_score = metrics_module.precision_score
recall_score = metrics_module.recall_score
f1_score = metrics_module.f1_score
roc_auc_score = metrics_module.roc_auc_score
confusion_matrix = metrics_module.confusion_matrix
classification_report = metrics_module.classification_report

from data import get_train_test_split
from train import MODEL_OUT_PATH


def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # probability of the positive class

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    return metrics, y_pred


def print_report(model, X_test, y_test) -> dict:
    metrics, y_pred = evaluate_model(model, X_test, y_test)

    print("=== Metrics ===")
    for name, value in metrics.items():
        print(f"{name:>10}: {value:.4f}")

    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))
    print("        (rows=actual, cols=predicted; order = [0, 1])")

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=["no default", "default"]))

    return metrics


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_train_test_split()
    model = import_module("joblib").load(MODEL_OUT_PATH)
    print_report(model, X_test, y_test)
