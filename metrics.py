from typing import Dict, List

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(
    predictions: List[int], targets: List[int]
) -> Dict[str, float]:
    if not predictions or not targets:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    return {
        "accuracy": round(accuracy_score(targets, predictions), 4),
        "precision": round(
            precision_score(
                targets, predictions, average="macro", zero_division=0
            ),
            4,
        ),
        "recall": round(
            recall_score(
                targets, predictions, average="macro", zero_division=0
            ),
            4,
        ),
        "f1": round(
            f1_score(targets, predictions, average="macro", zero_division=0),
            4,
        ),
    }


def classwise_metrics(
    predictions: List[int], targets: List[int], num_classes: int = 3
) -> Dict[str, Dict[str, float]]:
    per_class = {}
    for c in range(num_classes):
        y_true_bin = [1 if t == c else 0 for t in targets]
        y_pred_bin = [1 if p == c else 0 for p in predictions]
        per_class[ID_TO_LABEL[c]] = {
            "precision": round(
                precision_score(y_true_bin, y_pred_bin, zero_division=0), 4
            ),
            "recall": round(
                recall_score(y_true_bin, y_pred_bin, zero_division=0), 4
            ),
            "f1": round(
                f1_score(y_true_bin, y_pred_bin, zero_division=0), 4
            ),
            "support": y_true_bin.count(1),
        }
    return per_class


ID_TO_LABEL = {0: "No Fit", 1: "Fit"}


if __name__ == "__main__":
    preds = [0, 1, 1, 0, 0, 1, 1, 1]
    targets = [0, 1, 0, 1, 0, 1, 0, 1]
    print("Macro metrics:", compute_metrics(preds, targets))
    print("Per-class metrics:", classwise_metrics(preds, targets, 2))
