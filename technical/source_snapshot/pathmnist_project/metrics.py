import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score

from .common import LABELS, write_json


def classification_metrics(y, probabilities):
    y, p = np.asarray(y), np.asarray(probabilities)
    if y.ndim != 1 or p.shape != (len(y), 9) or not len(y):
        raise ValueError("Expected N labels and an N by 9 probability matrix")
    if not np.isin(y, range(9)).all():
        raise ValueError("Labels must be integers 0 through 8")
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-5):
        raise ValueError("Invalid class probabilities")
    prediction = p.argmax(1)
    cm = confusion_matrix(y, prediction, labels=range(9))
    precision, recall, f1, support = precision_recall_fscore_support(y, prediction, labels=range(9), zero_division=0)
    classes, aucs = [], []
    for c in range(9):
        tp, fn = int(cm[c, c]), int(cm[c].sum() - cm[c, c])
        fp = int(cm[:, c].sum() - cm[c, c])
        tn = len(y) - tp - fn - fp
        binary = (y == c).astype(int)
        auc = float(roc_auc_score(binary, p[:, c])) if 0 < binary.sum() < len(y) else None
        aucs.append(auc)
        classes.append({"label": c, "name": LABELS[c], "precision": float(precision[c]), "recall": float(recall[c]),
                        "f1": float(f1[c]), "support": int(support[c]), "sensitivity": float(recall[c]) if tp + fn else None,
                        "specificity": tn / (tn + fp) if tn + fp else None, "auc_ovr": auc})
    return {"n": len(y), "accuracy": float(accuracy_score(y, prediction)), "macro_f1": float(f1.mean()),
            "macro_auc_ovr": float(np.mean(aucs)) if all(a is not None for a in aucs) else None,
            "auc_note": "null when any class has no positive or no negative samples",
            "confusion_matrix": cm.tolist(), "per_class": classes}


def save_predictions(directory, split, indices, y, p):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / f"{split}_predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "true_label", "predicted_label"] + [f"p_{i}" for i in range(9)])
        for index, label, probabilities in zip(indices, y, p, strict=True):
            writer.writerow([f"{split}:{index}", int(label), int(probabilities.argmax())] + probabilities.tolist())
    metrics = classification_metrics(y, p)
    write_json(directory / f"{split}_metrics.json", metrics)
    return metrics
