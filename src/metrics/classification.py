"""Classification metrics with explicit multiclass behavior."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

from src.metrics.calibration import _validate


def classification_metrics(probabilities, labels) -> dict[str, float]:
    probs, target = _validate(probabilities, labels)
    prediction = probs.argmax(axis=1)
    output = {
        "accuracy": float(accuracy_score(target, prediction)),
        "macro_f1": float(f1_score(target, prediction, average="macro", zero_division=0)),
    }
    recalls = recall_score(
        target,
        prediction,
        labels=np.arange(probs.shape[1]),
        average=None,
        zero_division=0,
    )
    output.update({f"recall_class_{index}": float(value) for index, value in enumerate(recalls)})
    try:
        if probs.shape[1] == 2:
            output["auroc"] = float(roc_auc_score(target, probs[:, 1]))
        else:
            output["auroc_ovr_macro"] = float(
                roc_auc_score(
                    target,
                    probs,
                    labels=np.arange(probs.shape[1]),
                    multi_class="ovr",
                    average="macro",
                )
            )
    except ValueError:
        # A small split may lack one class; accuracy/F1 remain defined and AUROC does not.
        output["auroc" if probs.shape[1] == 2 else "auroc_ovr_macro"] = float("nan")
    return output
