from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support


def evaluate_binary_classifier(
    y_true,
    y_proba,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Standard binary classification evaluation.

    Used by:
      - Step 4.5: model comparison
      - Step 4.6: threshold optimization
      - Step 4.9: offline A/B evaluation

    Parameters
    ----------
    y_true : array-like
        Ground truth labels (0/1)

    y_proba : array-like
        Predicted probabilities for class=1

    threshold : float
        Decision threshold

    Returns
    -------
    dict
        {
            "auc": float,
            "precision": float,
            "recall": float,
            "f1": float
        }
    """
    y_proba = np.asarray(y_proba, dtype=float)

    auc = float(roc_auc_score(y_true, y_proba))

    preds = (y_proba >= float(threshold)).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        preds,
        average="binary",
        zero_division=0,
    )

    return {
        "auc": auc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
