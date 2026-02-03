from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix


def threshold_report(
    y_true,
    y_proba,
    fp_cost: float,
    fn_cost: float,
    thresholds: Optional[Iterable[float]] = None,
) -> Dict[str, float]:
    """
    Pick the best threshold using business cost.

    total_cost = FP * fp_cost + FN * fn_cost

    Returns a compact report including:
    - best threshold
    - PR-AUC
    - confusion matrix counts
    - total cost
    """
    y_proba = np.asarray(y_proba, dtype=float)
    y_true = np.asarray(y_true, dtype=int)

    pr_auc = float(average_precision_score(y_true, y_proba))

    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    best = None
    for t in thresholds:
        preds = (y_proba >= float(t)).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        cost = float(fp) * float(fp_cost) + float(fn) * float(fn_cost)

        if best is None or cost < best["total_cost"]:
            best = {
                "threshold": float(t),
                "tn": float(tn),
                "fp": float(fp),
                "fn": float(fn),
                "tp": float(tp),
                "total_cost": float(cost),
                "pr_auc": float(pr_auc),
            }

    return best
