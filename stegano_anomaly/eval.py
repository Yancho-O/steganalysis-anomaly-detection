from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


@dataclass(frozen=True)
class Metrics:
    roc_auc: float
    average_precision: float
    pr_curve: Tuple[np.ndarray, np.ndarray, np.ndarray]  # precision, recall, thresholds
    roc_curve: Tuple[np.ndarray, np.ndarray, np.ndarray]  # fpr, tpr, thresholds

def compute_metrics(y_true: np.ndarray, scores: np.ndarray) -> Metrics:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)

    roc_auc = float(roc_auc_score(y_true, scores))
    ap = float(average_precision_score(y_true, scores))

    prec, rec, pr_thr = precision_recall_curve(y_true, scores)
    fpr, tpr, roc_thr = roc_curve(y_true, scores)
    return Metrics(
        roc_auc=roc_auc,
        average_precision=ap,
        pr_curve=(prec, rec, pr_thr),
        roc_curve=(fpr, tpr, roc_thr),
    )
