from __future__ import annotations

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.mean(y_true == y_pred))


def confusion_matrix_binary(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return [[TP, FN],[FP, TN]] for labels {+1,-1} with +1 as positive."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == -1)))
    fp = int(np.sum((y_true == -1) & (y_pred == 1)))
    tn = int(np.sum((y_true == -1) & (y_pred == -1)))

    return np.asarray([[tp, fn], [fp, tn]], dtype=int)
