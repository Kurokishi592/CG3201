from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LinearSVMResult:
    w: np.ndarray
    b: float


class LinearSoftMarginSVM:
    """From-scratch soft-margin linear SVM via SGD on hinge loss + L2 regularization.

    Objective (standard soft-margin SVM):
        min_{w,b} 1/2 ||w||^2 + C * sum_i max(0, 1 - y_i (w^T x_i + b))

    We do stochastic subgradient descent.
    """

    def __init__(self, C: float = 1.0, learning_rate: float = 1e-3, epochs: int = 50, seed: int = 0):
        self.C = float(C)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.seed = int(seed)
        self.w: np.ndarray | None = None
        self.b: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearSoftMarginSVM":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        n, d = X.shape

        rng = np.random.default_rng(self.seed)
        self.w = np.zeros(d, dtype=np.float32)
        self.b = 0.0

        for _ in range(self.epochs):
            idx = rng.permutation(n)
            for i in idx:
                xi = X[i]
                yi = y[i]

                margin = yi * (float(np.dot(self.w, xi)) + self.b)

                # Subgradient for 1/2||w||^2 is w.
                if margin >= 1.0:
                    grad_w = self.w
                    grad_b = 0.0
                else:
                    grad_w = self.w - self.C * yi * xi
                    grad_b = -self.C * yi

                self.w = self.w - self.learning_rate * grad_w
                self.b = self.b - self.learning_rate * grad_b

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("Model not fitted")
        X = np.asarray(X, dtype=np.float32)
        return X @ self.w + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        return np.where(scores >= 0.0, 1, -1)
