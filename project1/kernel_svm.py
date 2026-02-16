from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def kernel_linear(x: np.ndarray, z: np.ndarray) -> float:
    return float(np.dot(x, z))


def kernel_poly(x: np.ndarray, z: np.ndarray, gamma: float = 1.0, coef0: float = 1.0, degree: int = 3) -> float:
    return float((gamma * np.dot(x, z) + coef0) ** degree)


def kernel_rbf(x: np.ndarray, z: np.ndarray, gamma: float = 0.1) -> float:
    diff = x - z
    return float(np.exp(-gamma * np.dot(diff, diff)))


@dataclass
class KernelSVMModel:
    alphas: np.ndarray
    b: float
    X_train: np.ndarray
    y_train: np.ndarray
    kernel_name: str
    kernel_params: dict

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        y = self.y_train.astype(np.float32)
        a = self.alphas.astype(np.float32)

        # Compute f(x) = sum_i alpha_i y_i K(x_i, x) + b
        out = np.zeros(X.shape[0], dtype=np.float32)
        for j in range(X.shape[0]):
            s = 0.0
            xj = X[j]
            for i in range(self.X_train.shape[0]):
                if a[i] == 0.0:
                    continue
                s += a[i] * y[i] * self._K(self.X_train[i], xj)
            out[j] = s + self.b
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        return np.where(scores >= 0.0, 1, -1)

    def _K(self, x: np.ndarray, z: np.ndarray) -> float:
        if self.kernel_name == "linear":
            return kernel_linear(x, z)
        if self.kernel_name == "poly":
            return kernel_poly(x, z, **self.kernel_params)
        if self.kernel_name == "rbf":
            return kernel_rbf(x, z, **self.kernel_params)
        raise ValueError(f"Unknown kernel: {self.kernel_name}")


class KernelSoftMarginSVM_SMO:
    """Binary soft-margin kernel SVM trained with a simple SMO.

    Optimizes the standard kernel SVM dual:
        max_a  sum_i a_i - 1/2 sum_{i,j} a_i a_j y_i y_j K(x_i,x_j)
        s.t.   0 <= a_i <= C, sum_i a_i y_i = 0

    This is intentionally minimal and readable (project wants from-scratch).
    """

    def __init__(
        self,
        C: float = 1.0,
        kernel: str = "rbf",
        kernel_params: dict | None = None,
        tol: float = 1e-3,
        max_passes: int = 10,
        seed: int = 0,
    ):
        self.C = float(C)
        self.kernel = str(kernel)
        self.kernel_params = dict(kernel_params or {})
        self.tol = float(tol)
        self.max_passes = int(max_passes)
        self.seed = int(seed)

    def fit(self, X: np.ndarray, y: np.ndarray) -> KernelSVMModel:
        # Use float64 for better numerical stability in SMO updates.
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        n = X.shape[0]

        rng = np.random.default_rng(self.seed)

        alphas = np.zeros(n, dtype=np.float64)
        b = 0.0

        # Precompute kernel matrix for speed (n is small here ~100).
        K = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                K[i, j] = self._K(X[i], X[j])

        # Cache f values to avoid recomputing inner sums repeatedly.
        # f_vec[i] = sum_k alpha_k y_k K(x_k, x_i) + b
        f_vec = (alphas * y) @ K + b

        def update_f_vec(i: int, j: int, ai_old: float, aj_old: float) -> None:
            # Incremental update using delta alphas.
            dai = alphas[i] - ai_old
            daj = alphas[j] - aj_old
            if dai == 0.0 and daj == 0.0:
                return
            # f_vec[t] += dai*y_i*K(i,t) + daj*y_j*K(j,t)
            f_vec[:] = f_vec + dai * y[i] * K[i, :] + daj * y[j] * K[j, :]

        def E(i: int) -> float:
            return float(f_vec[i] - y[i])

        def select_j(i: int) -> int:
            # Heuristic: choose j that maximizes |E_i - E_j|.
            Ei = E(i)
            non_i = np.arange(n)
            # Prefer non-bound alphas for more movement.
            non_bound = np.where((alphas > 1e-8) & (alphas < self.C - 1e-8))[0]
            candidates = non_bound if non_bound.size > 1 else non_i
            j = int(candidates[np.argmax(np.abs((f_vec[candidates] - y[candidates]) - Ei))])
            if j == i:
                # fallback random different index
                j = int(rng.integers(0, n - 1))
                if j >= i:
                    j += 1
            return j

        passes = 0
        while passes < self.max_passes:
            num_changed = 0
            for i in range(n):
                Ei = E(i)

                if (y[i] * Ei < -self.tol and alphas[i] < self.C) or (y[i] * Ei > self.tol and alphas[i] > 0):
                    j = select_j(i)
                    Ej = E(j)

                    ai_old = float(alphas[i])
                    aj_old = float(alphas[j])

                    if y[i] != y[j]:
                        L = max(0.0, aj_old - ai_old)
                        H = min(self.C, self.C + aj_old - ai_old)
                    else:
                        L = max(0.0, ai_old + aj_old - self.C)
                        H = min(self.C, ai_old + aj_old)

                    if abs(H - L) < 1e-12:
                        continue

                    eta = 2.0 * float(K[i, j]) - float(K[i, i]) - float(K[j, j])
                    if eta >= 0:
                        continue

                    # Update alpha_j and clip
                    aj_new = aj_old - (y[j] * (Ei - Ej)) / eta
                    aj_new = float(np.clip(aj_new, L, H))
                    if abs(aj_new - aj_old) < 1e-5:
                        continue

                    # Update alpha_i
                    ai_new = ai_old + y[i] * y[j] * (aj_old - aj_new)

                    alphas[i] = ai_new
                    alphas[j] = aj_new

                    update_f_vec(i, j, ai_old, aj_old)

                    # Update b
                    b1 = b - Ei - y[i] * (ai_new - ai_old) * float(K[i, i]) - y[j] * (aj_new - aj_old) * float(K[i, j])
                    b2 = b - Ej - y[i] * (ai_new - ai_old) * float(K[i, j]) - y[j] * (aj_new - aj_old) * float(K[j, j])

                    if 0 < ai_new < self.C:
                        b = b1
                    elif 0 < aj_new < self.C:
                        b = b2
                    else:
                        b = 0.5 * (b1 + b2)

                    # Keep f_vec consistent after b changes.
                    f_vec[:] = (alphas * y) @ K + b

                    num_changed += 1

            passes = passes + 1 if num_changed == 0 else 0

        return KernelSVMModel(
            alphas=alphas.astype(np.float64),
            b=float(b),
            X_train=X.astype(np.float64),
            y_train=y.astype(np.float64),
            kernel_name=self.kernel,
            kernel_params=self.kernel_params,
        )

    def _K(self, x: np.ndarray, z: np.ndarray) -> float:
        if self.kernel == "linear":
            return kernel_linear(x, z)
        if self.kernel == "poly":
            return kernel_poly(x, z, **self.kernel_params)
        if self.kernel == "rbf":
            return kernel_rbf(x, z, **self.kernel_params)
        raise ValueError(f"Unknown kernel: {self.kernel}")
