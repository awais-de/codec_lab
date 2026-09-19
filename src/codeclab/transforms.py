"""PCA / KLT: the fixed decorrelating rotation applied before quantization."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["PCA", "pca"]


@dataclass
class PCA:
    """A fitted rotation; ``components`` columns are covariance eigenvectors, ordered
    by decreasing ``explained_variance``.
    """

    mean: np.ndarray
    components: np.ndarray          # (D, D)
    explained_variance: np.ndarray  # (D,), decreasing

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Original coordinates -> principal-axis coordinates (decorrelated)."""
        return (np.asarray(X, dtype=float) - self.mean) @ self.components

    def inverse(self, Y: np.ndarray) -> np.ndarray:
        """Principal-axis coordinates -> original coordinates."""
        return np.asarray(Y, dtype=float) @ self.components.T + self.mean


def pca(X: np.ndarray) -> PCA:
    """Fit a PCA rotation to ``X`` (n, D) from its sample covariance."""
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)
    cov = np.cov(X - mean, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    return PCA(mean=mean, components=evecs[:, order], explained_variance=evals[order])
