"""Linear transforms applied before quantization.

PCA / KLT is the classical stand-in for a neural encoder: a fixed rotation onto
the principal axes that removes second-order (linear) correlation. EXP-03 uses it
to show that VQ's advantage over SQ is decorrelation a transform can do — but
only when paired with bit allocation (rotation alone is distortion-invariant at
equal rate).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["PCA", "pca"]


@dataclass
class PCA:
    """A fitted PCA/KLT rotation.

    ``components`` columns are eigenvectors of the sample covariance, ordered by
    decreasing eigenvalue; ``explained_variance`` are those eigenvalues.
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
    """Fit a PCA rotation to ``X`` (n, D) from its sample covariance.

    For the 2D equicorrelation Gaussian ``[[1, rho], [rho, 1]]`` the eigenvectors
    are exactly the +/-45 deg rotation and the eigenvalues are ``1 +/- rho`` —
    hand-checkable.
    """
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)
    cov = np.cov(X - mean, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    return PCA(mean=mean, components=evecs[:, order], explained_variance=evals[order])
