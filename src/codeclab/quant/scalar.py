"""Scalar quantizer: Lloyd-Max levels trained on samples."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["lloyd_max", "ScalarQuantizer", "quantize_per_axis"]


@dataclass
class ScalarQuantizer:
    """A trained 1-D quantizer; ``boundaries`` is ``len(levels) + 1`` long, -inf/+inf at the ends."""

    levels: np.ndarray
    boundaries: np.ndarray

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Sample -> cell index in ``[0, len(levels))``."""
        return np.clip(np.digitize(x, self.boundaries) - 1, 0, len(self.levels) - 1)

    def decode(self, idx: np.ndarray) -> np.ndarray:
        """Cell index -> reconstruction level."""
        return self.levels[idx]

    def quantize(self, x: np.ndarray) -> np.ndarray:
        """Sample -> reconstruction (encode then decode)."""
        return self.decode(self.encode(x))


def lloyd_max(
    x: np.ndarray, bits: int, *, n_iter: int = 100, tol: float = 1e-10
) -> ScalarQuantizer:
    """Train a ``2**bits``-level Lloyd-Max quantizer on ``x``, alternating midpoint
    boundaries and cell centroids. An empty cell keeps its previous level.
    """
    x = np.asarray(x, dtype=float).ravel()
    n_levels = 2 ** bits
    levels = np.linspace(x.min(), x.max(), n_levels)

    prev = None
    for _ in range(n_iter):
        boundaries = np.concatenate(
            [[-np.inf], (levels[1:] + levels[:-1]) / 2, [np.inf]]
        )
        cell = np.digitize(x, boundaries) - 1
        levels = np.array(
            [
                x[cell == k].mean() if np.any(cell == k) else levels[k]
                for k in range(n_levels)
            ]
        )
        if prev is not None and np.allclose(levels, prev, atol=tol):
            break
        prev = levels.copy()

    boundaries = np.concatenate([[-np.inf], (levels[1:] + levels[:-1]) / 2, [np.inf]])
    return ScalarQuantizer(levels=levels, boundaries=boundaries)


def quantize_per_axis(X: np.ndarray, bits, **kw) -> np.ndarray:
    """Independent Lloyd-Max per column of ``X`` (n, D). ``bits`` is an int or a
    per-axis sequence; a zero-bit axis is reconstructed at its mean.
    """
    X = np.asarray(X, dtype=float)
    D = X.shape[1]
    bits_per_axis = np.broadcast_to(bits, (D,)).astype(int)
    X_hat = np.empty_like(X)
    for j in range(D):
        b = int(bits_per_axis[j])
        if b <= 0:
            X_hat[:, j] = X[:, j].mean()
        else:
            q = lloyd_max(X[:, j], b, **kw)
            X_hat[:, j] = q.quantize(X[:, j])
    return X_hat
