"""Scalar quantizers: mid-rise uniform, and optimal Lloyd-Max (trained on samples).

Canonical implementations for the experiment suite. Written fresh here; the
validated originals live frozen in ``theory/04_uniform_sq.py`` and
``theory/05_lloyd_max.py``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["uniform_quantize", "lloyd_max", "ScalarQuantizer", "quantize_per_axis"]


def uniform_quantize(x: np.ndarray, lo: float, hi: float, bits: int) -> np.ndarray:
    """Mid-rise uniform scalar quantizer on ``[lo, hi]`` with ``2**bits`` levels.

    Values outside ``[lo, hi]`` are clipped. Returns the reconstruction ``x_hat``.
    """
    n_levels = 2 ** bits
    delta = (hi - lo) / n_levels
    x_clipped = np.clip(x, lo, hi)
    cell = np.floor((x_clipped - lo) / delta).astype(int)
    cell = np.clip(cell, 0, n_levels - 1)
    return lo + (cell + 0.5) * delta


@dataclass
class ScalarQuantizer:
    """A trained 1-D quantizer: ``levels`` with the ``boundaries`` between them.

    ``boundaries`` has length ``len(levels) + 1`` with -inf / +inf at the ends.
    """

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
    """Train an optimal ``2**bits``-level Lloyd-Max quantizer on samples ``x``.

    Alternates the two optimality conditions until the levels stop moving:
      - boundary update: interior boundaries = midpoints of adjacent levels
      - level update:    each level = centroid (mean) of the samples in its cell

    Empty-cell guard: a cell that catches no samples keeps its previous level.
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


def quantize_per_axis(X: np.ndarray, bits: int, **kw) -> np.ndarray:
    """Fit an independent Lloyd-Max quantizer to each column of ``X`` (n, D).

    This is the fair scalar baseline the suite compares VQ against: ``bits`` per
    dimension, no cross-dimension structure exploited. Returns ``X_hat``.
    """
    X = np.asarray(X, dtype=float)
    X_hat = np.empty_like(X)
    for j in range(X.shape[1]):
        q = lloyd_max(X[:, j], bits, **kw)
        X_hat[:, j] = q.quantize(X[:, j])
    return X_hat
