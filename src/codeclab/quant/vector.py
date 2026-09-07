"""Vector quantizer: LBG (generalised Lloyd) codebook design.

Canonical implementation for the experiment suite. The validated original lives
frozen in ``theory/06_vector_quantization.py``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["kmeans_refine", "lbg", "VectorQuantizer"]


def kmeans_refine(
    X: np.ndarray, codewords: np.ndarray, *, n_iter: int = 100, tol: float = 1e-11
) -> tuple[np.ndarray, list[float]]:
    """Lloyd iteration on a fixed-size codebook: assign -> centroid -> repeat.

    Returns the refined codewords and the per-iteration distortion history
    (per-sample MSE, i.e. summed squared error divided by dimension so it is
    comparable to a scalar quantizer's MSE).
    """
    X = np.asarray(X, dtype=float)
    D = X.shape[1]
    history: list[float] = []

    for _ in range(n_iter):
        dists = np.linalg.norm(X[:, None, :] - codewords[None, :, :], axis=2)
        assign = np.argmin(dists, axis=1)
        new_cw = np.array(
            [
                X[assign == k].mean(axis=0) if np.any(assign == k) else codewords[k]
                for k in range(len(codewords))
            ]
        )
        mse = np.mean(np.sum((X - new_cw[assign]) ** 2, axis=1)) / D
        history.append(float(mse))
        codewords = new_cw
        if len(history) > 1 and abs(history[-2] - history[-1]) < tol:
            break

    return codewords, history


def lbg(X: np.ndarray, n_codewords: int, *, eps: float = 1e-2, **kw) -> "VectorQuantizer":
    """Design a codebook of size ``n_codewords`` by additive splitting + refinement.

    Grows 1 -> 2 -> 4 -> ... , splitting every codeword as ``c +/- eps`` and
    re-running :func:`kmeans_refine` after each split. ``n_codewords`` need not be
    a power of two (the last split is truncated).
    """
    X = np.asarray(X, dtype=float)
    codewords = X.mean(axis=0, keepdims=True)
    while len(codewords) < n_codewords:
        codewords = np.concatenate([codewords + eps, codewords - eps], axis=0)
        if len(codewords) > n_codewords:
            codewords = codewords[:n_codewords]
        codewords, _ = kmeans_refine(X, codewords, **kw)
    return VectorQuantizer(codewords=codewords)


@dataclass
class VectorQuantizer:
    """A trained codebook."""

    codewords: np.ndarray

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Rows of ``X`` -> nearest-codeword indices."""
        dists = np.linalg.norm(X[:, None, :] - self.codewords[None, :, :], axis=2)
        return np.argmin(dists, axis=1)

    def decode(self, idx: np.ndarray) -> np.ndarray:
        return self.codewords[idx]

    def quantize(self, X: np.ndarray) -> np.ndarray:
        return self.decode(self.encode(X))
