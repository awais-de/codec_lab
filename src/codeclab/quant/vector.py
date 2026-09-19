"""Vector quantizer: LBG (generalised Lloyd) codebook design."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["kmeans_refine", "lbg", "VectorQuantizer"]


def kmeans_refine(
    X: np.ndarray, codewords: np.ndarray, *, n_iter: int = 100, tol: float = 1e-11
) -> tuple[np.ndarray, list[float]]:
    """Lloyd iteration on a fixed-size codebook. Returns the codewords and the
    per-iteration distortion history as per-sample MSE (divided by ``D``).
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
    """Codebook of size ``n_codewords``, grown by splitting each codeword as
    ``c +/- eps`` and refining after each split. Need not be a power of two.
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
