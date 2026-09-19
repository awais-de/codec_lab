"""Distortion metrics for ``(n, D)`` block sources. MSE is per-sample (over ``n * D``),
so SQ and VQ at equal bits/dim compare on the same footing.
"""
from __future__ import annotations

import numpy as np

__all__ = ["mse", "snr_db", "sqnr_db"]


def _as2d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return a[:, None] if a.ndim == 1 else a


def mse(x: np.ndarray, x_hat: np.ndarray) -> float:
    """Per-sample mean squared error."""
    x, x_hat = _as2d(x), _as2d(x_hat)
    return float(np.mean(np.sum((x - x_hat) ** 2, axis=1)) / x.shape[1])


def snr_db(x: np.ndarray, x_hat: np.ndarray) -> float:
    """Signal power over error power, in dB."""
    x, x_hat = _as2d(x), _as2d(x_hat)
    sig = np.mean(np.sum(x ** 2, axis=1)) / x.shape[1]
    err = mse(x, x_hat)
    return float(10.0 * np.log10(sig / err))


def sqnr_db(x: np.ndarray, x_hat: np.ndarray) -> float:
    """Signal-to-quantization-noise ratio, in dB (variance-based)."""
    x, x_hat = _as2d(x), _as2d(x_hat)
    sig = float(np.var(x))
    noise = float(np.var(x - x_hat))
    return float(10.0 * np.log10(sig / noise))
