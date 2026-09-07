"""Toy-stage distortion metrics for block sources of shape ``(n, D)``.

Conventions
-----------
- MSE is *per-sample*: total squared error divided by ``n * D``, so a scalar
  quantizer at ``b`` bits/dim and a vector quantizer at ``b`` bits/dim are
  compared on the same footing.
- SNR and SQNR are in dB. For a zero-mean source they coincide; both are kept
  because the suite reports "SQNR" for quantization specifically.
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
