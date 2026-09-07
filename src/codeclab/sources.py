"""Synthetic signal generators — the "sources" the quantizers are pointed at.

A *source* here is an information-theory source: a random process with known
statistics that emits the data to be compressed. Nothing is downloaded or read
from disk; every function returns a fresh NumPy array from a seeded RNG.

Every generator returns an array of shape ``(n, D)`` (float64), zero-mean, so
that downstream code can treat all sources uniformly.

Real speech loading (LibriSpeech) lands in a separate ``speech`` module at
Cluster 4 (EXP-13+); it does not belong here.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "gaussian_2d",
    "correlated_gaussian",
    "colored_noise",
    "laplacian",
]


def gaussian_2d(rho: float, n: int, *, seed: int = 0) -> np.ndarray:
    """2D zero-mean Gaussian with unit variances and correlation ``rho``.

    Covariance ``[[1, rho], [rho, 1]]``. Used by EXP-01 and EXP-09.
    """
    if not -1.0 < rho < 1.0:
        raise ValueError(f"rho must be in (-1, 1), got {rho}")
    cov = np.array([[1.0, rho], [rho, 1.0]])
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(np.zeros(2), cov, size=n)


def correlated_gaussian(
    rho: float, D: int, n: int, *, seed: int = 0, kind: str = "ar1"
) -> np.ndarray:
    """D-dimensional zero-mean Gaussian with a controllable correlation structure.

    Parameters
    ----------
    rho : float
        Correlation parameter in (-1, 1).
    D : int
        Dimensionality.
    kind : {"ar1", "equi"}
        ``"ar1"``  -> Toeplitz covariance ``Sigma[i, j] = rho ** |i - j|``
                      (AR(1)-style: nearby dimensions correlate more).
        ``"equi"`` -> equicorrelation ``Sigma[i, j] = rho`` for ``i != j``, 1 on
                      the diagonal (every pair equally correlated).

    Used by EXP-02, EXP-03..EXP-08 (with D=2 those reduce to the 2D case).
    """
    if not -1.0 < rho < 1.0:
        raise ValueError(f"rho must be in (-1, 1), got {rho}")
    if D < 1:
        raise ValueError(f"D must be >= 1, got {D}")

    idx = np.arange(D)
    if kind == "ar1":
        cov = rho ** np.abs(idx[:, None] - idx[None, :])
    elif kind == "equi":
        cov = np.full((D, D), rho)
        np.fill_diagonal(cov, 1.0)
    else:
        raise ValueError(f"kind must be 'ar1' or 'equi', got {kind!r}")

    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(np.zeros(D), cov, size=n)


def colored_noise(slope: float, n: int, *, D: int = 1, seed: int = 0) -> np.ndarray:
    """Power-law ("colored") noise with power spectral density ~ f**(-slope).

    slope = 0 -> white, 1 -> pink, 2 -> brown/red. Each of the ``D`` columns is
    an independent realisation, standardised to zero mean and unit variance.
    Used by EXP-11. Requires the ``colorednoise`` package.
    """
    import colorednoise as cn

    rng = np.random.default_rng(seed)
    cols = []
    for _ in range(D):
        x = cn.powerlaw_psd_gaussian(slope, n, random_state=rng)
        cols.append((x - x.mean()) / x.std())
    return np.stack(cols, axis=1)


def laplacian(scale: float, D: int, n: int, *, seed: int = 0) -> np.ndarray:
    """Zero-mean i.i.d. Laplacian source, shape ``(n, D)``.

    Heavier tails than a Gaussian of the same variance (variance = 2 * scale**2).
    Used by EXP-11 as a non-Gaussian contrast.
    """
    rng = np.random.default_rng(seed)
    return rng.laplace(loc=0.0, scale=scale, size=(n, D))
