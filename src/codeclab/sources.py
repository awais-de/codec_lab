"""Synthetic sources: seeded generators returning zero-mean ``(n, D)`` float64 arrays."""
from __future__ import annotations

import numpy as np

__all__ = [
    "gaussian_2d",
    "uniform_2d",
    "correlated_gaussian",
    "ring_2d",
    "arc_2d",
    "arc_2d_covariance",
]


def gaussian_2d(rho: float, n: int, *, seed: int = 0) -> np.ndarray:
    """2D zero-mean Gaussian, unit variances, covariance ``[[1, rho], [rho, 1]]``."""
    if not -1.0 < rho < 1.0:
        raise ValueError(f"rho must be in (-1, 1), got {rho}")
    cov = np.array([[1.0, rho], [rho, 1.0]])
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(np.zeros(2), cov, size=n)


def uniform_2d(n: int, *, seed: int = 0) -> np.ndarray:
    """2D i.i.d. uniform, each axis ``U(-sqrt3, sqrt3)``: zero-mean, unit-variance,
    and zero shape gain (a flat pdf makes uniform point density optimal).
    """
    a = np.sqrt(3.0)
    rng = np.random.default_rng(seed)
    return rng.uniform(-a, a, size=(n, 2))


def correlated_gaussian(
    rho: float, D: int, n: int, *, seed: int = 0, kind: str = "ar1"
) -> np.ndarray:
    """D-dim zero-mean Gaussian. ``kind="ar1"`` gives ``Sigma[i, j] = rho ** |i - j|``,
    ``"equi"`` gives ``rho`` off-diagonal and 1 on the diagonal.
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


def ring_2d(n: int, *, radius: float = 1.0, radial_noise: float = 0.1, seed: int = 0) -> np.ndarray:
    """2D points scattered near a circle: zero-mean, isotropic covariance
    ``~ 0.5*(radius**2 + radial_noise**2) * I``, exact as ``radial_noise -> 0``.
    """
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 2 * np.pi, size=n)
    r = radius + rng.normal(0.0, radial_noise, size=n)
    return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)


def arc_2d(n: int, angle_deg: float, *, radius: float = 1.0, radial_noise: float = 0.1,
           seed: int = 0) -> np.ndarray:
    """Points scattered near an arc of angular extent ``angle_deg``, centred on its
    analytic mean. ``angle_deg=360`` is :func:`ring_2d` (same draw order, zero mean).
    """
    alpha = np.deg2rad(angle_deg)
    if not 0 < alpha <= 2 * np.pi:
        raise ValueError(f"angle_deg must be in (0, 360], got {angle_deg}")
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, alpha, size=n)
    r = radius + rng.normal(0.0, radial_noise, size=n)
    mean = radius * np.array([np.sin(alpha) / alpha, (1 - np.cos(alpha)) / alpha])
    return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1) - mean


def arc_2d_covariance(angle_deg: float, *, radius: float = 1.0,
                      radial_noise: float = 0.1) -> np.ndarray:
    """Closed-form covariance of :func:`arc_2d`; equals ``0.5*(radius**2+radial_noise**2)*I``
    at ``angle_deg=360``.
    """
    a = np.deg2rad(angle_deg)
    m2 = radius ** 2 + radial_noise ** 2
    ex, ey = radius * np.sin(a) / a, radius * (1 - np.cos(a)) / a
    vx = m2 * (0.5 + np.sin(2 * a) / (4 * a)) - ex ** 2
    vy = m2 * (0.5 - np.sin(2 * a) / (4 * a)) - ey ** 2
    cxy = m2 * (1 - np.cos(2 * a)) / (4 * a) - ex * ey
    return np.array([[vx, cxy], [cxy, vy]])
