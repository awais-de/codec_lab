"""Hand-built ring encoders for the ring-claim audit: a polar seam (discontinuous) and a
seamless fold along a Hamiltonian cycle of the 4x4 grid. Each comes with its inverse."""
from __future__ import annotations

import numpy as np

__all__ = ["GRID_CYCLE", "matched_perp_scale", "polar_seam", "unpolar", "fold_ring", "unfold_ring"]

GRID_CYCLE = np.array(
    [(0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (1, 2), (1, 1), (2, 1),
     (2, 2), (2, 3), (3, 3), (3, 2), (3, 1), (3, 0), (2, 0), (1, 0)],
    dtype=float,
) - 1.5


def polar_seam(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    theta = np.arctan2(X[:, 1], X[:, 0]) % (2 * np.pi)
    r = np.hypot(X[:, 0], X[:, 1])
    return np.stack([theta, r], axis=1)


def unpolar(Z: np.ndarray) -> np.ndarray:
    Z = np.asarray(Z, dtype=float)
    theta, r = Z[:, 0], Z[:, 1]
    return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)


def matched_perp_scale(radius: float = 1.0) -> float:
    """Perpendicular scale that keeps the ring's noise-to-spacing ratio on the fold."""
    return len(GRID_CYCLE) / (2 * np.pi * radius)


def fold_ring(X: np.ndarray, *, radius: float = 1.0, perp_scale: float | None = None) -> np.ndarray:
    """Angle -> arc length along GRID_CYCLE (unit spacing); radial deviation -> perpendicular
    offset times ``perp_scale`` (default: matched to the ring's noise-to-spacing ratio)."""
    X = np.asarray(X, dtype=float)
    n = len(GRID_CYCLE)
    scale = matched_perp_scale(radius) if perp_scale is None else perp_scale
    theta = np.arctan2(X[:, 1], X[:, 0]) % (2 * np.pi)
    r = np.hypot(X[:, 0], X[:, 1])
    s = theta / (2 * np.pi) * n
    i = np.floor(s).astype(int) % n
    f = s - np.floor(s)
    a = GRID_CYCLE[i]
    tang = GRID_CYCLE[(i + 1) % n] - a
    normal = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
    return a + f[:, None] * tang + ((r - radius) * scale)[:, None] * normal


def unfold_ring(Z: np.ndarray, *, radius: float = 1.0, perp_scale: float | None = None) -> np.ndarray:
    """Exact inverse of fold_ring on the path; nearest-segment projection off it."""
    Z = np.asarray(Z, dtype=float)
    n = len(GRID_CYCLE)
    scale = matched_perp_scale(radius) if perp_scale is None else perp_scale
    A = GRID_CYCLE
    T = np.roll(GRID_CYCLE, -1, axis=0) - A
    d = Z[:, None, :] - A[None, :, :]
    f = np.clip(np.einsum("mkd,kd->mk", d, T), 0.0, 1.0)
    P = A[None] + f[..., None] * T[None]
    k = np.argmin(np.linalg.norm(Z[:, None, :] - P, axis=2), axis=1)
    rows = np.arange(len(Z))
    fk, Pk = f[rows, k], P[rows, k]
    normal = np.stack([-T[k, 1], T[k, 0]], axis=1)
    off = np.einsum("md,md->m", Z - Pk, normal)
    theta = (k + fk) / n * 2 * np.pi
    r = radius + off / scale
    return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
