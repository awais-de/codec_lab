"""Latent covariance diagnostics: off-diagonal energy, participation ratio."""
from __future__ import annotations

import numpy as np

__all__ = ["latent_diagnostics"]


def latent_diagnostics(Z: np.ndarray) -> dict:
    Z = np.asarray(Z, dtype=float)
    cov = np.cov(Z, rowvar=False)
    D = cov.shape[0]
    off_diag = cov[np.triu_indices(D, k=1)]
    off_diag_energy = float(np.sum(off_diag ** 2))
    evals = np.clip(np.linalg.eigvalsh(cov), 1e-12, None)
    participation_ratio = float(evals.sum() ** 2 / np.sum(evals ** 2))
    return {
        "cov": cov,
        "off_diag_energy": off_diag_energy,
        "eigenvalues": evals,
        "participation_ratio": participation_ratio,
    }
