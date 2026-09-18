"""Encoder fidelity diagnostics: un-quantized reconstruction SQNR, encoder Jacobian
determinant along the support, and a far-pair collapse ratio."""
from __future__ import annotations

import numpy as np
import torch

from ..metrics.toy import sqnr_db
from .autoencoder import AutoEncoder

__all__ = ["reconstruction_fidelity"]


def reconstruction_fidelity(
    ae: AutoEncoder,
    X: np.ndarray,
    *,
    n_jacobian: int = 2000,
    pair_min_angle_deg: float = 30.0,
    n_pairs: int = 20000,
    seed: int = 0,
) -> dict:
    X = np.asarray(X, dtype=np.float32)
    D = X.shape[1]
    ae.eval()
    Xt = torch.from_numpy(X)
    with torch.no_grad():
        Z = ae.encode(Xt)
        X_rec = ae.decoder(Z).numpy()
        z_raw = ae.encoder(Xt)
        scale = torch.sqrt(ae.target_var / z_raw.var(dim=0, unbiased=False).sum().clamp_min(1e-8)).item()
    Z = Z.numpy()

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n_jacobian, len(X)), replace=False)
    dets = np.empty(len(idx))
    for n, i in enumerate(idx):
        J = torch.autograd.functional.jacobian(ae.encoder, Xt[i])
        dets[n] = torch.det(J).item() * scale ** D
    sign = np.sign(dets)
    majority = 1.0 if (sign > 0).mean() >= 0.5 else -1.0
    angles = np.arctan2(X[idx, 1], X[idx, 0])

    i = rng.integers(0, len(X), size=n_pairs)
    j = rng.integers(0, len(X), size=n_pairs)
    theta = np.arctan2(X[:, 1], X[:, 0])
    dtheta = np.abs((theta[i] - theta[j] + np.pi) % (2 * np.pi) - np.pi)
    keep = dtheta > np.deg2rad(pair_min_angle_deg)
    i, j = i[keep], j[keep]
    ratio = np.linalg.norm(Z[i] - Z[j], axis=1) / np.linalg.norm(X[i] - X[j], axis=1)

    return {
        "recon_sqnr_db": sqnr_db(X, X_rec),
        "det_min_abs": float(np.abs(dets).min()),
        "det_median_abs": float(np.median(np.abs(dets))),
        "det_sign_flip_frac": float((sign != majority).mean()),
        "pair_ratio_min": float(ratio.min()),
        "pair_ratio_p01": float(np.percentile(ratio, 1)),
        "det_samples": dets.tolist(),
        "det_angles": angles.tolist(),
    }
