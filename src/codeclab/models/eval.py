"""Freeze-then-swap SQ vs VQ evaluation on a trained autoencoder's latent."""
from __future__ import annotations

import numpy as np
import torch

from ..diagnostics import latent_diagnostics
from ..metrics.toy import sqnr_db
from ..quant.scalar import quantize_per_axis
from ..quant.vector import lbg
from ..rd import rate_matched_codebook_size, vq_gain_db, water_filling_bits
from .autoencoder import AutoEncoder

__all__ = ["evaluate_sq_vs_vq"]


def evaluate_sq_vs_vq(ae: AutoEncoder, X: np.ndarray, bits_per_dim: int, *, lbg_eps: float = 0.01) -> dict:
    X = np.asarray(X, dtype=np.float32)
    D = X.shape[1]
    ae.eval()
    with torch.no_grad():
        Z = ae.encode(torch.from_numpy(X)).numpy()

    diag = latent_diagnostics(Z)

    total_bits = bits_per_dim * D
    K = rate_matched_codebook_size(bits_per_dim, D)
    bits = water_filling_bits(np.diag(diag["cov"]), total_bits)
    K_sq = int(round(2 ** bits.sum()))
    assert K_sq == K, f"K_sq={K_sq} != K_vq={K}"

    Z_hat_sq = quantize_per_axis(Z, bits)
    vq = lbg(Z, K, eps=lbg_eps)
    Z_hat_vq = vq.quantize(Z)

    with torch.no_grad():
        X_hat_sq = ae.decoder(torch.from_numpy(Z_hat_sq.astype(np.float32))).numpy()
        X_hat_vq = ae.decoder(torch.from_numpy(Z_hat_vq.astype(np.float32))).numpy()

    sqnr_sq_latent = sqnr_db(Z, Z_hat_sq)
    sqnr_vq_latent = sqnr_db(Z, Z_hat_vq)
    sqnr_sq_signal = sqnr_db(X, X_hat_sq)
    sqnr_vq_signal = sqnr_db(X, X_hat_vq)

    return {
        "sqnr_sq_latent_db": sqnr_sq_latent,
        "sqnr_vq_latent_db": sqnr_vq_latent,
        "gain_latent_db": vq_gain_db(sqnr_vq_latent, sqnr_sq_latent),
        "sqnr_sq_signal_db": sqnr_sq_signal,
        "sqnr_vq_signal_db": sqnr_vq_signal,
        "gain_signal_db": vq_gain_db(sqnr_vq_signal, sqnr_sq_signal),
        "bits_allocation": bits.tolist(),
        "K": K,
        "T": 1,
        "latent_cov": diag["cov"].tolist(),
        "off_diag_energy": diag["off_diag_energy"],
        "eigenvalues": diag["eigenvalues"].tolist(),
        "participation_ratio": diag["participation_ratio"],
    }
