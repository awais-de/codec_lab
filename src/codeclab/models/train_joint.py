"""Alternating encoder/quantizer training: a real SQ or VQ sits inside the loop
via a straight-through estimator, instead of the quantizer-blind noise proxy
``train_autoencoder`` uses. Lets the encoder adapt to its own quantizer."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..quant.scalar import lloyd_max
from ..quant.vector import lbg
from ..rd import water_filling_bits
from .autoencoder import AutoEncoder

__all__ = ["train_joint"]


def _fit_sq(Z: np.ndarray, total_bits: int):
    bits = water_filling_bits(Z.var(axis=0), total_bits)
    quantizers = [lloyd_max(Z[:, j], int(b)) if b > 0 else None for j, b in enumerate(bits)]
    return quantizers, bits


def _apply_sq(Z: np.ndarray, quantizers) -> np.ndarray:
    Z_hat = np.empty_like(Z)
    for j, q in enumerate(quantizers):
        Z_hat[:, j] = Z[:, j].mean() if q is None else q.quantize(Z[:, j])
    return Z_hat


def train_joint(
    X: np.ndarray,
    quantizer: str,
    capacity: int | list[int] | None,
    *,
    bits_per_dim: int = 2,
    start_bits_per_dim: int | None = None,
    activation=nn.Tanh,
    outer_rounds: int = 20,
    inner_epochs: int = 100,
    lr: float = 1e-2,
    seed: int = 0,
) -> AutoEncoder:
    """``start_bits_per_dim`` anneals the in-loop quantizer from fine (low
    distortion, low STE gradient bias) down to ``bits_per_dim`` over the outer
    rounds, so the encoder can find a global rearrangement (e.g. decorrelation)
    before the target rate's coarse cells dominate the gradient. Omit for a
    fixed rate throughout.
    """
    if quantizer not in ("sq", "vq"):
        raise ValueError(f"quantizer must be 'sq' or 'vq', got {quantizer!r}")
    if start_bits_per_dim is None:
        start_bits_per_dim = bits_per_dim
    torch.manual_seed(seed)
    dim = X.shape[1]

    ae = AutoEncoder(dim=dim, capacity=capacity, activation=activation)
    Xt = torch.from_numpy(np.asarray(X, dtype=np.float32))
    ae.set_target_var(Xt)
    opt = torch.optim.Adam(ae.parameters(), lr=lr)

    for round_idx in range(outer_rounds):
        frac = round_idx / max(outer_rounds - 1, 1)
        round_bits_per_dim = start_bits_per_dim + frac * (bits_per_dim - start_bits_per_dim)
        total_bits = max(1, round(round_bits_per_dim * dim))
        K = 2 ** total_bits

        with torch.no_grad():
            Z = ae.encode(Xt).numpy()
        if quantizer == "sq":
            sq_quantizers, _ = _fit_sq(Z, total_bits)
        else:
            vq = lbg(Z, K)

        for _ in range(inner_epochs):
            opt.zero_grad()
            z = ae.encode(Xt)
            z_np = z.detach().numpy()
            z_q_np = _apply_sq(z_np, sq_quantizers) if quantizer == "sq" else vq.quantize(z_np)
            z_q = z + (torch.from_numpy(z_q_np.astype(np.float32)) - z).detach()
            x_hat = ae.decoder(z_q)
            loss = torch.mean((x_hat - Xt) ** 2)
            loss.backward()
            opt.step()

    ae.eval()
    return ae
