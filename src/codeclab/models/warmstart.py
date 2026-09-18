"""Warm-start tooling for the ring-claim audit: fit an autoencoder to a hand-built latent
map, then continue training under the main ladder's blind-proxy objective while logging."""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from .autoencoder import AutoEncoder

__all__ = ["fit_to_map", "continue_training", "marginal_modes"]


def fit_to_map(
    ae: AutoEncoder,
    X: np.ndarray,
    Z_target: np.ndarray,
    *,
    epochs: int = 3000,
    lr: float = 1e-2,
    seed: int = 0,
) -> dict:
    """Supervised fit: encoder -> Z_target (rescaled to the AE's conserved variance),
    decoder -> X from that latent."""
    torch.manual_seed(seed)
    Xt = torch.from_numpy(np.asarray(X, dtype=np.float32))
    Zt = torch.from_numpy(np.asarray(Z_target, dtype=np.float32))
    ae.set_target_var(Xt)
    Zt = Zt * torch.sqrt(ae.target_var / Zt.var(dim=0, unbiased=False).sum())
    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss_enc = torch.mean((ae.encode(Xt) - Zt) ** 2)
        loss_dec = torch.mean((ae.decoder(Zt) - Xt) ** 2)
        (loss_enc + loss_dec).backward()
        opt.step()
    ae.eval()
    return {"fit_mse_encoder": loss_enc.item(), "fit_mse_decoder": loss_dec.item()}


def continue_training(
    ae: AutoEncoder,
    X: np.ndarray,
    *,
    rate_noise: float,
    epochs: int,
    lr: float = 1e-2,
    log_every: int = 100,
    seed: int = 0,
    callback: Callable[[int, AutoEncoder], None] | None = None,
) -> None:
    """The exact objective of ``train_autoencoder`` (reconstruction MSE through additive
    uniform noise on the conserved-variance latent), from the AE's current weights."""
    torch.manual_seed(seed)
    Xt = torch.from_numpy(np.asarray(X, dtype=np.float32))
    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    if callback is not None:
        callback(0, ae)
    for epoch in range(1, epochs + 1):
        opt.zero_grad()
        z = ae.encode(Xt)
        noise = (torch.rand_like(z) - 0.5) * rate_noise
        x_hat = ae.decoder(z + noise)
        loss = torch.mean((x_hat - Xt) ** 2)
        loss.backward()
        opt.step()
        if callback is not None and epoch % log_every == 0:
            callback(epoch, ae)
    ae.eval()


def marginal_modes(Z: np.ndarray, *, bins: int = 32, smooth: int = 5, min_rel_height: float = 0.30) -> list[int]:
    """Number of local maxima in each latent marginal's histogram (a fold has 4 per axis,
    a ring or ellipse 2, a blob 1). Defaults calibrated on those four references."""
    Z = np.asarray(Z, dtype=float)
    out = []
    for j in range(Z.shape[1]):
        h, _ = np.histogram(Z[:, j], bins=bins)
        h = np.convolve(h, np.ones(smooth) / smooth, mode="same")
        peaks = [i for i in range(1, len(h) - 1)
                 if h[i] > h[i - 1] and h[i] >= h[i + 1] and h[i] >= min_rel_height * h.max()]
        out.append(len(peaks))
    return out
