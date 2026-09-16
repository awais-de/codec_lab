"""Trains an autoencoder with reconstruction MSE plus an additive-noise rate proxy."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .autoencoder import AutoEncoder

__all__ = ["train_autoencoder"]


def train_autoencoder(
    X: np.ndarray,
    capacity: int | list[int] | None = None,
    *,
    activation=nn.Tanh,
    rate_noise: float = 1.0,
    epochs: int = 2000,
    lr: float = 1e-2,
    seed: int = 0,
) -> AutoEncoder:
    torch.manual_seed(seed)
    dim = X.shape[1]
    ae = AutoEncoder(dim=dim, capacity=capacity, activation=activation)
    Xt = torch.from_numpy(np.asarray(X, dtype=np.float32))
    ae.set_target_var(Xt)

    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        z = ae.encode(Xt)
        noise = (torch.rand_like(z) - 0.5) * rate_noise
        x_hat = ae.decoder(z + noise)
        loss = torch.mean((x_hat - Xt) ** 2)
        loss.backward()
        opt.step()

    ae.eval()
    return ae
