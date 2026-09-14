"""Toy autoencoder: linear or 1-hidden-layer, with variance-conserving latents."""
from __future__ import annotations

import torch
from torch import nn

__all__ = ["AutoEncoder"]


def _coder(dim: int, capacity: int | None, activation) -> nn.Module:
    if capacity is None or capacity <= 0:
        return nn.Linear(dim, dim)
    return nn.Sequential(nn.Linear(dim, capacity), activation(), nn.Linear(capacity, dim))


class AutoEncoder(nn.Module):
    def __init__(self, dim: int = 2, capacity: int | None = None, activation=nn.Tanh):
        super().__init__()
        self.dim = dim
        self.encoder = _coder(dim, capacity, activation)
        self.decoder = _coder(dim, capacity, activation)
        self.register_buffer("target_var", torch.tensor(float(dim)))

    def set_target_var(self, x: torch.Tensor) -> None:
        with torch.no_grad():
            self.target_var.fill_(x.var(dim=0, unbiased=False).sum().item())

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        z_var = z.var(dim=0, unbiased=False).sum().clamp_min(1e-8)
        return z * torch.sqrt(self.target_var / z_var)

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        return z, self.decoder(z)
