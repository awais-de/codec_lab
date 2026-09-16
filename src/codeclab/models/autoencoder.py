"""Toy autoencoder: linear or multi-hidden-layer, with variance-conserving latents."""
from __future__ import annotations

import torch
from torch import nn

__all__ = ["AutoEncoder"]


def _coder(dim: int, capacity: int | list[int] | None, activation) -> nn.Module:
    if capacity is None:
        return nn.Linear(dim, dim)
    hidden = [capacity] if isinstance(capacity, int) else list(capacity)
    sizes = [dim, *hidden, dim]
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class AutoEncoder(nn.Module):
    def __init__(self, dim: int = 2, capacity: int | list[int] | None = None, activation=nn.Tanh):
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
