"""Toy neural codec: autoencoder + training + freeze-then-swap SQ/VQ evaluation."""
from .autoencoder import AutoEncoder
from .eval import evaluate_sq_vs_vq
from .train import train_autoencoder

__all__ = ["AutoEncoder", "train_autoencoder", "evaluate_sq_vs_vq"]
