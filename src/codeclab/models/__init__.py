"""Toy neural codec: autoencoder + training + freeze-then-swap SQ/VQ evaluation."""
from .autoencoder import AutoEncoder
from .eval import evaluate_own_quantizer, evaluate_sq_vs_vq
from .fidelity import reconstruction_fidelity
from .train import train_autoencoder
from .train_joint import train_joint
from .warmstart import continue_training, fit_to_map, marginal_modes

__all__ = [
    "AutoEncoder",
    "train_autoencoder",
    "train_joint",
    "evaluate_sq_vs_vq",
    "evaluate_own_quantizer",
    "reconstruction_fidelity",
    "fit_to_map",
    "continue_training",
    "marginal_modes",
]
