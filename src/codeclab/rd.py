"""Rate-distortion helpers shared across experiments.

Deliberately thin: the per-experiment sweep loop lives in each
``experiments/exp_*/run.py``. This module only holds the small conversions that
would otherwise be re-derived (and occasionally mis-derived) in every script.
"""
from __future__ import annotations

import numpy as np

__all__ = ["rate_matched_codebook_size", "vq_gain_db", "water_filling_bits"]


def rate_matched_codebook_size(bits_per_dim: float, D: int) -> int:
    """Codebook size for a VQ that spends the same bits as ``bits_per_dim`` * D.

    A scalar quantizer at ``b`` bits on each of ``D`` dimensions spends ``b*D``
    bits per vector, i.e. ``2**(b*D)`` distinguishable reconstructions. A fair VQ
    gets a codebook of that size.
    """
    size = 2 ** (bits_per_dim * D)
    if size != int(size):
        raise ValueError(
            f"bits_per_dim*D = {bits_per_dim*D} is not an integer; "
            "codebook size is ambiguous"
        )
    return int(size)


def vq_gain_db(quality_vq_db: float, quality_sq_db: float) -> float:
    """VQ's advantage over SQ, in dB — the suite's headline quantity.

    Positive means VQ reconstructs the source better at the same rate.
    """
    return float(quality_vq_db - quality_sq_db)


def water_filling_bits(variances, total_bits: int):
    """Integer bit allocation across axes by greedy marginal gain.

    Hands each of ``total_bits`` bits to the axis whose quantization error is
    currently largest (proxy ``variance * 2**(-2b)``, i.e. Gaussian high-rate
    distortion). Reduces to an equal split when the variances match; loads the
    high-variance axes when they differ — the classical transform-coding
    allocation. Returns an int array summing to ``total_bits``.
    """
    v = np.asarray(variances, dtype=float)
    b = np.zeros(len(v), dtype=int)
    for _ in range(int(round(total_bits))):
        b[int(np.argmax(v * 4.0 ** (-b)))] += 1
    return b
