"""Rate-distortion helpers: rate matching, VQ gain in dB, integer bit allocation."""
from __future__ import annotations

import numpy as np

__all__ = ["rate_matched_codebook_size", "vq_gain_db", "water_filling_bits"]


def rate_matched_codebook_size(bits_per_dim: float, D: int) -> int:
    """Codebook size ``2**(bits_per_dim * D)`` -- a VQ rate-matched to per-axis SQ."""
    size = 2 ** (bits_per_dim * D)
    if size != int(size):
        raise ValueError(
            f"bits_per_dim*D = {bits_per_dim*D} is not an integer; "
            "codebook size is ambiguous"
        )
    return int(size)


def vq_gain_db(quality_vq_db: float, quality_sq_db: float) -> float:
    """VQ's advantage over SQ in dB; positive means VQ reconstructs better at equal rate."""
    return float(quality_vq_db - quality_sq_db)


def water_filling_bits(variances, total_bits: int):
    """Integer bit allocation: each bit goes to the axis with the largest
    ``variance * 4**(-b)`` (greedy high-rate proxy). Sums to ``total_bits``.
    """
    v = np.asarray(variances, dtype=float)
    b = np.zeros(len(v), dtype=int)
    for _ in range(int(round(total_bits))):
        b[int(np.argmax(v * 4.0 ** (-b)))] += 1
    return b
