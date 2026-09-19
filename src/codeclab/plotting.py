"""Shared matplotlib house style and two recurring plot shapes."""
from __future__ import annotations

from typing import Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt

__all__ = ["use_house_style", "gain_vs_x", "rd_curve", "SQ_COLOR", "VQ_COLOR"]

SQ_COLOR = "#c0392b"   # optimal scalar
VQ_COLOR = "#2471a3"   # vector
BASELINE_COLOR = "#7f8c8d"


def use_house_style() -> None:
    mpl.rcParams.update(
        {
            "figure.figsize": (7, 4.5),
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "axes.titlesize": 11,
            "legend.frameon": False,
        }
    )


def gain_vs_x(
    x: Sequence[float],
    gain_db: Sequence[float],
    *,
    xlabel: str,
    title: str,
    baseline: float | None = None,
    baseline_label: str = "classical baseline",
    ax: plt.Axes | None = None,
):
    """VQ gain (dB) against a swept variable, with an optional baseline line."""
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(x, gain_db, marker="o", color=VQ_COLOR, label="VQ gain over SQ")
    if baseline is not None:
        ax.axhline(baseline, ls="--", color=BASELINE_COLOR, label=baseline_label)
    ax.axhline(0.0, lw=0.8, color="k", alpha=0.4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("VQ gain (dB)")
    ax.set_title(title)
    ax.legend()
    return ax


def rd_curve(
    curves: dict[str, tuple[Sequence[float], Sequence[float]]],
    *,
    title: str,
    xlabel: str = "rate (bits/dim)",
    ylabel: str = "SQNR (dB)",
    ax: plt.Axes | None = None,
):
    """One line per entry in ``curves`` mapping label -> (rates, qualities)."""
    if ax is None:
        _, ax = plt.subplots()
    palette = {"SQ": SQ_COLOR, "VQ": VQ_COLOR}
    for label, (rates, quality) in curves.items():
        ax.plot(
            rates, quality, marker="o",
            color=palette.get(label),
            label=label,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    return ax
