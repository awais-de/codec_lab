"""EXP-03 — Decorrelating transform (PCA) + SQ vs VQ.  Tracks issue #3 (rung 3).

EXP-02 showed VQ's advantage over SQ is memory gain — exploiting inter-dimension
correlation. This removes the correlation *before* quantizing, with a fixed
linear rotation (PCA), and asks whether VQ's advantage collapses back to the
EXP-01 granular floor.

Two things have to happen for PCA to help a scalar quantizer:
  1. rotate onto the principal axes  (decorrelate)
  2. re-allocate bits to the now-unequal axis variances

Rotation *alone* does nothing — Lloyd-Max distortion is linear in variance, so a
rotation conserves total distortion at equal rate. The gain is entirely in the
re-allocation. The script measures all of it:

    {no transform, PCA+equal bits, PCA+allocation}  x  {SQ, VQ}
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from codeclab.runctx import RunContext, load_config
from codeclab import sources
from codeclab.transforms import pca
from codeclab.quant.scalar import quantize_per_axis
from codeclab.quant.vector import lbg
from codeclab.metrics.toy import sqnr_db
from codeclab.rd import rate_matched_codebook_size, vq_gain_db, water_filling_bits
from codeclab.plotting import use_house_style, SQ_COLOR, VQ_COLOR

HERE = Path(__file__).parent

# EXP-01 granular floor at 2 bits/dim (0.018 space-filling + 0.376 shape)
EXP01_FLOOR_DB = 0.394


def main() -> None:
    cfg = load_config(HERE / "config.yaml")
    ctx = RunContext(experiment=cfg["experiment"], config=cfg).start()
    use_house_style()

    n = cfg["source"]["n_samples"]
    seed = cfg["source"]["seed"]
    b = cfg["quant"]["bits_per_dim"]
    eps = cfg["quant"]["lbg_eps"]
    K = rate_matched_codebook_size(b, D=2)
    total_bits = b * 2

    rows = []  # see header of metrics.csv below
    for rho in cfg["sweep"]["rho"]:
        X = sources.gaussian_2d(rho, n, seed=seed)

        # --- no transform (reproduces EXP-02) ---
        sq_plain = sqnr_db(X, quantize_per_axis(X, b))
        vq_plain = sqnr_db(X, lbg(X, K, eps=eps).quantize(X))

        # --- PCA rotation ---
        p = pca(X)
        Y = p.forward(X)
        alloc = water_filling_bits(p.explained_variance, total_bits)

        sq_pca_equal = sqnr_db(X, p.inverse(quantize_per_axis(Y, b)))
        sq_pca_alloc = sqnr_db(X, p.inverse(quantize_per_axis(Y, alloc)))
        vq_pca = sqnr_db(X, p.inverse(lbg(Y, K, eps=eps).quantize(Y)))

        g_plain = vq_gain_db(vq_plain, sq_plain)
        g_pca = vq_gain_db(vq_pca, sq_pca_alloc)
        rows.append((
            rho, sq_plain, vq_plain, sq_pca_equal, sq_pca_alloc, vq_pca,
            g_plain, g_pca, int(alloc[0]), int(alloc[1]),
            float(p.explained_variance[0]), float(p.explained_variance[1]),
        ))
        print(
            f"rho={rho:4.2f}  alloc={alloc[0]}/{alloc[1]}  "
            f"gain(plain)={g_plain:+.3f}  gain(PCA)={g_pca:+.3f}   "
            f"SQ+PCA={sq_pca_alloc:6.3f}  vs  VQ_plain={vq_plain:6.3f} dB"
        )

    arr = np.array(rows)
    with open(ctx.path("metrics.csv"), "w") as fh:
        fh.write(
            "rho,sqnr_sq_plain,sqnr_vq_plain,sqnr_sq_pca_equal,sqnr_sq_pca_alloc,"
            "sqnr_vq_pca,gain_plain_db,gain_pca_db,bits_hi,bits_lo,eval_hi,eval_lo\n"
        )
        for r in rows:
            fh.write(",".join(f"{x:.6f}" for x in r) + "\n")

    rho = arr[:, 0]

    # --- plot 1: VQ gain vs rho, no transform vs PCA+allocation ---
    fig, ax = plt.subplots()
    ax.plot(rho, arr[:, 6], marker="o", color=VQ_COLOR, label="no transform")
    ax.plot(rho, arr[:, 7], marker="s", color=SQ_COLOR, label="PCA + bit allocation")
    ax.axhline(EXP01_FLOOR_DB, ls="--", color="#7f8c8d",
               label=f"EXP-01 granular floor ({EXP01_FLOOR_DB:.2f} dB)")
    ax.axhline(0.0, lw=0.8, color="k", alpha=0.4)
    ax.set_xlabel(r"source correlation $\rho$")
    ax.set_ylabel("VQ gain over SQ (dB)")
    ax.set_title("EXP-03: PCA + allocation collapses the VQ advantage")
    ax.legend()
    fig.savefig(ctx.path("vq_gain_pca_vs_plain.png"), bbox_inches="tight")

    # --- plot 2: SQNR routes — SQ-after-PCA reaches VQ-no-transform ---
    fig2, ax2 = plt.subplots()
    ax2.plot(rho, arr[:, 2], marker="o", color=VQ_COLOR, label="VQ, no transform")
    ax2.plot(rho, arr[:, 4], marker="s", color=SQ_COLOR, label="SQ, PCA + allocation")
    ax2.plot(rho, arr[:, 3], marker="x", ls=":", color="#7f8c8d",
             label="SQ, PCA + equal bits (= no transform)")
    ax2.plot(rho, arr[:, 1], marker="^", color="#c0392b", alpha=0.5,
             label="SQ, no transform")
    ax2.set_xlabel(r"source correlation $\rho$")
    ax2.set_ylabel("SQNR (dB)")
    ax2.set_title("EXP-03: scalar quantization after PCA vs vector quantization")
    ax2.legend()
    fig2.savefig(ctx.path("sq_after_pca_vs_vq.png"), bbox_inches="tight")

    run_dir = ctx.finish(n_points=len(rows))
    print(f"\nResults saved to {run_dir}")


if __name__ == "__main__":
    main()
