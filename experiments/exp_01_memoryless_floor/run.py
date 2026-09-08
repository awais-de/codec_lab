"""EXP-01 — Memoryless SQ vs VQ: the granular floor.  Tracks issue #11 (rung 1).

Rung 1 of the classical SQ-vs-VQ ladder. With an i.i.d. source there is nothing
for VQ's *memory gain* to exploit, so it is zero by definition. Whatever edge
LBG still holds over per-axis Lloyd-Max is the **granular floor**:

    granular floor  =  space-filling gain  +  shape gain

- space-filling gain: granular cell geometry only (hexagons beat squares).
  Source-independent, -> 0.167 dB in 2D as R -> infinity.
- shape gain: from the marginal pdf. Zero for a uniform source (a flat pdf makes
  uniform point density optimal), positive for a Gaussian.

Sweeping a Gaussian and a uniform source together splits the floor: the uniform
gain is space-filling alone; Gaussian minus uniform is the shape gain.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from codeclab.runctx import RunContext, load_config
from codeclab import sources
from codeclab.quant.scalar import quantize_per_axis
from codeclab.quant.vector import lbg
from codeclab.metrics.toy import sqnr_db
from codeclab.rd import rate_matched_codebook_size, vq_gain_db
from codeclab.plotting import use_house_style, SQ_COLOR, VQ_COLOR

HERE = Path(__file__).parent

# hexagonal vs square granular cells: 10 log10( (1/12) / (5 / (36 sqrt 3)) )
SPACE_FILLING_2D_DB = float(10.0 * np.log10((1 / 12) / (5 / (36 * np.sqrt(3)))))

# fixed-rate optimal scalar (Lloyd-Max) SQNR for a unit Gaussian — standard table
GAUSS_LLOYD_MAX_DB = {1: 4.40, 2: 9.30, 3: 14.62, 4: 20.22, 5: 26.01}


def main() -> None:
    cfg = load_config(HERE / "config.yaml")
    ctx = RunContext(experiment=cfg["experiment"], config=cfg).start()
    use_house_style()

    n = cfg["source"]["n_samples"]
    seed = cfg["source"]["seed"]
    eps = cfg["quant"]["lbg_eps"]
    rates = list(cfg["sweep"]["rate"])

    print(f"reference: hexagonal space-filling gain (R->inf) = {SPACE_FILLING_2D_DB:.3f} dB\n")

    make_source = {
        "gaussian": lambda: sources.gaussian_2d(0.0, n, seed=seed),
        "uniform": lambda: sources.uniform_2d(n, seed=seed),
    }

    rows = []  # (rate, source, sqnr_sq_db, sqnr_vq_db, gain_db)
    for R in rates:
        K = rate_matched_codebook_size(R, D=2)
        for name, make in make_source.items():
            X = make()
            X_hat_sq = quantize_per_axis(X, R)
            X_hat_vq = lbg(X, K, eps=eps).quantize(X)
            s_sq = sqnr_db(X, X_hat_sq)
            s_vq = sqnr_db(X, X_hat_vq)
            g = vq_gain_db(s_vq, s_sq)
            rows.append((R, name, s_sq, s_vq, g))
            print(
                f"R={R}  K={K:4d}  {name:8s}  "
                f"SQNR_SQ={s_sq:6.2f} dB  SQNR_VQ={s_vq:6.2f} dB  gain={g:+.3f} dB"
            )

    def gain(R, name):
        return next(r[4] for r in rows if r[0] == R and r[1] == name)

    def sqnr_sq(R, name):
        return next(r[2] for r in rows if r[0] == R and r[1] == name)

    # --- decomposition: uniform gain = space-filling, Gaussian - uniform = shape ---
    print("\ngranular-floor decomposition (dB):")
    dec = []
    for R in rates:
        g_uni, g_gau = gain(R, "uniform"), gain(R, "gaussian")
        dec.append((R, g_uni, g_gau - g_uni, g_gau))
        print(
            f"  R={R}:  space-filling ~ {g_uni:+.3f}   "
            f"shape ~ {g_gau - g_uni:+.3f}   total(Gaussian) = {g_gau:+.3f}"
        )

    # --- save ---
    with open(ctx.path("metrics.csv"), "w") as fh:
        fh.write("rate,source,sqnr_sq_db,sqnr_vq_db,vq_gain_db\n")
        for R, name, s_sq, s_vq, g in rows:
            fh.write(f"{R},{name},{s_sq:.6f},{s_vq:.6f},{g:.6f}\n")

    with open(ctx.path("decomposition.csv"), "w") as fh:
        fh.write("rate,space_filling_db,shape_db,total_gaussian_db\n")
        for R, sf, sh, tot in dec:
            fh.write(f"{R},{sf:.6f},{sh:.6f},{tot:.6f}\n")

    # --- plot 1: the granular floor ---
    fig, ax = plt.subplots()
    ax.plot(rates, [gain(R, "gaussian") for R in rates], marker="o",
            color=VQ_COLOR, label="i.i.d. Gaussian  (space-filling + shape)")
    ax.plot(rates, [gain(R, "uniform") for R in rates], marker="s",
            color=SQ_COLOR, label="i.i.d. uniform  (space-filling only)")
    ax.axhline(SPACE_FILLING_2D_DB, ls="--", color="#7f8c8d",
               label=f"hexagonal space-filling, R→∞  ({SPACE_FILLING_2D_DB:.2f} dB)")
    ax.axhline(0.0, lw=0.8, color="k", alpha=0.4)
    ax.set_xlabel("rate R (bits/dim)")
    ax.set_ylabel("VQ gain over optimal SQ (dB)")
    ax.set_title("EXP-01: the granular floor — memoryless SQ vs VQ")
    ax.set_xticks(rates)
    ax.legend()
    fig.savefig(ctx.path("granular_floor.png"), bbox_inches="tight")

    # --- plot 2: scalar-quantizer sanity vs closed form ---
    rr = np.array(rates, dtype=float)
    fig2, ax2 = plt.subplots()
    ax2.plot(rr, [sqnr_sq(R, "uniform") for R in rates], marker="s",
             color=SQ_COLOR, label="uniform SQ (measured)")
    ax2.plot(rr, 6.02 * rr, ls="--", color=SQ_COLOR, alpha=0.6,
             label="6.02·R  (uniform, exact)")
    ax2.plot(rr, [sqnr_sq(R, "gaussian") for R in rates], marker="o",
             color=VQ_COLOR, label="Gaussian SQ (measured)")
    ax2.plot(rr, [GAUSS_LLOYD_MAX_DB.get(int(R), np.nan) for R in rates], ls=":",
             color=VQ_COLOR, alpha=0.7, label="Gaussian Lloyd–Max (table)")
    ax2.set_xlabel("rate R (bits/dim)")
    ax2.set_ylabel("SQNR (dB)")
    ax2.set_title("EXP-01: scalar-quantizer sanity check")
    ax2.set_xticks(rates)
    ax2.legend()
    fig2.savefig(ctx.path("sq_sanity.png"), bbox_inches="tight")

    run_dir = ctx.finish(n_points=len(rows))
    print(f"\nResults saved to {run_dir}")


if __name__ == "__main__":
    main()
