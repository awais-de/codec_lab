"""EXP-02 — VQ gain vs source correlation (ρ sweep).  Tracks issue #1 (rung 2).

Rung 2 of the classical SQ-vs-VQ ladder. EXP-01 measured the memoryless granular
floor; here correlation is turned on and the memory gain should appear on top of
that floor as −5·log₁₀(1−ρ²) dB at D=2.
"""
from pathlib import Path

import numpy as np

from codeclab.runctx import RunContext, load_config
from codeclab import sources
from codeclab.quant.scalar import quantize_per_axis
from codeclab.quant.vector import lbg
from codeclab.metrics.toy import sqnr_db
from codeclab.rd import rate_matched_codebook_size, vq_gain_db
from codeclab.plotting import use_house_style, gain_vs_x

HERE = Path(__file__).parent


def main() -> None:
    cfg = load_config(HERE / "config.yaml")
    ctx = RunContext(experiment=cfg["experiment"], config=cfg).start()
    use_house_style()

    n = cfg["source"]["n_samples"]
    seed = cfg["source"]["seed"]
    b = cfg["quant"]["bits_per_dim"]
    eps = cfg["quant"]["lbg_eps"]
    K = rate_matched_codebook_size(b, D=2)

    rows = []  # (rho, sqnr_sq_db, sqnr_vq_db, gain_db)
    for rho in cfg["sweep"]["rho"]:
        X = sources.gaussian_2d(rho, n, seed=seed)

        # SQ baseline: independent optimal Lloyd-Max on each axis, b bits/axis.
        X_hat_sq = quantize_per_axis(X, b)

        # VQ: one LBG codebook of size K = 2**(b*D), rate-matched to the SQ above.
        X_hat_vq = lbg(X, K, eps=eps).quantize(X)

        sqnr_sq = sqnr_db(X, X_hat_sq)
        sqnr_vq = sqnr_db(X, X_hat_vq)
        gain = vq_gain_db(sqnr_vq, sqnr_sq)

        rows.append((rho, sqnr_sq, sqnr_vq, gain))
        print(
            f"rho={rho:4.2f}  "
            f"SQNR_SQ={sqnr_sq:6.3f} dB  SQNR_VQ={sqnr_vq:6.3f} dB  "
            f"gain={gain:+.3f} dB"
        )

    # --- save ---
    arr = np.array(rows)
    np.savetxt(
        ctx.path("metrics.csv"), arr,
        header="rho,sqnr_sq_db,sqnr_vq_db,gain_db", delimiter=",", comments="",
    )

    ax = gain_vs_x(
        arr[:, 0], arr[:, 3],
        xlabel=r"source correlation $\rho$",
        title="EXP-01: VQ gain over optimal SQ vs source correlation",
    )
    ax.figure.savefig(ctx.path("vq_gain_vs_rho.png"), bbox_inches="tight")

    run_dir = ctx.finish(n_points=len(rows))
    print(f"Results saved to {run_dir}")


if __name__ == "__main__":
    main()
