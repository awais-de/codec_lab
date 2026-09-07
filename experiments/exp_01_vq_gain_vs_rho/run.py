"""EXP-01 — VQ gain vs source correlation (ρ sweep).  Tracks issue #1.

Skeleton: the plumbing (config, run directory, saving) is wired up. The sweep
itself is left to fill in — that's the experiment.
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

        # TODO: SQ baseline — quantize_per_axis(X, b) -> X_hat_sq
        # TODO: VQ           — lbg(X, K, eps=eps).quantize(X) -> X_hat_vq
        # TODO: sqnr_db(X, X_hat_*) for each, then vq_gain_db(...)
        raise NotImplementedError("fill in the sweep body")

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
    print(f"wrote {run_dir}")


if __name__ == "__main__":
    main()
