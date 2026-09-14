"""EXP-05 -- classical SQ vs VQ baseline on a ring source. Tracks issue #13."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from codeclab.runctx import RunContext, load_config
from codeclab import sources
from codeclab.quant.scalar import quantize_per_axis
from codeclab.quant.vector import lbg
from codeclab.metrics.toy import sqnr_db
from codeclab.rd import rate_matched_codebook_size, vq_gain_db
from codeclab.plotting import use_house_style, SQ_COLOR, VQ_COLOR

HERE = Path(__file__).parent


def main() -> None:
    cfg = load_config(HERE / "config.yaml")
    ctx = RunContext(experiment=cfg["experiment"], config=cfg).start()
    use_house_style()

    n = cfg["source"]["n_samples"]
    seed = cfg["source"]["seed"]
    radius = cfg["source"]["radius"]
    radial_noise = cfg["source"]["radial_noise"]
    b = cfg["quant"]["bits_per_dim"]
    eps = cfg["quant"]["lbg_eps"]

    X = sources.ring_2d(n, radius=radius, radial_noise=radial_noise, seed=seed)
    cov = np.cov(X, rowvar=False)
    print(f"source covariance:\n{cov}\n")

    K = rate_matched_codebook_size(b, D=2)
    X_hat_sq = quantize_per_axis(X, b)
    X_hat_vq = lbg(X, K, eps=eps).quantize(X)

    sqnr_sq = sqnr_db(X, X_hat_sq)
    sqnr_vq = sqnr_db(X, X_hat_vq)
    gain = vq_gain_db(sqnr_vq, sqnr_sq)
    print(f"SQNR_SQ={sqnr_sq:.3f} dB  SQNR_VQ={sqnr_vq:.3f} dB  gain={gain:+.3f} dB")

    with open(ctx.path("metrics.csv"), "w") as fh:
        fh.write("sqnr_sq_db,sqnr_vq_db,gain_db,cov_00,cov_01,cov_11\n")
        fh.write(f"{sqnr_sq},{sqnr_vq},{gain},{cov[0, 0]},{cov[0, 1]},{cov[1, 1]}\n")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    for ax, X_hat, color, title in (
        (axes[0], X_hat_sq, SQ_COLOR, "SQ reconstruction"),
        (axes[1], X_hat_vq, VQ_COLOR, "VQ reconstruction"),
    ):
        ax.scatter(X[:, 0], X[:, 1], s=2, alpha=0.15, color="#7f8c8d", label="source")
        ax.scatter(X_hat[:, 0], X_hat[:, 1], s=12, color=color, label="reconstruction")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.legend(markerscale=3)
    fig.savefig(ctx.path("ring_sq_vs_vq.png"), bbox_inches="tight")

    run_dir = ctx.finish(K=K)
    print(f"\nResults saved to {run_dir}")


if __name__ == "__main__":
    main()
