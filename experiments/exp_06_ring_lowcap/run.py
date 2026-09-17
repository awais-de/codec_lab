"""EXP-06 -- low-capacity neural codec on the ring source. Tracks issue #5."""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import torch

from codeclab.runctx import RunContext, load_config
from codeclab import sources
from codeclab.diagnostics import latent_diagnostics
from codeclab.models import train_autoencoder, evaluate_sq_vs_vq
from codeclab.plotting import use_house_style, SQ_COLOR, VQ_COLOR
from codeclab.quant.scalar import quantize_per_axis
from codeclab.quant.vector import lbg
from codeclab.rd import rate_matched_codebook_size, water_filling_bits

HERE = Path(__file__).parent

REF_CLASSICAL_GAIN_DB = 3.29


def main() -> None:
    cfg = load_config(HERE / "config.yaml")
    ctx = RunContext(experiment=cfg["experiment"], config=cfg).start()
    use_house_style()

    n = cfg["source"]["n_samples"]
    seed = cfg["source"]["seed"]
    radius = cfg["source"]["radius"]
    radial_noise = cfg["source"]["radial_noise"]
    b = cfg["quant"]["bits_per_dim"]
    capacity = cfg["model"]["capacity"]
    rate_noise = cfg["train"]["rate_noise"]
    epochs = cfg["train"]["epochs"]
    lr = cfg["train"]["lr"]

    X = sources.ring_2d(n, radius=radius, radial_noise=radial_noise, seed=seed)

    ae = train_autoencoder(X, capacity=capacity, rate_noise=rate_noise,
                            epochs=epochs, lr=lr, seed=seed)
    result = evaluate_sq_vs_vq(ae, X, bits_per_dim=b)

    print(f"reference (classical, no transform): gain {REF_CLASSICAL_GAIN_DB:+.3f} dB\n")
    print(
        f"capacity=h={capacity}  gain(latent)={result['gain_latent_db']:+.3f} dB  "
        f"gain(signal)={result['gain_signal_db']:+.3f} dB  "
        f"off-diag energy={result['off_diag_energy']:.4f}  "
        f"participation ratio={result['participation_ratio']:.3f}"
    )

    with open(ctx.path("results.json"), "w") as fh:
        json.dump(result, fh, indent=2)

    ae.eval()
    with torch.no_grad():
        Z = ae.encode(torch.from_numpy(X.astype(np.float32))).numpy()

    diag = latent_diagnostics(Z)
    K = rate_matched_codebook_size(b, D=2)
    bits = water_filling_bits(np.diag(diag["cov"]), b * 2)
    Z_hat_sq = quantize_per_axis(Z, bits)
    Z_hat_vq = lbg(Z, K).quantize(Z)
    with torch.no_grad():
        X_hat_sq = ae.decoder(torch.from_numpy(Z_hat_sq.astype(np.float32))).numpy()
        X_hat_vq = ae.decoder(torch.from_numpy(Z_hat_vq.astype(np.float32))).numpy()

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for ax, pts, color, title in (
        (axes[0, 0], X_hat_sq, SQ_COLOR, "SQ reconstruction (signal space)"),
        (axes[0, 1], X_hat_vq, VQ_COLOR, "VQ reconstruction (signal space)"),
    ):
        ax.scatter(X[:, 0], X[:, 1], s=2, alpha=0.15, color="#7f8c8d", label="source")
        ax.scatter(pts[:, 0], pts[:, 1], s=12, color=color, label="reconstruction")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.legend(markerscale=3)
    for ax, pts, color, title in (
        (axes[1, 0], Z_hat_sq, SQ_COLOR, "SQ reconstruction (latent space)"),
        (axes[1, 1], Z_hat_vq, VQ_COLOR, "VQ reconstruction (latent space)"),
    ):
        ax.scatter(Z[:, 0], Z[:, 1], s=2, alpha=0.15, color="#7f8c8d", label="latent")
        ax.scatter(pts[:, 0], pts[:, 1], s=12, color=color, label="reconstruction")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.legend(markerscale=3)
    fig.suptitle("Ring source, neural encoder h=4")
    fig.savefig(ctx.path("ring_lowcap_sq_vs_vq.png"), bbox_inches="tight")

    run_dir = ctx.finish(capacity=capacity)
    print(f"\nResults saved to {run_dir}")


if __name__ == "__main__":
    main()
