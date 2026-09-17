"""EXP-04b -- Gaussian, quantizer-in-the-loop (control). Tracks issue #14."""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import torch

from codeclab.runctx import RunContext, load_config
from codeclab import sources
from codeclab.models import train_joint, evaluate_own_quantizer
from codeclab.plotting import use_house_style, SQ_COLOR, VQ_COLOR
from codeclab.rd import vq_gain_db

HERE = Path(__file__).parent

REF_FAIR_GAIN_LATENT_DB = 0.706
REF_FAIR_GAIN_SIGNAL_DB = -0.494


def main() -> None:
    cfg = load_config(HERE / "config.yaml")
    ctx = RunContext(experiment=cfg["experiment"], config=cfg).start()
    use_house_style()

    n = cfg["source"]["n_samples"]
    seed = cfg["source"]["seed"]
    rho = cfg["source"]["rho"]
    b = cfg["quant"]["bits_per_dim"]
    capacity = cfg["model"]["capacity"]
    outer_rounds = cfg["train"]["outer_rounds"]
    inner_epochs = cfg["train"]["inner_epochs"]
    lr = cfg["train"]["lr"]

    X = sources.gaussian_2d(rho, n, seed=seed)

    ae_sq = train_joint(X, "sq", capacity, bits_per_dim=b,
                         outer_rounds=outer_rounds, inner_epochs=inner_epochs, lr=lr, seed=seed)
    ae_vq = train_joint(X, "vq", capacity, bits_per_dim=b,
                         outer_rounds=outer_rounds, inner_epochs=inner_epochs, lr=lr, seed=seed)

    result_sq = evaluate_own_quantizer(ae_sq, X, "sq", bits_per_dim=b)
    result_vq = evaluate_own_quantizer(ae_vq, X, "vq", bits_per_dim=b)

    gain_latent_db = vq_gain_db(result_vq["sqnr_latent_db"], result_sq["sqnr_latent_db"])
    gain_signal_db = vq_gain_db(result_vq["sqnr_signal_db"], result_sq["sqnr_signal_db"])

    print(f"reference (fair, EXP-04 h=4): gain(latent)={REF_FAIR_GAIN_LATENT_DB:+.3f} dB  "
          f"gain(signal)={REF_FAIR_GAIN_SIGNAL_DB:+.3f} dB\n")
    print(f"joint (unconstrained): gain(latent)={gain_latent_db:+.3f} dB  gain(signal)={gain_signal_db:+.3f} dB")
    print(f"  SQ-encoder: bits={result_sq['bits_allocation']}  "
          f"PR={result_sq['participation_ratio']:.3f}  off-diag={result_sq['off_diag_energy']:.4f}")
    print(f"  VQ-encoder: PR={result_vq['participation_ratio']:.3f}  "
          f"off-diag={result_vq['off_diag_energy']:.4f}")

    out = {
        "gain_latent_db": gain_latent_db,
        "gain_signal_db": gain_signal_db,
        "sq_encoder": result_sq,
        "vq_encoder": result_vq,
        "reference_fair": {
            "gain_latent_db": REF_FAIR_GAIN_LATENT_DB,
            "gain_signal_db": REF_FAIR_GAIN_SIGNAL_DB,
        },
    }
    with open(ctx.path("results.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    with torch.no_grad():
        Z_sq = ae_sq.encode(torch.from_numpy(X.astype(np.float32))).numpy()
        Z_vq = ae_vq.encode(torch.from_numpy(X.astype(np.float32))).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].scatter(Z_sq[:, 0], Z_sq[:, 1], s=2, alpha=0.15, color=SQ_COLOR)
    axes[0].set_title("SQ-in-the-loop encoder: latent")
    axes[0].set_aspect("equal")
    axes[1].scatter(Z_vq[:, 0], Z_vq[:, 1], s=2, alpha=0.15, color=VQ_COLOR)
    axes[1].set_title("VQ-in-the-loop encoder: latent")
    axes[1].set_aspect("equal")
    fig.savefig(ctx.path("gaussian_joint_latents.png"), bbox_inches="tight")

    run_dir = ctx.finish(capacity=capacity)
    print(f"\nResults saved to {run_dir}")


if __name__ == "__main__":
    main()
