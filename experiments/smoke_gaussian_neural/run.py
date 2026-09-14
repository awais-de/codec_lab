"""Smoke test for the neural harness -- not a numbered experiment.

Checks that a linear, small, and large autoencoder trained on the rho=0.85
Gaussian source all land near the EXP-03 floor rather than EXP-02's
no-transform number.
"""
from pathlib import Path
import json

from codeclab.runctx import RunContext, load_config
from codeclab import sources
from codeclab.models import train_autoencoder, evaluate_sq_vs_vq

HERE = Path(__file__).parent

REF_NO_TRANSFORM_GAIN_DB = 3.38
REF_PCA_GAIN_DB = 0.30
REF_FLOOR_DB = 0.394
PASS_TOLERANCE_DB = 0.5


def main() -> None:
    cfg = load_config(HERE / "config.yaml")
    ctx = RunContext(experiment=cfg["experiment"], config=cfg).start()

    n = cfg["source"]["n_samples"]
    seed = cfg["source"]["seed"]
    rho = cfg["source"]["rho"]
    b = cfg["quant"]["bits_per_dim"]
    rate_noise = cfg["train"]["rate_noise"]
    epochs = cfg["train"]["epochs"]
    lr = cfg["train"]["lr"]

    X = sources.gaussian_2d(rho, n, seed=seed)

    print(
        f"reference (rho={rho}): no-transform gain {REF_NO_TRANSFORM_GAIN_DB} dB, "
        f"PCA gain {REF_PCA_GAIN_DB} dB, floor {REF_FLOOR_DB} dB\n"
    )

    rows = []
    for capacity in cfg["sweep"]["capacity"]:
        ae = train_autoencoder(X, capacity=capacity, rate_noise=rate_noise,
                                epochs=epochs, lr=lr, seed=seed)
        result = evaluate_sq_vs_vq(ae, X, bits_per_dim=b)
        result["capacity"] = "linear" if capacity is None else capacity
        rows.append(result)

        tag = "linear" if capacity is None else f"h={capacity}"
        near_floor = abs(result["gain_latent_db"] - REF_FLOOR_DB) < PASS_TOLERANCE_DB
        print(
            f"capacity={tag:8s}  gain(latent)={result['gain_latent_db']:+.3f} dB  "
            f"gain(signal)={result['gain_signal_db']:+.3f} dB  "
            f"bits={result['bits_allocation']}  "
            f"off-diag energy={result['off_diag_energy']:.4f}  "
            f"participation ratio={result['participation_ratio']:.3f}  "
            f"{'OK' if near_floor else '** CHECK **'}"
        )

    with open(ctx.path("results.json"), "w") as fh:
        json.dump(rows, fh, indent=2)

    run_dir = ctx.finish(n_points=len(rows))
    print(f"\nResults saved to {run_dir}")


if __name__ == "__main__":
    main()
