"""AUDIT-01 -- reconstruction fidelity of the EXP-06/07 ring encoders. Tracks issue #16."""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np

from codeclab.runctx import RunContext, load_config
from codeclab import sources
from codeclab.models import train_autoencoder, evaluate_sq_vs_vq, reconstruction_fidelity
from codeclab.plotting import use_house_style

HERE = Path(__file__).parent


def main() -> None:
    cfg = load_config(HERE / "config.yaml")
    ctx = RunContext(experiment=cfg["experiment"], config=cfg).start()
    use_house_style()

    src, tr = cfg["source"], cfg["train"]
    X = sources.ring_2d(src["n_samples"], radius=src["radius"], radial_noise=src["radial_noise"], seed=src["seed"])
    b = cfg["quant"]["bits_per_dim"]
    bar = cfg["pass_bar"]

    rows = []
    det_traces = {}
    for name, cap in cfg["networks"].items():
        for seed in cfg["seeds"]:
            ae = train_autoencoder(X, capacity=cap, rate_noise=tr["rate_noise"], epochs=tr["epochs"], lr=tr["lr"], seed=seed)
            gain = evaluate_sq_vs_vq(ae, X, bits_per_dim=b)["gain_latent_db"]
            reproduced = None
            if seed == src["seed"]:
                reproduced = bool(np.isclose(gain, cfg["reference_gain_latent_db"][name], rtol=0, atol=1e-9))
            fid = reconstruction_fidelity(ae, X, seed=seed)
            if seed == cfg["seeds"][0]:
                det_traces[name] = (np.array(fid["det_angles"]), np.array(fid["det_samples"]))
            passed = (fid["recon_sqnr_db"] >= bar["recon_min_db"]
                      and fid["det_sign_flip_frac"] == 0.0
                      and fid["pair_ratio_min"] > bar["pair_ratio_min"])
            row = {"network": name, "capacity": cap, "seed": seed, "gain_latent_db": gain, "reproduced": reproduced,
                   **{k: v for k, v in fid.items() if k not in ("det_samples", "det_angles")}, "pass": bool(passed)}
            rows.append(row)
            print(f"{name:8s} h={cap:<3d} seed={seed}  gain={gain:+.3f}"
                  f"{'  reproduced=' + str(reproduced) if reproduced is not None else ''}"
                  f"  recon={fid['recon_sqnr_db']:6.2f} dB  |detJ| min={fid['det_min_abs']:.3f} med={fid['det_median_abs']:.3f}"
                  f"  sign-flips={fid['det_sign_flip_frac']:.4f}  pair-ratio min={fid['pair_ratio_min']:.3f} p01={fid['pair_ratio_p01']:.3f}"
                  f"  -> {'PASS' if passed else 'FAIL'}")

    with open(ctx.path("results.json"), "w") as fh:
        json.dump({"pass_bar": bar, "rows": rows}, fh, indent=2)

    fig, axes = plt.subplots(1, len(det_traces), figsize=(6 * len(det_traces), 4.5), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, (name, (ang, det)) in zip(axes, det_traces.items()):
        order = np.argsort(ang)
        ax.plot(np.degrees(ang[order]), det[order], ".", ms=3)
        ax.axhline(0.0, color="k", lw=0.8, alpha=0.5)
        ax.set_xlabel("ring angle (deg)")
        ax.set_ylabel("det J(encoder)")
        ax.set_title(f"{name} (h={cfg['networks'][name]}), seed {cfg['seeds'][0]}")
    fig.suptitle("Ring encoders, encoder Jacobian determinant along the ring")
    fig.savefig(ctx.path("det_jacobian_vs_angle.png"), bbox_inches="tight")

    run_dir = ctx.finish(n_rows=len(rows))
    print(f"\nResults saved to {run_dir}")


if __name__ == "__main__":
    main()
