"""AUDIT-03 -- warm-start at the fold, continue under the blind proxy. Tracks issue #18."""
from pathlib import Path
import copy
import json

import matplotlib.pyplot as plt
import numpy as np
import torch

from codeclab.runctx import RunContext, load_config
from codeclab import sources
from codeclab.folds import fold_ring
from codeclab.models import (AutoEncoder, evaluate_sq_vs_vq, fit_to_map,
                             continue_training, marginal_modes)
from codeclab.plotting import use_house_style

HERE = Path(__file__).parent
CLASSICAL_GAIN_DB = 3.29
FLOOR_GAIN_DB = 0.39


def snapshot(ae, X, b):
    ev = evaluate_sq_vs_vq(ae, X, bits_per_dim=b)
    with torch.no_grad():
        Z = ae.encode(torch.from_numpy(X.astype(np.float32))).numpy()
    return {"gain_latent_db": ev["gain_latent_db"], "gain_signal_db": ev["gain_signal_db"],
            "participation_ratio": ev["participation_ratio"], "modes": marginal_modes(Z)}, Z


def main() -> None:
    cfg = load_config(HERE / "config.yaml")
    ctx = RunContext(experiment=cfg["experiment"], config=cfg).start()
    use_house_style()

    src = cfg["source"]
    X = sources.ring_2d(src["n_samples"], radius=src["radius"], radial_noise=src["radial_noise"], seed=src["seed"])
    b = cfg["quant"]["bits_per_dim"]
    Z_fold = fold_ring(X, radius=src["radius"], perp_scale=cfg["fold"]["perp_scale"])
    ws, ct = cfg["warmstart"], cfg["continue"]

    warm_rows, traj, final = [], [], []
    scatters = {}
    for name, cap in cfg["networks"].items():
        for seed in cfg["seeds"]:
            torch.manual_seed(seed)
            ae = AutoEncoder(dim=2, capacity=cap)
            fit = fit_to_map(ae, X, Z_fold, epochs=ws["epochs"], lr=ws["lr"], seed=seed)
            snap, Z0 = snapshot(ae, X, b)
            accepted = bool(snap["gain_latent_db"] <= cfg["warmstart_max_gain_db"])
            warm_rows.append({"network": name, "capacity": cap, "seed": seed, **fit, **snap, "accepted": accepted})
            print(f"[warm start] {name:8s} seed={seed}  fit mse enc/dec={fit['fit_mse_encoder']:.4f}/{fit['fit_mse_decoder']:.4f}"
                  f"  gain={snap['gain_latent_db']:+.2f} dB  PR={snap['participation_ratio']:.2f}  modes={snap['modes']}"
                  f"  -> {'accepted' if accepted else 'rejected'}")
            if not accepted:
                continue
            for rn in cfg["rate_noises"]:
                ae_rn = copy.deepcopy(ae)

                def cb(epoch, m, _name=name, _seed=seed, _rn=rn):
                    s, _ = snapshot(m, X, b)
                    traj.append({"network": _name, "seed": _seed, "rate_noise": _rn, "epoch": epoch, **s})

                continue_training(ae_rn, X, rate_noise=rn, epochs=ct["epochs"], lr=ct["lr"],
                                  log_every=ct["log_every"], seed=seed, callback=cb)
                s_end, Z_end = snapshot(ae_rn, X, b)
                final.append({"network": name, "seed": seed, "rate_noise": rn, **s_end})
                if seed == cfg["seeds"][0]:
                    scatters[(name, rn)] = (Z0, Z_end)
                print(f"    rate_noise={rn:.1f}  gain {snap['gain_latent_db']:+.2f} -> {s_end['gain_latent_db']:+.2f} dB"
                      f"  PR {snap['participation_ratio']:.2f} -> {s_end['participation_ratio']:.2f}"
                      f"  modes {snap['modes']} -> {s_end['modes']}")

    print("\n[verdict] median final gain (dB) across seeds, classical 3.29, floor ~0.4")
    verdict = []
    for name in cfg["networks"]:
        for rn in cfg["rate_noises"]:
            g = [r["gain_latent_db"] for r in final if r["network"] == name and r["rate_noise"] == rn]
            if not g:
                continue
            m = [min(r["modes"]) for r in final if r["network"] == name and r["rate_noise"] == rn]
            med = float(np.median(g))
            state = "fold survived" if med < 1.0 else ("unfolded" if med > 2.5 else "partial")
            verdict.append({"network": name, "rate_noise": rn, "median_gain_latent_db": med,
                            "median_min_modes": float(np.median(m)), "state": state})
            print(f"  {name:8s} rate_noise={rn:.1f}  median gain={med:+.2f}  min-modes={np.median(m):.0f}  -> {state}")

    with open(ctx.path("results.json"), "w") as fh:
        json.dump({"warm_start": warm_rows, "final": final, "verdict": verdict}, fh, indent=2)
    with open(ctx.path("trajectories.csv"), "w") as fh:
        fh.write("network,seed,rate_noise,epoch,gain_latent_db,gain_signal_db,participation_ratio,modes0,modes1\n")
        for r in traj:
            fh.write(f"{r['network']},{r['seed']},{r['rate_noise']},{r['epoch']},{r['gain_latent_db']},"
                     f"{r['gain_signal_db']},{r['participation_ratio']},{r['modes'][0]},{r['modes'][1]}\n")

    nets = [n for n in cfg["networks"] if any(r["network"] == n for r in final)]
    if nets:
        fig, axes = plt.subplots(1, len(nets), figsize=(6 * len(nets), 4.5), sharey=True)
        axes = np.atleast_1d(axes)
        cmap = plt.get_cmap("viridis")
        for ax, name in zip(axes, nets):
            for k, rn in enumerate(cfg["rate_noises"]):
                for seed in cfg["seeds"]:
                    pts = [(r["epoch"], r["gain_latent_db"]) for r in traj
                           if r["network"] == name and r["rate_noise"] == rn and r["seed"] == seed]
                    if not pts:
                        continue
                    e, g = zip(*pts)
                    ax.plot(e, g, color=cmap(k / max(len(cfg["rate_noises"]) - 1, 1)),
                            lw=1.8 if seed == cfg["seeds"][0] else 0.7, alpha=1.0 if seed == cfg["seeds"][0] else 0.5,
                            label=f"rate_noise={rn}" if seed == cfg["seeds"][0] else None)
            ax.axhline(CLASSICAL_GAIN_DB, ls="--", color="#7f8c8d", label="classical (no encoder)")
            ax.axhline(FLOOR_GAIN_DB, ls=":", color="#7f8c8d", label="memoryless floor")
            ax.set_xlabel("epochs of blind-proxy training after the warm start")
            ax.set_ylabel("VQ gain, latent (dB)")
            ax.set_title(f"{name} (capacity {cfg['networks'][name]})")
            ax.legend(fontsize=8)
        fig.suptitle("Ring source, warm-started at the fold: does the blind proxy unfold it?")
        fig.savefig(ctx.path("gain_vs_epoch.png"), bbox_inches="tight")

    keys = [k for k in scatters if k[0] == nets[0]] if nets else []
    if keys:
        fig, axes = plt.subplots(2, len(keys), figsize=(4.5 * len(keys), 9))
        axes = np.atleast_2d(axes)
        for col, key in enumerate(keys):
            Z0, Z1 = scatters[key]
            for row, (Z, tag) in enumerate(((Z0, "before"), (Z1, "after"))):
                ax = axes[row, col]
                ax.scatter(Z[:, 0], Z[:, 1], s=2, alpha=0.15)
                ax.set_aspect("equal")
                ax.set_title(f"{key[0]}, rate_noise={key[1]}: {tag}")
        fig.suptitle("Latent before and after blind-proxy training (seed 0)")
        fig.savefig(ctx.path("latent_before_after.png"), bbox_inches="tight")

    run_dir = ctx.finish(n_final=len(final))
    print(f"\nResults saved to {run_dir}")


if __name__ == "__main__":
    main()
