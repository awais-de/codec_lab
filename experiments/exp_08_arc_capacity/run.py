"""EXP-08 -- capacity threshold vs arc openness. Tracks issue #7."""
from pathlib import Path
import argparse
import json
import multiprocessing as mp
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from codeclab.runctx import RunContext, load_config
from codeclab import sources
from codeclab.models import train_autoencoder, evaluate_sq_vs_vq
from codeclab.plotting import use_house_style

HERE = Path(__file__).parent

# EXP-05b (#19), signal space: classical no-transform gain and the ideal-straightening ceiling
REF_PLAIN_DB = {60: 1.955, 200: 3.954, 340: 3.421}
REF_CEILING_DB = {60: 0.260, 200: 0.004, 340: 0.003}


def make_source(angle_deg, src):
    X = sources.arc_2d(src["n_samples"], angle_deg, radius=src["radius"],
                       radial_noise=src["radial_noise"], seed=src["seed"])
    if src.get("normalize_total_variance"):
        X = X / np.sqrt(X.var(axis=0).sum())
    return X


def run_cell(task):
    torch.set_num_threads(task["threads"])
    X = make_source(task["angle_deg"], task["source"])
    ae = train_autoencoder(X, capacity=task["capacity"], rate_noise=task["rate_noise"],
                           epochs=task["epochs"], lr=task["lr"], seed=task["seed"])
    r = evaluate_sq_vs_vq(ae, X, bits_per_dim=task["bits_per_dim"])
    return {
        "angle_deg": task["angle_deg"], "capacity": task["capacity"],
        "rate_noise": task["rate_noise"], "seed": task["seed"],
        "source_seed": task["source"]["seed"],
        "gain_latent_db": r["gain_latent_db"], "gain_signal_db": r["gain_signal_db"],
        "sqnr_sq_latent_db": r["sqnr_sq_latent_db"], "sqnr_vq_latent_db": r["sqnr_vq_latent_db"],
        "participation_ratio": r["participation_ratio"], "off_diag_energy": r["off_diag_energy"],
        "bits_allocation": r["bits_allocation"],
    }


def cap_label(c):
    return "linear" if c is None else f"h={c}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(HERE / args.config)
    ctx = RunContext(experiment=cfg["experiment"], config=cfg).start()
    use_house_style()

    src, sw, tr = cfg["source"], cfg["sweep"], cfg["train"]
    b = cfg["quant"]["bits_per_dim"]
    tol = cfg["threshold"]["tolerance_db"]
    threads = cfg["runtime"]["threads_per_worker"]
    workers = args.workers or cfg["runtime"]["workers"] or max(1, (os.cpu_count() or 2) // 2)

    print("per-angle proxy SNR after normalisation (per-axis variance 0.5):")
    for a in sw["angle_deg"]:
        X = make_source(a, src)
        pa = X.var(axis=0).sum() / 2
        snr = "  ".join(f"w={w}: {10*np.log10(pa/(w**2/12)):5.1f} dB" for w in sw["rate_noise"])
        print(f"  α={a:>3}  total var={X.var(axis=0).sum():.4f}  {snr}")
    print("  (comparable across angles by construction; real 2-bit quantizer ≈ 13.5 dB)\n")

    tasks = [{"angle_deg": a, "capacity": c, "rate_noise": w, "seed": s, "source": src,
              "epochs": tr["epochs"], "lr": tr["lr"], "bits_per_dim": b, "threads": threads}
             for a in sw["angle_deg"] for c in sw["capacity"]
             for w in sw["rate_noise"] for s in sw["seed"]]
    sc = cfg.get("spot_check")
    spot = [{"angle_deg": sc["angle_deg"], "capacity": sc["capacity"], "rate_noise": sc["rate_noise"],
             "seed": sw["seed"][0], "source": {**src, "seed": ss}, "epochs": tr["epochs"],
             "lr": tr["lr"], "bits_per_dim": b, "threads": threads} for ss in sc["source_seeds"]] if sc else []

    print(f"{len(tasks)} grid cells + {len(spot)} spot-check cells on {workers} workers "
          f"({threads} threads each)")
    t0 = time.time()
    with mp.Pool(workers) as pool:
        rows = pool.map(run_cell, tasks)
        spot_rows = pool.map(run_cell, spot) if spot else []
    print(f"done in {(time.time()-t0)/60:.1f} min\n")

    def cell(a, c, w):
        return [r["gain_latent_db"] for r in rows
                if r["angle_deg"] == a and r["capacity"] == c and r["rate_noise"] == w]

    summary, thresholds = [], []
    for a in sw["angle_deg"]:
        print(f"--- α = {a}°   classical no-transform {REF_PLAIN_DB[a]:+.3f} dB, "
              f"ceiling {REF_CEILING_DB[a]:+.3f} dB, target ≤ {REF_CEILING_DB[a]+tol:.3f} dB ---")
        for w in sw["rate_noise"]:
            meds = {c: float(np.median(cell(a, c, w))) for c in sw["capacity"]}
            effect = max(meds.values()) - min(meds.values())
            hit = [c for c in sw["capacity"] if meds[c] <= REF_CEILING_DB[a] + tol]
            thr = cap_label(hit[0]) if hit else f"not reached <= {sw['capacity'][-1]}"
            thresholds.append({"angle_deg": a, "rate_noise": w, "threshold": thr,
                               "medians": {cap_label(c): meds[c] for c in sw["capacity"]}})
            parts = []
            for c in sw["capacity"]:
                g = cell(a, c, w)
                spread = max(g) - min(g)
                flag = "!" if spread > effect else " "
                parts.append(f"{cap_label(c)}={meds[c]:+5.2f}[{spread:4.2f}]{flag}")
                summary.append({"angle_deg": a, "rate_noise": w, "capacity": cap_label(c),
                                "median_gain_latent_db": meds[c], "min": min(g), "max": max(g),
                                "seed_spread_db": spread, "inconclusive": bool(spread > effect)})
            print(f"  w={w:<4} " + "  ".join(parts) + f"   -> threshold: {thr}")
        print()

    spot_summary = None
    if spot_rows:
        g0 = cell(sc["angle_deg"], sc["capacity"], sc["rate_noise"])
        init_spread = max(g0) - min(g0)
        sg = [r["gain_latent_db"] for r in spot_rows]
        spot_summary = {"rows": spot_rows, "source_seed_spread_db": max(sg) - min(sg),
                        "init_seed_spread_db": init_spread}
        print(f"source-seed spot check (α={sc['angle_deg']}, {cap_label(sc['capacity'])}, w={sc['rate_noise']}): "
              f"source-seed spread {max(sg)-min(sg):.3f} dB vs init-seed spread {init_spread:.3f} dB")

    with open(ctx.path("results.json"), "w") as fh:
        json.dump({"reference_exp05b": {"plain": REF_PLAIN_DB, "ceiling": REF_CEILING_DB},
                   "rows": rows, "summary": summary, "thresholds": thresholds,
                   "spot_check": spot_summary}, fh, indent=2)
    keys = ["angle_deg", "rate_noise", "capacity", "median_gain_latent_db", "min", "max",
            "seed_spread_db", "inconclusive"]
    with open(ctx.path("metrics.csv"), "w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in summary:
            fh.write(",".join(str(r[k]) for k in keys) + "\n")

    xs = range(len(sw["capacity"]))
    fig, axes = plt.subplots(1, len(sw["angle_deg"]), figsize=(5.2 * len(sw["angle_deg"]), 4.4),
                             sharey=True)
    cmap = plt.get_cmap("viridis")
    for ax, a in zip(np.atleast_1d(axes), sw["angle_deg"]):
        for k, w in enumerate(sw["rate_noise"]):
            ax.plot(list(xs), [float(np.median(cell(a, c, w))) for c in sw["capacity"]],
                    marker="o", color=cmap(k / max(len(sw["rate_noise"]) - 1, 1)), label=f"rate_noise={w}")
        ax.axhline(REF_PLAIN_DB[a], ls="--", color="#7f8c8d", label="classical, no transform")
        ax.axhline(REF_CEILING_DB[a], ls=":", color="#7f8c8d", label="ideal straightening")
        ax.set_xticks(list(xs))
        ax.set_xticklabels([cap_label(c) for c in sw["capacity"]])
        ax.set_xlabel("encoder capacity")
        ax.set_ylabel("VQ gain, latent (dB)")
        ax.set_title(f"α = {a}°")
        ax.legend(fontsize=8)
    fig.suptitle("Arc family: does capacity close the SQ-VQ gap?")
    fig.savefig(ctx.path("gain_vs_capacity.png"), bbox_inches="tight")

    run_dir = ctx.finish(n_cells=len(rows), workers=workers)
    print(f"\nResults saved to {run_dir}")


if __name__ == "__main__":
    main()
