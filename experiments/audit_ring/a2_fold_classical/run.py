"""AUDIT-02 -- hand-built ring encoders (none / seam / fold) with SQ vs VQ. Tracks issue #17."""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np

from codeclab.runctx import RunContext, load_config
from codeclab import sources
from codeclab.folds import fold_ring, unfold_ring, matched_perp_scale, polar_seam, unpolar
from codeclab.metrics.toy import sqnr_db
from codeclab.plotting import use_house_style, SQ_COLOR, VQ_COLOR
from codeclab.quant.scalar import quantize_per_axis
from codeclab.quant.vector import lbg
from codeclab.rd import rate_matched_codebook_size, vq_gain_db, water_filling_bits

HERE = Path(__file__).parent

REF_EXP05 = {"sqnr_sq_db": 13.164, "sqnr_vq_db": 16.454, "gain_db": 3.290}


def ideal_points_ceiling_db(sigma: float, K: int) -> float:
    sig = (1 + sigma ** 2) / 2
    mse = (sigma ** 2 + (2 * np.pi / K) ** 2 / 12) / 2
    return float(10 * np.log10(sig / mse))


def main() -> None:
    cfg = load_config(HERE / "config.yaml")
    ctx = RunContext(experiment=cfg["experiment"], config=cfg).start()
    use_house_style()

    src = cfg["source"]
    X = sources.ring_2d(src["n_samples"], radius=src["radius"], radial_noise=src["radial_noise"], seed=src["seed"])
    r = np.hypot(X[:, 0], X[:, 1])
    b = cfg["quant"]["bits_per_dim"]
    eps = cfg["quant"]["lbg_eps"]
    K = rate_matched_codebook_size(b, D=2)
    total_bits = b * 2

    rows = [("none", lambda Z: Z, lambda Z: Z, None),
            ("seam", polar_seam, unpolar, None)]
    for label, ps in cfg["fold"]["perp_scales"].items():
        scale = matched_perp_scale(src["radius"]) if ps is None else float(ps)
        rows.append((f"fold ({label})",
                     lambda Z, s=scale: fold_ring(Z, radius=src["radius"], perp_scale=s),
                     lambda Z, s=scale: unfold_ring(Z, radius=src["radius"], perp_scale=s),
                     scale))

    ceiling = ideal_points_ceiling_db(src["radial_noise"], K)
    print(f"closed-form ceiling, {K} ideal points on the circle: {ceiling:.3f} dB")
    print(f"EXP-05 reference: SQ {REF_EXP05['sqnr_sq_db']} / VQ {REF_EXP05['sqnr_vq_db']} / gain +{REF_EXP05['gain_db']} dB\n")

    results = []
    recon = {}
    for name, enc, dec, scale in rows:
        Z = enc(X)
        bits = water_filling_bits(Z.var(axis=0), total_bits)
        Z_sq = quantize_per_axis(Z, bits)
        Z_vq = lbg(Z, K, eps=eps).quantize(Z)
        X_sq, X_vq = dec(Z_sq), dec(Z_vq)
        crossing = float(np.mean(np.abs(r - src["radius"]) * scale >= 0.5)) if scale is not None else 0.0
        row = {
            "encoder": name, "bits": bits.tolist(), "perp_scale": scale,
            "sqnr_sq_latent_db": sqnr_db(Z, Z_sq), "sqnr_vq_latent_db": sqnr_db(Z, Z_vq),
            "sqnr_sq_signal_db": sqnr_db(X, X_sq), "sqnr_vq_signal_db": sqnr_db(X, X_vq),
            "strand_crossing_frac": crossing,
        }
        row["gain_latent_db"] = vq_gain_db(row["sqnr_vq_latent_db"], row["sqnr_sq_latent_db"])
        row["gain_signal_db"] = vq_gain_db(row["sqnr_vq_signal_db"], row["sqnr_sq_signal_db"])
        results.append(row)
        recon[name] = (X_sq, X_vq)
        print(f"{name:18s} bits={bits.tolist()}  latent: SQ {row['sqnr_sq_latent_db']:6.2f} VQ {row['sqnr_vq_latent_db']:6.2f} "
              f"gain {row['gain_latent_db']:+5.2f} | signal: SQ {row['sqnr_sq_signal_db']:6.2f} VQ {row['sqnr_vq_signal_db']:6.2f} "
              f"gain {row['gain_signal_db']:+5.2f} | crossing {crossing:.3f}")

    with open(ctx.path("metrics.csv"), "w") as fh:
        keys = ["encoder", "bits", "perp_scale", "sqnr_sq_latent_db", "sqnr_vq_latent_db", "gain_latent_db",
                "sqnr_sq_signal_db", "sqnr_vq_signal_db", "gain_signal_db", "strand_crossing_frac"]
        fh.write(",".join(keys) + "\n")
        for row in results:
            fh.write(",".join(str(row[k]).replace(",", ";") for k in keys) + "\n")
    with open(ctx.path("results.json"), "w") as fh:
        json.dump({"ceiling_db": ceiling, "reference_exp05": REF_EXP05, "rows": results}, fh, indent=2)

    fig, axes = plt.subplots(len(rows), 2, figsize=(10, 5 * len(rows)))
    for i, (name, _, _, _) in enumerate(rows):
        X_sq, X_vq = recon[name]
        for ax, pts, color, tag in ((axes[i, 0], X_sq, SQ_COLOR, "SQ"), (axes[i, 1], X_vq, VQ_COLOR, "VQ")):
            ax.scatter(X[:, 0], X[:, 1], s=2, alpha=0.15, color="#7f8c8d", label="source")
            ax.scatter(pts[:, 0], pts[:, 1], s=12, color=color, label="reconstruction")
            ax.set_title(f"{name}: {tag} reconstruction (signal space)")
            ax.set_aspect("equal")
            ax.legend(markerscale=3)
    fig.suptitle("Ring source, hand-built encoders")
    fig.savefig(ctx.path("fold_classical_sq_vs_vq.png"), bbox_inches="tight")

    run_dir = ctx.finish(K=K, ceiling_db=ceiling)
    print(f"\nResults saved to {run_dir}")


if __name__ == "__main__":
    main()
