"""EXP-05b -- classical SQ vs VQ across arc openness. Tracks issue #19."""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np

from codeclab.runctx import RunContext, load_config
from codeclab import sources
from codeclab.folds import straighten_arc, unstraighten_arc
from codeclab.metrics.toy import sqnr_db
from codeclab.plotting import use_house_style, SQ_COLOR, VQ_COLOR
from codeclab.quant.scalar import quantize_per_axis
from codeclab.quant.vector import lbg
from codeclab.rd import rate_matched_codebook_size, vq_gain_db, water_filling_bits
from codeclab.transforms import pca

HERE = Path(__file__).parent

REF_EXP05 = {"sqnr_sq_db": 13.163791324137158, "sqnr_vq_db": 16.454004497551217,
             "gain_db": 3.2902131734140596}


def main() -> None:
    cfg = load_config(HERE / "config.yaml")
    ctx = RunContext(experiment=cfg["experiment"], config=cfg).start()
    use_house_style()

    src = cfg["source"]
    n, radius, noise, seed = src["n_samples"], src["radius"], src["radial_noise"], src["seed"]
    b, eps = cfg["quant"]["bits_per_dim"], cfg["quant"]["lbg_eps"]
    K = rate_matched_codebook_size(b, D=2)
    total_bits = b * 2
    angles = cfg["sweep"]["angle_deg"]

    print(f"reference EXP-05 (ring): SQ {REF_EXP05['sqnr_sq_db']:.3f}  VQ {REF_EXP05['sqnr_vq_db']:.3f}"
          f"  gain {REF_EXP05['gain_db']:+.3f} dB\n")

    rows, scatters = [], {}
    for a in angles:
        X = sources.arc_2d(n, a, radius=radius, radial_noise=noise, seed=seed)
        cov_meas, cov_form = np.cov(X, rowvar=False), sources.arc_2d_covariance(
            a, radius=radius, radial_noise=noise)
        cov_err = float(np.abs(cov_meas - cov_form).max())

        sq_plain = sqnr_db(X, quantize_per_axis(X, b))
        vq_plain = sqnr_db(X, lbg(X, K, eps=eps).quantize(X))

        p = pca(X)
        Y = p.forward(X)
        alloc_pca = water_filling_bits(p.explained_variance, total_bits)
        sq_pca_equal = sqnr_db(X, p.inverse(quantize_per_axis(Y, b)))
        sq_pca = sqnr_db(X, p.inverse(quantize_per_axis(Y, alloc_pca)))
        vq_pca = sqnr_db(X, p.inverse(lbg(Y, K, eps=eps).quantize(Y)))

        Z = straighten_arc(X, angle_deg=a, radius=radius)
        alloc_str = water_filling_bits(Z.var(axis=0), total_bits)
        sq_str = sqnr_db(X, unstraighten_arc(quantize_per_axis(Z, alloc_str), angle_deg=a, radius=radius))
        vq_str = sqnr_db(X, unstraighten_arc(lbg(Z, K, eps=eps).quantize(Z), angle_deg=a, radius=radius))

        row = {
            "angle_deg": a, "cov_max_err": cov_err,
            "cov_measured": cov_meas.tolist(), "cov_closed_form": cov_form.tolist(),
            "sqnr_sq_plain": sq_plain, "sqnr_vq_plain": vq_plain,
            "gain_plain_db": vq_gain_db(vq_plain, sq_plain),
            "sqnr_sq_pca_equal": sq_pca_equal, "sqnr_sq_pca": sq_pca, "sqnr_vq_pca": vq_pca,
            "gain_pca_db": vq_gain_db(vq_pca, sq_pca), "alloc_pca": alloc_pca.tolist(),
            "sqnr_sq_straight": sq_str, "sqnr_vq_straight": vq_str,
            "gain_straight_db": vq_gain_db(vq_str, sq_str), "alloc_straight": alloc_str.tolist(),
        }
        rows.append(row)
        scatters[a] = X
        print(f"angle={a:>3}  cov err={cov_err:.1e}  |  no transform: SQ {sq_plain:6.3f} VQ {vq_plain:6.3f} "
              f"gain {row['gain_plain_db']:+6.3f}  |  PCA{alloc_pca.tolist()}: SQ {sq_pca:6.3f} VQ {vq_pca:6.3f} "
              f"gain {row['gain_pca_db']:+6.3f}  |  straightened{alloc_str.tolist()}: SQ {sq_str:6.3f} "
              f"VQ {vq_str:6.3f} gain {row['gain_straight_db']:+6.3f}")

    ring = next(r for r in rows if r["angle_deg"] == 360)
    checks = {k: bool(np.isclose(ring[k], REF_EXP05[v], atol=5e-3))
              for k, v in (("sqnr_sq_plain", "sqnr_sq_db"), ("sqnr_vq_plain", "sqnr_vq_db"),
                           ("gain_plain_db", "gain_db"))}
    print(f"\nalpha=360 reproduces EXP-05: {checks}")

    with open(ctx.path("results.json"), "w") as fh:
        json.dump({"reference_exp05": REF_EXP05, "exp05_reproduced": checks, "rows": rows}, fh, indent=2)
    keys = ["angle_deg", "sqnr_sq_plain", "sqnr_vq_plain", "gain_plain_db", "sqnr_sq_pca_equal",
            "sqnr_sq_pca", "sqnr_vq_pca", "gain_pca_db", "sqnr_sq_straight", "sqnr_vq_straight",
            "gain_straight_db", "cov_max_err"]
    with open(ctx.path("metrics.csv"), "w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(f"{r[k]}" for k in keys) + "\n")

    fig, ax = plt.subplots()
    ax.plot(angles, [r["gain_plain_db"] for r in rows], marker="o", color=VQ_COLOR, label="no transform")
    ax.plot(angles, [r["gain_pca_db"] for r in rows], marker="s", color=SQ_COLOR, label="PCA + bit allocation")
    ax.plot(angles, [r["gain_straight_db"] for r in rows], marker="^", ls="--", color="#7f8c8d",
            label="ideal straightening (ceiling)")
    ax.axhline(0.0, lw=0.8, color="k", alpha=0.4)
    ax.set_xlabel("arc angle α (degrees)")
    ax.set_ylabel("VQ gain over SQ (dB)")
    ax.set_title("EXP-05b: VQ gain vs arc openness")
    ax.set_xticks(angles)
    ax.legend()
    fig.savefig(ctx.path("gain_vs_angle.png"), bbox_inches="tight")

    fig2, axes = plt.subplots(1, len(angles), figsize=(3.4 * len(angles), 3.6))
    for ax2, a in zip(np.atleast_1d(axes), angles):
        Xa = scatters[a]
        ax2.scatter(Xa[:, 0], Xa[:, 1], s=2, alpha=0.15, color="#7f8c8d")
        ax2.set_title(f"α = {a}°")
        ax2.set_aspect("equal")
    fig2.suptitle("Arc family (centred)")
    fig2.savefig(ctx.path("arc_sources.png"), bbox_inches="tight")

    run_dir = ctx.finish(K=K, n_angles=len(angles))
    print(f"\nResults saved to {run_dir}")


if __name__ == "__main__":
    main()
