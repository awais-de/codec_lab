import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
import os


# ============================================================
#  Scalar quantizers
# ============================================================
def uniform_quantize(x, x_min, x_max, b):
    """Mid-rise uniform scalar quantizer. Returns (x_hat, delta)."""
    N = 2 ** b
    delta = (x_max - x_min) / N
    x_clipped = np.clip(x, x_min, x_max)
    cell_idx = np.floor((x_clipped - x_min) / delta).astype(int)
    cell_idx = np.clip(cell_idx, 0, N - 1)
    x_hat = x_min + (cell_idx + 0.5) * delta
    return x_hat, delta


def lloyd_max_1d(x, N, n_iter=100, tol=1e-10):
    """Optimal scalar quantizer (Lloyd-Max) on samples x. Returns (levels, boundaries)."""
    levels = np.linspace(x.min(), x.max(), N)
    prev = None
    for _ in range(n_iter):
        boundaries = np.concatenate([[-np.inf], (levels[1:] + levels[:-1]) / 2, [np.inf]])
        idx = np.digitize(x, boundaries) - 1
        levels = np.array([x[idx == k].mean() if np.any(idx == k) else levels[k]
                           for k in range(N)])
        if prev is not None and np.allclose(levels, prev, atol=tol):
            break
        prev = levels.copy()
    boundaries = np.concatenate([[-np.inf], (levels[1:] + levels[:-1]) / 2, [np.inf]])
    return levels, boundaries


def lm_apply(x, levels, boundaries):
    return levels[np.digitize(x, boundaries) - 1]


# ============================================================
#  Vector quantizer (LBG = k-means + splitting)
# ============================================================
def kmeans_refine(X, codewords, n_iter=100, tol=1e-11):
    """Assign -> centroid -> repeat. Returns (codewords, dist_history)."""
    dist_history = []
    for _ in range(n_iter):
        # ASSIGN: each x to nearest codeword (Euclidean, L-D)
        assign = np.argmin(np.linalg.norm(X[:, None, :] - codewords[None, :, :], axis=2), axis=1)
        # UPDATE: codeword = centroid of its cluster (empty-cell guard: keep old)
        new_cw = np.array([X[assign == k].mean(axis=0) if np.any(assign == k) else codewords[k]
                           for k in range(len(codewords))])
        # DISTORTION: per-sample MSE (÷ L makes it comparable to scalar)
        D = np.mean(np.sum((X - new_cw[assign]) ** 2, axis=1)) / X.shape[1]
        dist_history.append(D)
        if len(dist_history) > 1 and abs(dist_history[-2] - dist_history[-1]) < tol:
            codewords = new_cw
            break
        codewords = new_cw
    return codewords, dist_history


def lbg(X, K_target, eps=1e-2):
    """Grow codebook 1 -> 2 -> ... -> K_target by ADDITIVE splitting + refinement."""
    codewords = X.mean(axis=0)[None, :]          # 1 codeword = centroid of all data
    while len(codewords) < K_target:
        # ADDITIVE split (works at the origin, unlike multiplicative)
        codewords = np.concatenate([codewords + eps, codewords - eps], axis=0)
        codewords, _ = kmeans_refine(X, codewords)
    return codewords


def vq_quantize(X, codewords):
    assign = np.argmin(np.linalg.norm(X[:, None, :] - codewords[None, :, :], axis=2), axis=1)
    return codewords[assign]


# ============================================================
#  Metric
# ============================================================
def sqnr_vec(X, x_hat):
    """Per-sample SQNR in dB for L-dim blocks."""
    sig = np.var(X)
    noise = np.mean(np.sum((X - x_hat) ** 2, axis=1)) / X.shape[1]
    return 10 * np.log10(sig / noise)


# ============================================================
#  Experiment: VQ vs optimal scalar, at two correlations
# ============================================================
def run(rho, b=2, L=2, n=4000, seed=0):
    np.random.seed(seed)
    K = 2 ** (b * L)                                  # rate-matched codebook size
    X = np.random.multivariate_normal([0, 0], [[1, rho], [rho, 1]], n)

    # --- VQ ---
    cw = lbg(X, K)
    x_hat_vq = vq_quantize(X, cw)
    s_vq = sqnr_vec(X, x_hat_vq)

    # --- Fair baseline: OPTIMAL SCALAR (Lloyd-Max) per axis, b bits each ---
    x_hat_sq = np.zeros_like(X)
    for j in range(L):
        lv, bnd = lloyd_max_1d(X[:, j], 2 ** b)
        x_hat_sq[:, j] = lm_apply(X[:, j], lv, bnd)
    s_sq = sqnr_vec(X, x_hat_sq)

    return X, cw, x_hat_sq, s_sq, s_vq, K


def plot_pair(ax_sq, ax_vq, X, cw, s_sq, s_vq, rho):
    lim = 4
    ax_sq.scatter(X[:, 0], X[:, 1], s=4, alpha=0.15, color="#333")
    lv0, bnd0 = lloyd_max_1d(X[:, 0], int(np.sqrt(len(cw))))
    lv1, bnd1 = lloyd_max_1d(X[:, 1], int(np.sqrt(len(cw))))
    for bx in bnd0[1:-1]:
        ax_sq.axvline(bx, color="#c0392b", lw=0.7)
    for by in bnd1[1:-1]:
        ax_sq.axhline(by, color="#c0392b", lw=0.7)
    gx, gy = np.meshgrid(lv0, lv1)
    ax_sq.scatter(gx.ravel(), gy.ravel(), color="#c0392b", s=45, marker='s',
                  edgecolor='white', zorder=5)
    ax_sq.set_title(f"Optimal SCALAR — {s_sq:.1f} dB", fontsize=11)

    ax_vq.scatter(X[:, 0], X[:, 1], s=4, alpha=0.15, color="#333")
    if len(cw) >= 4:
        vor = Voronoi(cw)
        voronoi_plot_2d(vor, ax=ax_vq, show_vertices=False, show_points=False,
                        line_colors="#2471a3", line_width=0.9)
    ax_vq.scatter(cw[:, 0], cw[:, 1], color="#2471a3", s=45, marker='o',
                  edgecolor='white', zorder=5)
    ax_vq.set_title(f"VECTOR — {s_vq:.1f} dB", fontsize=11)

    for ax in (ax_sq, ax_vq):
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")


if __name__ == "__main__":
    fig, axes = plt.subplots(2, 2, figsize=(11, 11))

    print(f"{'rho':>6} | {'scalar':>8} | {'VQ':>8} | {'gain':>7}")
    print("-" * 40)
    for row, rho in enumerate([0.85, 0.0]):
        X, cw, x_hat_sq, s_sq, s_vq, K = run(rho)
        gain = s_vq - s_sq
        print(f"{rho:>6} | {s_sq:>7.2f} | {s_vq:>7.2f} | {gain:>+6.2f}")
        plot_pair(axes[row, 0], axes[row, 1], X, cw, s_sq, s_vq, rho)
        axes[row, 0].set_ylabel(f"rho = {rho}\n\n$x_2$", fontsize=10)

    print("\nrho=0.85 gain = memory + space-filling")
    print("rho=0.00 gain = space-filling ONLY (bounded, ~fraction of a dB at L=2)")

    fig.suptitle("VQ vs Optimal Scalar — same bits/sample\n"
                 "top: correlated (big gain)   bottom: uncorrelated (space-filling only)",
                 fontsize=12)
    plt.tight_layout()
    here = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(here, "vq_comparison.png"), dpi=95)
    plt.show()