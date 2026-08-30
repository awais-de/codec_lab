# Lloyd-Max non-uniform scalar quantization: why a fixed step size is
# wasteful on a non-uniform source, the two Lloyd-Max conditions
# (nearest-neighbour + centroid), the alternating-minimization algorithm
# that IS 1D k-means, the high-rate point-density result (~ f_X(x)^(1/3)),
# companding as a practical shortcut to non-uniform SQ, Zador/Bennett's
# asymptotic distortion formula, and a head-to-head RD curve against
# uniform SQ. Placeholder comment -- rewrite in your own words before
# calling this one done, per your own rule.

import numpy as np
from scipy import integrate, stats
import matplotlib.pyplot as plt


def lloyd_max_1d(samples, n_levels, max_iter=100, tol=1e-8, seed=0):
    """1D Lloyd-Max quantizer via alternating minimization (== 1D k-means).
    Returns (centroids, boundaries, distortion_history)."""
    rng = np.random.default_rng(seed)
    centroids = np.sort(rng.choice(samples, size=n_levels, replace=False))
    distortion_history = []

    for _ in range(max_iter):
        # nearest-neighbour condition: boundary = midpoint of adjacent centroids
        boundaries = (centroids[:-1] + centroids[1:]) / 2
        cell_idx = np.searchsorted(boundaries, samples)

        # centroid condition: each centroid = mean of samples assigned to it
        sums = np.bincount(cell_idx, weights=samples, minlength=n_levels)
        counts = np.bincount(cell_idx, minlength=n_levels)
        new_centroids = np.where(counts > 0, sums / np.maximum(counts, 1), centroids)
        new_centroids = np.sort(new_centroids)

        quantized = new_centroids[cell_idx]
        distortion_history.append(np.mean((samples - quantized) ** 2))

        if np.max(np.abs(new_centroids - centroids)) < tol:
            centroids = new_centroids
            break
        centroids = new_centroids

    boundaries = (centroids[:-1] + centroids[1:]) / 2
    return centroids, boundaries, distortion_history


def quantize_with_levels(samples, centroids, boundaries):
    cell_idx = np.searchsorted(boundaries, samples)
    return centroids[cell_idx]


def uniform_quantize(x, delta):
    return np.round(x / delta) * delta


def sqnr_db(signal_power, noise_power):
    return 10 * np.log10(signal_power / noise_power)


def mu_law_compress(x, mu=255, xmax=1.0):
    return xmax * np.sign(x) * np.log1p(mu * np.abs(x) / xmax) / np.log1p(mu)


def mu_law_expand(y, mu=255, xmax=1.0):
    return (xmax / mu) * np.sign(y) * np.expm1(np.abs(y) / xmax * np.log1p(mu))


if __name__ == "__main__":

    # ---- 1. Why uniform SQ is suboptimal for a non-uniform source ----
    print("=" * 60)
    print("UNIFORM SQ vs LLOYD-MAX ON A LAPLACIAN SOURCE")
    print("=" * 60)

    np.random.seed(0)
    laplace_samples = np.random.laplace(loc=0, scale=1.0, size=200_000)

    n_levels = 16
    x_range = (-8, 8)
    delta = (x_range[1] - x_range[0]) / n_levels
    clipped = np.clip(laplace_samples, *x_range)
    uniform_out = uniform_quantize(clipped, delta)
    uniform_mse = np.mean((clipped - uniform_out) ** 2)

    centroids, boundaries, hist = lloyd_max_1d(laplace_samples, n_levels)
    lloyd_out = quantize_with_levels(laplace_samples, centroids, boundaries)
    lloyd_mse = np.mean((laplace_samples - lloyd_out) ** 2)

    print(f"Uniform SQ ({n_levels} levels) MSE: {uniform_mse:.5f}")
    print(f"Lloyd-Max  ({n_levels} levels) MSE: {lloyd_mse:.5f}")
    print(f"Lloyd-Max reduces MSE by {(1 - lloyd_mse/uniform_mse)*100:.1f}%")

    fig1, ax1 = plt.subplots(figsize=(9, 5))
    ax1.hist(laplace_samples, bins=200, range=x_range, density=True,
              alpha=0.4, label='Laplacian source density')
    uniform_levels = np.arange(x_range[0] + delta/2, x_range[1], delta)
    ax1.scatter(uniform_levels, np.zeros_like(uniform_levels), marker='|',
                s=400, color='C1', label='Uniform SQ levels')
    ax1.scatter(centroids, np.full_like(centroids, -0.02), marker='|',
                s=400, color='C2', label='Lloyd-Max levels')
    ax1.set_xlabel('x')
    ax1.set_title('Uniform levels waste precision in the tails;\n'
                   'Lloyd-Max concentrates levels where density is high')
    ax1.legend()

    # ---- 2. Alternating minimization == 1D k-means ----
    print("\n" + "=" * 60)
    print("CONVERGENCE: ALTERNATING MINIMIZATION")
    print("=" * 60)
    for i, d in enumerate(hist):
        print(f"  iter {i:2d}: MSE = {d:.6f}")
    monotonic = all(hist[i+1] <= hist[i] + 1e-12 for i in range(len(hist) - 1))
    print(f"Monotonically non-increasing? {monotonic}")
    print("Assignment step = nearest-neighbour condition (like k-means'")
    print("'assign to nearest centroid'). Update step = centroid condition")
    print("(like k-means' 'recompute cluster mean'). Same algorithm, 1D.")

    # ---- 3. Verify both Lloyd-Max conditions directly ----
    print("\n" + "=" * 60)
    print("VERIFYING THE TWO CONDITIONS AT CONVERGENCE")
    print("=" * 60)
    recomputed_boundaries = (centroids[:-1] + centroids[1:]) / 2
    print(f"Nearest-neighbour condition holds? {np.allclose(boundaries, recomputed_boundaries)}")
    cell_idx = np.searchsorted(boundaries, laplace_samples)
    recomputed_centroids = np.array([
        laplace_samples[cell_idx == k].mean() for k in range(n_levels)
    ])
    print(f"Centroid condition holds? {np.allclose(centroids, recomputed_centroids, atol=1e-3)}")

    # ---- 4. High-rate result: point density ~ f_X(x)^(1/3) ----
    print("\n" + "=" * 60)
    print("HIGH-RATE POINT DENSITY vs f_X(x)^(1/3)")
    print("=" * 60)
    centroids_hr, _, _ = lloyd_max_1d(laplace_samples, 64)
    spacing = np.diff(centroids_hr)
    empirical_density = 1 / spacing
    midpoints = (centroids_hr[:-1] + centroids_hr[1:]) / 2

    f_pdf = stats.laplace(0, 1).pdf
    theoretical_density = f_pdf(midpoints) ** (1 / 3)

    empirical_density /= np.trapz(empirical_density, midpoints)
    theoretical_density /= np.trapz(theoretical_density, midpoints)

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    ax2.plot(midpoints, empirical_density, 'o-', label='Empirical (1/spacing), normalized')
    ax2.plot(midpoints, theoretical_density, '--', label='f_X(x)^(1/3), normalized')
    ax2.set_xlabel('x')
    ax2.set_ylabel('Normalized point density')
    ax2.set_title('High-rate centroid density follows f_X(x)^(1/3)')
    ax2.legend()

    # ---- 5. Companding: mu-law as a cheap shortcut to non-uniform SQ ----
    print("\n" + "=" * 60)
    print("COMPANDING (mu-LAW) vs PLAIN UNIFORM vs LLOYD-MAX")
    print("=" * 60)
    mu, xmax = 255, x_range[1]
    compressed = mu_law_compress(clipped, mu, xmax)
    compressed_q = uniform_quantize(compressed, 2 * xmax / n_levels)
    companded_out = mu_law_expand(compressed_q, mu, xmax)
    companded_mse = np.mean((clipped - companded_out) ** 2)

    print(f"Plain uniform SQ MSE:    {uniform_mse:.5f}")
    print(f"mu-law companded SQ MSE: {companded_mse:.5f}")
    print(f"Lloyd-Max (optimal) MSE: {lloyd_mse:.5f}")
    print("Companding needs no knowledge of the source density beyond a fixed")
    print("nonlinearity -- cheaper than solving for centroids, and lands")
    print("between plain uniform and true Lloyd-Max.")

    # ---- 6. Zador / Bennett's asymptotic distortion formula (scalar case) ----
    print("\n" + "=" * 60)
    print("ZADOR'S FORMULA (k=1 / BENNETT'S INTEGRAL)")
    print("=" * 60)
    integral, _ = integrate.quad(lambda x: f_pdf(x) ** (1 / 3), -50, 50)
    bennett_constant = integral ** 3
    print(f"(integral f_X(x)^(1/3) dx)^3 = {bennett_constant:.5f}")
    print("D(L) ~= bennett_constant / (12 * L^2)  as L -> infinity\n")
    for L in [4, 8, 16, 32, 64, 128]:
        c_L, b_L, _ = lloyd_max_1d(laplace_samples, L)
        q_L = quantize_with_levels(laplace_samples, c_L, b_L)
        empirical_mse = np.mean((laplace_samples - q_L) ** 2)
        bennett_pred = bennett_constant / (12 * L ** 2)
        print(f"  L={L:3d}: empirical MSE={empirical_mse:.6f}  "
              f"Zador/Bennett prediction={bennett_pred:.6f}  "
              f"ratio={empirical_mse/bennett_pred:.3f}")
    print("Ratio should drift toward 1.0 as L grows -- that's the asymptote.")

    # ---- 7. RD curve: uniform SQ vs Lloyd-Max, on a Gaussian source ----
    print("\n" + "=" * 60)
    print("RD CURVE: UNIFORM SQ vs LLOYD-MAX ON A GAUSSIAN SOURCE")
    print("=" * 60)
    np.random.seed(1)
    gaussian_samples = np.random.randn(200_000)
    gx_range = (-4, 4)
    signal_power = gaussian_samples.var()

    bits_list = np.arange(2, 9)
    uniform_sqnr, lloyd_sqnr = [], []
    for b in bits_list:
        L = 2 ** b
        d = (gx_range[1] - gx_range[0]) / L
        clipped_g = np.clip(gaussian_samples, *gx_range)
        uq = uniform_quantize(clipped_g, d)
        uniform_sqnr.append(sqnr_db(signal_power, np.mean((clipped_g - uq) ** 2)))

        c, bnd, _ = lloyd_max_1d(gaussian_samples, L, max_iter=50)
        lq = quantize_with_levels(gaussian_samples, c, bnd)
        lloyd_sqnr.append(sqnr_db(signal_power, np.mean((gaussian_samples - lq) ** 2)))

    for b, u, l in zip(bits_list, uniform_sqnr, lloyd_sqnr):
        print(f"  {b} bits -> uniform SQNR={u:6.2f} dB   Lloyd-Max SQNR={l:6.2f} dB   "
              f"gain={l-u:+.2f} dB")

    fig3, ax3 = plt.subplots(figsize=(9, 5))
    ax3.plot(bits_list, uniform_sqnr, 'o-', label='Uniform SQ')
    ax3.plot(bits_list, lloyd_sqnr, 's-', label='Lloyd-Max')
    ax3.set_xlabel('Bits per sample')
    ax3.set_ylabel('SQNR (dB)')
    ax3.set_title('RD curve: uniform SQ vs Lloyd-Max on a Gaussian source')
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.show()
