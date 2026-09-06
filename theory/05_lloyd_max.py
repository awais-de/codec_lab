import numpy as np
import matplotlib.pyplot as plt


# ---- Uniform scalar quantizer (validated earlier) ----
def uniform_quantize(x, x_min, x_max, b):
    """Mid-rise uniform scalar quantizer. Returns (x_hat, delta)."""
    N = 2 ** b
    delta = (x_max - x_min) / N
    x_clipped = np.clip(x, x_min, x_max)
    cell_idx = np.floor((x_clipped - x_min) / delta).astype(int)
    cell_idx = np.clip(cell_idx, 0, N - 1)
    x_hat = x_min + (cell_idx + 0.5) * delta
    return x_hat, delta


# ---- Lloyd-Max quantizer (trained on samples) ----
def lloyd_max(x, N, n_iter=50, tol=1e-7):
    """
    Train an N-level Lloyd-Max quantizer on samples x.
    Returns (levels, boundaries, distortion_history).
    """
    # 1. INITIALISE levels (uniform spread over the data range)
    levels = np.linspace(x.min(), x.max(), N)

    dist_history = []

    for _ in range(n_iter):
        # 2. BOUNDARY UPDATE (E1.2): interior boundaries = midpoints of adjacent levels
        boundaries = np.concatenate([[-np.inf], (levels[1:] + levels[:-1]) / 2, [np.inf]])

        # 3. ASSIGN each sample to a cell
        cell_idx = np.digitize(x, boundaries) - 1

        # 4. LEVEL UPDATE (E1.3): each level = mean of samples in that cell (centroid)
        #    empty-cell guard: keep the old level if a cell caught no samples
        new_levels = []
        for i in range(N):
            if np.sum(cell_idx == i) == 0:
                new_levels.append(levels[i])
            else:
                new_levels.append(np.mean(x[cell_idx == i]))
        new_levels = np.array(new_levels)

        # 5. DISTORTION for this iteration (MSE)
        D = np.mean((x - new_levels[cell_idx]) ** 2)
        dist_history.append(D)

        # 6. CONVERGENCE CHECK
        if len(dist_history) > 1 and np.abs(dist_history[-1] - dist_history[-2]) < tol:
            levels = new_levels
            break
        levels = new_levels

    return levels, boundaries, dist_history


# ---- Apply a trained quantizer to samples ----
def apply_quantizer(x, levels, boundaries):
    """Map each x to its cell's level via the boundaries. Returns x_hat."""
    cell_idx = np.digitize(x, boundaries) - 1
    return levels[cell_idx]


def sqnr(x, x_hat):
    """Signal power / quantization noise power, in dB."""
    return 10 * np.log10(np.var(x) / np.var(x - x_hat))


if __name__ == "__main__":
    np.random.seed(0)
    x = np.random.normal(0, 0.3, 100_000)
    x = np.clip(x, -1, 1)

    b = 3                      # compare at a fixed bitrate
    N = 2 ** b

    # --- Lloyd-Max ---
    levels, boundaries, hist = lloyd_max(x, N)
    x_hat_lm = apply_quantizer(x, levels, boundaries)
    sqnr_lm = sqnr(x, x_hat_lm)

    # --- Uniform baseline: SAME N, SAME range (apples-to-apples) ---
    x_hat_uni, _ = uniform_quantize(x, -1, 1, b)
    sqnr_uni = sqnr(x, x_hat_uni)

    # --- Report ---
    print(f"N = {N} levels  (b = {b} bits)")
    print(f"Lloyd-Max SQNR : {sqnr_lm:.2f} dB")
    print(f"Uniform   SQNR : {sqnr_uni:.2f} dB")
    print(f"Gain (LM - uni): {sqnr_lm - sqnr_uni:+.2f} dB")

    # monotonicity check on distortion history
    diffs = np.diff(hist)
    monotone = np.all(diffs <= 1e-12)
    print(f"\ndist_history monotonically non-increasing? {monotone}")
    print(f"  iterations to converge: {len(hist)}")
    print(f"  D start -> end: {hist[0]:.6e} -> {hist[-1]:.6e}")

    # show where the levels ended up
    print(f"\nLloyd-Max levels : {np.round(np.sort(levels), 3)}")
    uni_levels = np.array([-1 + (k + 0.5) * (2 / N) for k in range(N)])
    print(f"Uniform   levels : {np.round(uni_levels, 3)}")

    # --- Sanity plots ---
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(hist, marker='o', ms=3)
    plt.title("Lloyd-Max Distortion History (should decrease monotonically)")
    plt.xlabel("Iteration")
    plt.ylabel("Distortion (MSE)")
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.hist(x, bins=100, alpha=0.4, label="Signal")
    for L in levels:
        plt.axvline(L, color='C1', linestyle='-', alpha=0.8)
    for L in uni_levels:
        plt.axvline(L, color='C2', linestyle='--', alpha=0.8)
    plt.title("Level placement: LM (solid) clusters near 0, uniform (dashed) evenly spread")
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.legend()

    plt.tight_layout()
    plt.show()