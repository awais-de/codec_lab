import numpy as np
import matplotlib.pyplot as plt


# ---- Quantizer ----
def uniform_quantize(x, x_min, x_max, b):
    """Mid-rise uniform scalar quantizer. Returns (x_hat, delta)."""
    N = 2 ** b
    delta = (x_max - x_min) / N
    x_clipped = np.clip(x, x_min, x_max)
    cell_idx = np.floor((x_clipped - x_min) / delta).astype(int)  # encode (lossy)
    cell_idx = np.clip(cell_idx, 0, N - 1)                        # top-boundary guard
    x_hat = x_min + (cell_idx + 0.5) * delta                      # decode (lookup)
    return x_hat, delta


# ---- Per-b analysis: quantize, measure, report ----
def analyze(x, x_min, x_max, b):
    """Quantize at b bits, print error stats vs theory, return (e, delta)."""
    x_hat, delta = uniform_quantize(x, x_min, x_max, b)
    e = x - x_hat

    var_measured = np.var(e)
    var_theory = delta ** 2 / 12          # holds only under high-resolution
    half = delta / 2

    print(f"[b={b}]  delta={delta:.5f}")
    print(f"  var(e) measured = {var_measured:.6e}")
    print(f"  delta^2/12      = {var_theory:.6e}")
    print(f"  e in [{e.min():.5f}, {e.max():.5f}]  |  bound +/- {half:.5f}"
          f"  |  within? {e.min() >= -half and e.max() <= half}")
    return e, delta


# ---- Error histogram with +/- delta/2 markers ----
def plot_error_hist(e, delta, b, bins=100):
    plt.figure()
    plt.hist(e, bins=bins)
    plt.title(f"Histogram of error for b={b}")
    plt.axvline(-delta / 2, color="red", linestyle="--")
    plt.axvline(+delta / 2, color="red", linestyle="--")


def sqnr(x, x_hat):
    """Signal power / quantization noise power, in dB."""
    signal_power = np.var(x)
    noise_power = np.var(x - x_hat)
    return 10 * np.log10(signal_power / noise_power)

if __name__ == "__main__":
    np.random.seed(0)

    X_MIN, X_MAX = -1, 1

    x = np.random.normal(0, 0.3, 100_000)
    x = np.clip(x, X_MIN, X_MAX)

    e6, d6 = analyze(x, X_MIN, X_MAX, b=6)
    plot_error_hist(e6, d6, b=6)

    e1, d1 = analyze(x, X_MIN, X_MAX, b=1)
    plot_error_hist(e1, d1, b=1)

    bits = np.arange(1, 9)
    sqnr_values = []

    for b in bits:
        x_hat, _ = uniform_quantize(x, X_MIN, X_MAX, b)
        sqnr_values.append(sqnr(x, x_hat))

    plt.figure()
    plt.plot(bits, sqnr_values, marker='o', label='Measured SQNR')
    plt.plot(bits, 6.02 * bits + 1.76, linestyle='--', label='Theoretical ~6dB/bit')
    plt.xlabel('Bitrate (bits/sample)')
    plt.ylabel('SQNR (dB)')
    plt.title('Rate-Distortion Curve — Uniform Scalar Quantizer')
    plt.legend()
    plt.grid()

    plt.show()  # single call at the end — shows all figures together