import numpy as np
import matplotlib.pyplot as plt


def uniform_quantize(x, delta):
    """Map continuous x to the nearest multiple of delta (mid-rise grid)."""
    return np.round(x / delta) * delta


def sqnr_db(signal_power, noise_power):
    """Signal-to-Quantization-Noise Ratio, in dB."""
    return 10 * np.log10(signal_power / noise_power)


def theoretical_noise_power(delta):
    """Quantization noise power under the uniform-error assumption:
    Var(U[-delta/2, delta/2]) = delta^2 / 12."""
    return delta ** 2 / 12


if __name__ == "__main__":

    # ---- 1. The mapping: continuous input -> discrete staircase ----
    print("=" * 60)
    print("UNIFORM QUANTIZATION: THE STAIRCASE MAPPING")
    print("=" * 60)

    delta = 0.5
    x = np.linspace(-3, 3, 1000)
    x_q = uniform_quantize(x, delta)

    print(f"Step size Delta = {delta}")
    print(f"Sample mapping: x=0.37 -> {uniform_quantize(np.array([0.37]), delta)[0]}")
    print(f"Sample mapping: x=1.24 -> {uniform_quantize(np.array([1.24]), delta)[0]}")

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(x, x, '--', color='gray', label='Identity (no quantization)')
    ax1.plot(x, x_q, color='C0', label=f'Quantized (Delta={delta})')
    ax1.set_xlabel('Input x')
    ax1.set_ylabel('Quantized output')
    ax1.set_title('Uniform Scalar Quantizer: the staircase')
    ax1.legend()
    ax1.grid(True)

    # ---- 2. Quantization error distribution ----
    print("\n" + "=" * 60)
    print("QUANTIZATION ERROR")
    print("=" * 60)

    error = x - x_q
    print(f"Error range: [{error.min():.4f}, {error.max():.4f}]  "
          f"(expect ~[-Delta/2, Delta/2] = [{-delta/2}, {delta/2}])")
    print(f"Empirical error variance: {error.var():.6f}")
    print(f"Theoretical Delta^2/12:    {theoretical_noise_power(delta):.6f}")

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.hist(error, bins=50, density=True, alpha=0.7, label='Empirical error')
    ax2.axhline(1 / delta, color='r', linestyle='--',
                label=f'Uniform density 1/Delta = {1/delta:.2f}')
    ax2.set_xlabel('Quantization error (x - x_q)')
    ax2.set_ylabel('Density')
    ax2.set_title('Quantization error looks uniform on [-Delta/2, Delta/2]')
    ax2.legend()

    # ---- 3. SQNR vs bitrate -- the ~6 dB/bit rule, on a Gaussian source ----
    print("\n" + "=" * 60)
    print("SQNR vs BITRATE -- Gaussian source")
    print("=" * 60)

    np.random.seed(0)
    signal = np.random.randn(200_000)   # unit-variance Gaussian source
    signal_power = signal.var()
    x_range = (-4, 4)                    # clip range: ~4 sigma covers it

    bits_list = np.arange(2, 13)
    sqnr_list = []
    for b in bits_list:
        n_levels = 2 ** b
        d = (x_range[1] - x_range[0]) / n_levels
        clipped = np.clip(signal, x_range[0], x_range[1])
        q = uniform_quantize(clipped, d)
        noise_power = np.mean((clipped - q) ** 2)
        sqnr_list.append(sqnr_db(signal_power, noise_power))

    for b, s in zip(bits_list, sqnr_list):
        print(f"  {b:2d} bits -> SQNR = {s:6.2f} dB")

    # fit the slope only over the higher-bit range, where the
    # high-resolution assumption actually holds (see section 4)
    high_res_mask = bits_list >= 6
    slope = np.polyfit(bits_list[high_res_mask], np.array(sqnr_list)[high_res_mask], 1)[0]
    print(f"\nFitted slope (bits >= 6): {slope:.2f} dB/bit  (theory: ~6.02 dB/bit)")

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.plot(bits_list, sqnr_list, 'o-', label='Measured SQNR')
    ax3.set_xlabel('Bits per sample')
    ax3.set_ylabel('SQNR (dB)')
    ax3.set_title('Rate-Distortion curve: uniform quantizer on Gaussian source')
    ax3.grid(True)
    ax3.legend()

    # ---- 4. High-resolution assumption: where it holds, where it breaks ----
    print("\n" + "=" * 60)
    print("HIGH-RESOLUTION ASSUMPTION: LOW BITS BREAKS IT")
    print("=" * 60)
    for b in [1, 2, 4, 8]:
        n_levels = 2 ** b
        d = (x_range[1] - x_range[0]) / n_levels
        clipped = np.clip(signal, x_range[0], x_range[1])
        q = uniform_quantize(clipped, d)
        err = clipped - q
        empirical_var = err.var()
        theoretical_var = theoretical_noise_power(d)
        overload_fraction = np.mean(np.abs(signal) > x_range[1])
        print(f"  {b:2d} bits (Delta={d:.4f}): empirical var={empirical_var:.5f}, "
              f"theoretical Delta^2/12={theoretical_var:.5f}, "
              f"ratio={empirical_var/theoretical_var:.3f}, "
              f"overloaded={overload_fraction*100:.2f}%")

    plt.tight_layout()
    plt.show()
