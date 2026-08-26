import numpy as np
import matplotlib.pyplot as plt


def shannon_entropy(probs):
    # Input: probability distribution
    # Output: entropy H(X) in bits
    probs = np.asarray(probs, dtype=float)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def generate_source(probs, n_symbols):
    # Input: probability distribution, number of symbols to generate
    # Output: numpy array of symbols drawn from that distribution
    return np.random.choice(len(probs), size=n_symbols, p=probs)


def naive_encode(symbols, n_outcomes):
    # Input: symbols array, number of possible outcomes
    # Output: bits used — fixed length = log2(n_outcomes) per symbol
    bits_per_symbol = np.ceil(np.log2(n_outcomes))
    total_bits = len(symbols) * bits_per_symbol
    return total_bits


def entropy_encode(symbols, probs):
    # Input: symbols array, probability distribution
    # Output: theoretical minimum bits = H(X) per symbol
    h_x = shannon_entropy(probs)
    total_bits = len(symbols) * h_x
    return total_bits


def essential_bit_content(probs, delta):
    # Input: probability distribution, acceptable failure probability delta
    # Output: Hdelta(X) = log2(|Sdelta|) — MacKay eq 4.19
    # Size of the smallest subset covering (1-delta) probability mass
    sorted_probs = np.sort(probs)[::-1]
    cumulative_probs = np.cumsum(sorted_probs)
    index = np.searchsorted(cumulative_probs, 1 - delta)
    subset_size = index + 1
    return np.log2(subset_size)


if __name__ == "__main__":

    probs = np.array([0.7, 0.2, 0.1])
    n_symbols = 1000

    symbols = generate_source(probs, n_symbols)

    naive_bits = naive_encode(symbols, len(probs))
    entropy_bits = entropy_encode(symbols, probs)
    h_x = shannon_entropy(probs)

    print("=" * 50)
    print("SOURCE CODING THEOREM")
    print("=" * 50)
    print(f"Naive bits per symbol:   {naive_bits / n_symbols:.4f}")
    print(f"Entropy bound H(X):      {entropy_bits / n_symbols:.4f}")
    print(f"Redundancy (wasted):     {(naive_bits - entropy_bits) / n_symbols:.4f} bits/symbol")
    print(f"Cannot compress below H(X) = {h_x:.4f} bits without losing information")

    print("\n" + "=" * 50)
    print("ESSENTIAL BIT CONTENT vs DELTA")
    print("=" * 50)
    for delta in [0, 0.01, 0.05, 0.1, 0.2]:
        h_delta = essential_bit_content(probs, delta)
        print(f"delta={delta:.2f} → Hdelta(X) = {h_delta:.4f} bits")

    print("\n" + "=" * 50)
    print("EFFECT OF DISTRIBUTION SKEWNESS")
    print("=" * 50)
    uniform = np.array([0.33, 0.33, 0.34])
    skewed = np.array([0.9, 0.09, 0.01])
    print(f"Uniform distribution:  H(X) = {shannon_entropy(uniform):.4f} bits")
    print(f"Skewed distribution:   H(X) = {shannon_entropy(skewed):.4f} bits")
    print(f"More skewed = lower entropy = less bits needed to describe source")

    # Plot: Hdelta(X) vs delta
    deltas = np.linspace(0, 0.5, 100)
    h_deltas = [essential_bit_content(probs, d) for d in deltas]

    plt.figure(figsize=(8, 5))
    plt.plot(deltas, h_deltas, label='Hdelta(X) — essential bit content')
    plt.axhline(y=h_x, color='r', linestyle='--', label=f'H(X) = {h_x:.4f} bits — entropy bound')
    plt.axhline(y=np.ceil(np.log2(len(probs))), color='g', linestyle='--',
                label=f'Naive = {np.ceil(np.log2(len(probs))):.0f} bits')
    plt.xlabel('delta (acceptable failure probability)')
    plt.ylabel('Hdelta(X) bits')
    plt.title('Essential bit content vs acceptable risk\n'
              'Higher delta = willing to lose more = fewer bits needed')
    plt.legend()
    plt.grid()
    plt.show()