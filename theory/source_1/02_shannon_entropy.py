import numpy as np

def information_content(p):
    # h(x) = log2(1/p)
    # handle zeros: 0 * log(0) = 0 by convention
    return np.where(p > 0, np.log2(1/p), 0)

def shannon_entropy(probs):
    # H(X) = sum of p(x) * log2(1/p(x)) for all outcomes
    # probs is a numpy array of probabilities that sum to 1
    return np.sum(probs * information_content(probs))

def joint_entropy(joint_probs):
    # H(X,Y) = sum over all pairs of p(x,y) * log2(1/p(x,y))
    # joint_probs is a 2D numpy array
    return np.sum(joint_probs * information_content(joint_probs))

def kl_divergence(p, q):
    # D_KL(P||Q) = sum of p * log2(p/q)
    # use np.where to handle zeros
    return np.sum(np.where((p > 0) & (q > 0), p * np.log2(p / q), 0))


if __name__ == "__main__":

    # --- Shannon Entropy ---
    print("=" * 50)
    print("SHANNON ENTROPY")
    print("=" * 50)

    # Uniform distribution — should equal log2(N)
    uniform = np.array([0.25, 0.25, 0.25, 0.25])
    print(f"Uniform over 4 outcomes: H(X) = {shannon_entropy(uniform):.4f} bits")
    print(f"Expected (log2(4)):               {np.log2(4):.4f} bits")

    # Certain outcome — should equal 0
    certain = np.array([1.0, 0.0, 0.0, 0.0])
    print(f"Certain outcome:          H(X) = {shannon_entropy(certain):.4f} bits")

    # Biased coin
    biased = np.array([0.9, 0.1])
    print(f"Biased coin (0.9/0.1):    H(X) = {shannon_entropy(biased):.4f} bits")

    # --- Joint Entropy ---
    print("\n" + "=" * 50)
    print("JOINT ENTROPY")
    print("=" * 50)

    # Independent variables — H(X,Y) should equal H(X) + H(Y)
    independent = np.array([[0.25, 0.25], [0.25, 0.25]])
    hx = shannon_entropy(independent.sum(axis=1))
    hy = shannon_entropy(independent.sum(axis=0))
    hxy = joint_entropy(independent)
    print(f"Independent variables:")
    print(f"  H(X)          = {hx:.4f} bits")
    print(f"  H(Y)          = {hy:.4f} bits")
    print(f"  H(X) + H(Y)   = {hx + hy:.4f} bits")
    print(f"  H(X,Y)        = {hxy:.4f} bits")
    print(f"  Equal? {np.isclose(hxy, hx + hy)}")

    # Correlated variables — H(X,Y) should be less than H(X) + H(Y)
    correlated = np.array([[0.4, 0.1], [0.1, 0.4]])
    hx_c = shannon_entropy(correlated.sum(axis=1))
    hy_c = shannon_entropy(correlated.sum(axis=0))
    hxy_c = joint_entropy(correlated)
    print(f"\nCorrelated variables:")
    print(f"  H(X) + H(Y)   = {hx_c + hy_c:.4f} bits")
    print(f"  H(X,Y)        = {hxy_c:.4f} bits")
    print(f"  H(X,Y) < H(X)+H(Y)? {hxy_c < hx_c + hy_c}")

    # --- KL Divergence ---
    print("\n" + "=" * 50)
    print("KL DIVERGENCE")
    print("=" * 50)

    # Property 1: KL >= 0
    p = np.array([0.9, 0.1])
    q = np.array([0.5, 0.5])
    print(f"P = {p}, Q = {q}")
    print(f"D_KL(P||Q) = {kl_divergence(p, q):.4f} bits  (>= 0? {kl_divergence(p, q) >= 0})")

    # Property 2: Asymmetry
    print(f"\nAsymmetry demonstration:")
    print(f"D_KL(P||Q) = {kl_divergence(p, q):.4f} bits")
    print(f"D_KL(Q||P) = {kl_divergence(q, p):.4f} bits")
    print(f"Symmetric? {np.isclose(kl_divergence(p, q), kl_divergence(q, p))}")

    # Property 3: KL = 0 when P = Q
    print(f"\nWhen P = Q:")
    print(f"D_KL(P||P) = {kl_divergence(p, p):.4f} bits  (should be 0)")

    # Thesis connection
    print("\n" + "=" * 50)
    print("THESIS CONNECTION — D-VAE KL collapse")
    print("=" * 50)
    latent_normal = np.array([0.6, 0.3, 0.1])   # peaked — high entropy latent
    prior_normal  = np.array([0.33, 0.33, 0.34]) # near-uniform prior
    print(f"Latent distribution (peaked):  H = {shannon_entropy(latent_normal):.4f} bits")
    print(f"Prior distribution (uniform):  H = {shannon_entropy(prior_normal):.4f} bits")
    print(f"D_KL(latent||prior) = {kl_divergence(latent_normal, prior_normal):.4f} bits")
    print(f"Minimizing this KL forces latent toward prior — entropy collapses")