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
    # Example usage
    probs = np.array([0.5, 0.5])  # Binary random variable
    print(f"Shannon Entropy H(X) = {shannon_entropy(probs)} bits")

    joint_probs = np.array([[0.25, 0.25], [0.25, 0.25]])  # Joint distribution of two binary variables
    print(f"Joint Entropy H(X,Y) = {joint_entropy(joint_probs)} bits")