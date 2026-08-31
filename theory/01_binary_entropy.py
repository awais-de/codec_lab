import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt

def binary_entropy(x):
    # computes H₂(x) = x·log2(1/x) + (1-x)·log2(1/(1-x))
    return x * np.log2(1/x) + (1-x) * np.log2(1/(1-x))

def log_combinations(N, r):
    # computes log C(N,r) directly using scipy.special.comb
    return np.log2(float(comb(N, r, exact=True)))

def n_times_entropy(N, r):
    # computes N · H₂(r/N)
    return N * binary_entropy(r / N)

if __name__ == "__main__":
    N = 1000
    r = 500
    print(f"Binary entropy H₂({r}/{N}) = {binary_entropy(r/N)}")
    print(f"log C({N},{r}) = {log_combinations(N, r)}")
    print(f"N * H₂({r}/{N}) = {n_times_entropy(N, r)}")    

    nums = np.linspace(0.001, 0.999, 1000)

    entropies = binary_entropy(nums)

    plt.plot(nums, entropies)
    plt.title("Binary Entropy Function H₂(x)")
    plt.xlabel("x")
    plt.ylabel("H₂(x)")
    plt.grid()
    plt.show()  
