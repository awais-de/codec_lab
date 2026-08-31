import numpy as np
import matplotlib.pyplot as plt

# ---- 1. The quantizer ----
def uniform_quantize(x, x_min, x_max, b):
    # compute N from b
    # compute delta from range and N
    # map each value in x to its cell's reconstruction level
    #   - figure out which cell each x lands in
    #   - convert that cell index to the reconstruction level
    #   - guard the top-boundary case (x == x_max)
    # return x_hat
    N = 2 ** b
    delta = (x_max - x_min) / N
    # Clip x to the range [x_min, x_max]
    x_clipped = np.clip(x, x_min, x_max)
    # Compute the cell index for each value in x_clipped
    cell_idx = np.floor((x_clipped - x_min) / delta).astype(int)
    # Guard the top-boundary case
    cell_idx = np.clip(cell_idx, 0, N - 1)
    # Compute the reconstruction level for each cell index
    x_hat = x_min + (cell_idx + 0.5) * delta
    return x_hat



if __name__ == "__main__":
    # ---- 2. Test signal ----
    # draw ~100_000 Gaussian samples, sigma = 0.3
    # clip to [x_min, x_max] so you stay in the granular regime
    x = np.random.normal(0, 0.3, 100000)
    x = np.clip(x, -1, 1)
        


    # ---- 3. Quantize at b = 6 ----
    # set x_min, x_max, b
    x_min, x_max = -1, 1
    b = 6
    # call your quantizer
    x_hat = uniform_quantize(x, x_min, x_max, b)
    # compute error e = x - x_hat
    e = x - x_hat
    # compute delta again here so you can draw the +/- delta/2 lines
    N = 2 ** b
    delta = (x_max - x_min) / N


    # ---- 4. Check the two predictions ----
    # (a) histogram of e  -> flat across [-delta/2, +delta/2]?
    plt.hist(e, bins=100)
    plt.title('Histogram of error for b=6')
    plt.axvline(x=-delta/2, color='red', linestyle='--')
    plt.axvline(x=delta/2, color='red', linestyle='--')
    plt.show()
    #     - mark x = -delta/2 and x = +delta/2 as vertical lines
    print(f"e.min() = {e.min()}, e.max() = {e.max()}")
    print(f"delta/2 = {delta/2}")
    # (b) print e.min() and e.max() -> both inside +/- delta/2?
    print(f"Are both e.min() and e.max() within [-delta/2, +delta/2]? {e.min() >= -delta/2 and e.max() <= delta/2}")


    # ---- 5. Break it at b = 1 ----

    # same signal, quantize at b = 1
    #Re assign the values just for b and compute N again
    b = 1
    N = 2 ** b
    #Recompute delta for b = 1
    delta = (x_max - x_min) / N
    x_hat = uniform_quantize(x, x_min, x_max, b)
    e = x - x_hat
    # histogram of that error -> should NOT be flat
    plt.hist(e, bins=100)
    plt.title('Histogram of error for b=1')
    #     - mark x = -delta/2 and x = +delta/2 as vertical lines
    plt.axvline(x=-delta/2, color='red', linestyle='--')    
    plt.show()
