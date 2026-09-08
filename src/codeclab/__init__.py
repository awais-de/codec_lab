"""codeclab — shared code for the codec-lab quantization experiment suite.

Modules
-------
sources      synthetic signal generators (Gaussian, correlated Gaussian, colored noise, Laplacian)
quant.scalar uniform + Lloyd-Max scalar quantizers
quant.vector LBG vector quantizer
transforms   PCA / KLT decorrelating rotation (classical stand-in for an encoder)
metrics.toy  MSE / SNR / SQNR
rd           rate-distortion helpers, VQ-gain-in-dB, bit allocation
plotting     shared matplotlib house style
runctx       config loading + timestamped result directories

`theory/` is a frozen learning log and is never imported here.
"""

__version__ = "0.1.0"
