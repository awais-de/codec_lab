# EXP-03: Decorrelating transform (PCA) + SQ vs VQ

Tracks **[#3](https://github.com/awais-de/codec_lab/issues/3)** · Stage: Reference · Rung 3 of the classical SQ-vs-VQ ladder

## Question
EXP-02 showed VQ's advantage over SQ is memory gain — exploiting correlation. If
we remove the correlation *before* quantizing, with a fixed linear transform,
does VQ's advantage collapse to the EXP-01 granular floor?

## Hypothesis
Two steps are needed for PCA to help a scalar quantizer:
1. **rotate** onto the principal axes (decorrelate)
2. **re-allocate bits** to the now-unequal axis variances (`1±ρ`)

Rotation *alone* does nothing: Lloyd–Max distortion is linear in variance, so an
orthonormal rotation conserves total distortion at equal rate. The gain is
entirely in the re-allocation. With both steps, per-axis SQ should match VQ and
the VQ gain should collapse to ~0.4 dB (the EXP-01 floor) — wherever the
integer bit budget actually lets the allocation change.

PCA is the simplest possible "encoder" — a fixed rotation handed the covariance.
If it captures the memory gain, the question for a neural codec becomes *"does
the encoder decorrelate?"* rather than *"SQ vs VQ?"*.

## Setup
- Source: 2D Gaussian, ρ swept over `config.yaml` (the EXP-02 grid)
- SQ: per-axis Lloyd–Max; bits allocated by greedy water-filling (`codeclab.rd.water_filling_bits`)
- VQ: LBG, `K = 2**(2·bits_per_dim) = 16`
- Bitrate: 2 bits/dim = 4 bits/vector, split across the two axes
- Conditions: `{no transform, PCA+equal bits, PCA+allocation} × {SQ, VQ}`

## Hand-check anchors
- PCA eigenvectors = ±45° rotation (exact for equicorrelation); eigenvalues `1±ρ`
- greedy allocation gives `(2,2)` for ρ ≲ 0.5, `(3,1)` for ρ ≳ 0.6 (continuous optimum
  `Δb = ½·log₂((1+ρ)/(1−ρ))`, ≈ 1.8 bits at ρ=0.85)
- `SQ, PCA + equal bits` should sit exactly on `SQ, no transform`
- recovered memory gain = `gain_plain − gain_pca` should track `−5·log₁₀(1−ρ²)`
- cross-check: `SQNR(SQ, PCA + allocation)` ≈ `SQNR(VQ, no transform)`

## Run
```bash
pip install -e .
python experiments/exp_03_pca_decorrelation/run.py
```
Output → `results/exp_03_pca_decorrelation/results_<timestamp>/`:
`vq_gain_pca_vs_plain.png`, `sq_after_pca_vs_vq.png`, `metrics.csv`,
`config.snapshot.yaml`, `meta.json`.

## Definition of done
- [x] `codeclab/transforms.py` (pca/klt) + `rd.water_filling_bits`
- [x] `run.py` implemented
- [ ] plots + `metrics.csv` generated and committed
- [ ] one-paragraph interpretation added below
- [ ] residual-after-PCA VQ gain recorded for EXP-06

## Result
_(fill in after the run: does gain(PCA) collapse to the floor, where does it peel
off from gain(plain), does SQ+PCA reach VQ-no-transform)_
