# EXP-02: VQ gain vs source correlation (ρ sweep)

Tracks **[#1](https://github.com/awais-de/codec_lab/issues/1)** · Stage: Reference · Rung 2 of the classical SQ-vs-VQ ladder

## Question
Once the memoryless granular floor is known (EXP-01), how much *extra* does VQ
gain as the source becomes correlated — and does it follow the classical
memory-gain law?

## Hypothesis
VQ exploits inter-dimension correlation that SQ structurally ignores. At D=2 the
memory gain is `−10·½·log₁₀(1−ρ²) = −5·log₁₀(1−ρ²)` dB (eigenvalues of
`[[1,ρ],[ρ,1]]` are `1±ρ`, so `|Σ| = 1−ρ²`). Total VQ gain should be

```
VQ gain(ρ)  =  floor (from EXP-01, ≈ 0.39 dB at R=2)  +  −5·log₁₀(1−ρ²)
```

so `VQ gain(ρ) − VQ gain(0)` should land *exactly* on `−5·log₁₀(1−ρ²)` — the
floor cancels.

## Setup
- Source: 2D Gaussian, unit variances, correlation ρ swept over `config.yaml`
- SQ: optimal Lloyd-Max, `bits_per_dim` bits on each axis
- VQ: LBG, codebook rate-matched to the same total bits (`2**(bits_per_dim * 2)` = 16)
- Metric: SQNR (dB) for each quantizer; VQ gain = SQNR_VQ − SQNR_SQ

## Hand-check anchors
- `|Σ| = 1−ρ²` → memory gain `−5·log₁₀(1−ρ²)`; at ρ=0.85 that is 2.78 dB
- ρ=0 intercept ≈ the EXP-01 floor (0.39 dB at R=2)
- predicted total at ρ=0.85 ≈ 0.39 + 2.78 = 3.2 dB
- SQNR_SQ flat across ρ (marginals are always unit-variance; SQ cannot see correlation)

## Run
```bash
pip install -e .          # once, from repo root
python experiments/exp_02_vq_gain_vs_rho/run.py
```
Output lands in `results/exp_02_vq_gain_vs_rho/results_<timestamp>/`:
`vq_gain_vs_rho.png`, `metrics.csv`, `config.snapshot.yaml`, `meta.json`.

## Definition of done
- [x] `run.py` sweep implemented
- [x] plot + `metrics.csv` generated and committed
- [ ] interpretation reframed around `gain − gain₀` vs `−5·log₁₀(1−ρ²)`
- [x] ρ=0.85 operating point recorded (3.38 dB; SQNR_SQ 9.27, SQNR_VQ 12.65)

## Result
Curve is monotonically increasing and convex; measured VQ gain tracks
`−5·log₁₀(1−ρ²)` plus the EXP-01 floor. ρ=0.85 → **3.38 dB** VQ gain over optimal
SQ. Full table and interpretation on issue #1.

_(TODO: re-plot as `gain − gain(ρ=0)` against the analytic curve now that EXP-01
has characterised the floor.)_
