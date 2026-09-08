# EXP-01: Memoryless SQ vs VQ — the granular floor

Tracks **[#11](https://github.com/awais-de/codec_lab/issues/11)** · Stage: Reference · Rung 1 of the classical SQ-vs-VQ ladder

## Question
With an i.i.d. source — nothing for VQ's memory gain to exploit — how much can a
vector quantizer still beat optimal scalar quantization, and what is that
residual made of?

## Background: the three classical gains
VQ's advantage over SQ factors into three independent parts (Lookabaugh & Gray, 1989):

| gain | where it comes from | i.i.d. source? |
|------|---------------------|----------------|
| memory | correlation between dimensions | **0** — nothing to exploit |
| shape | shape of the marginal pdf | 0 for uniform, > 0 for Gaussian |
| space-filling | granular cell geometry (hexagon vs square) | present, source-independent |

Rung 1 isolates the bottom two — the **granular floor** — by killing memory gain.
Rung 2 (EXP-02, the ρ sweep) turns correlation back on and watches memory gain
appear on top of this floor.

## Hypothesis
- VQ gain on a memoryless source is small (tenths of a dB) and roughly rate-independent.
- **Uniform source:** flat pdf ⇒ zero shape gain ⇒ VQ gain is *pure space-filling*,
  approaching `10·log₁₀[(1/12)/(5/36√3)] = 0.167 dB` (hexagonal vs square) as R grows.
  At low rate it sits below that — the square domain boundary dilutes the interior
  hexagonal packing.
- **Gaussian source:** VQ gain = space-filling + shape. Subtracting the uniform
  result isolates the shape gain (positive, growing slightly with R).

## Setup
- Sources: 2D i.i.d., zero-mean unit-variance — Gaussian (`gaussian_2d`, ρ=0) and
  uniform (`uniform_2d`, each axis U(−√3, √3))
- SQ: per-axis Lloyd–Max, R bits/axis
- VQ: LBG, codebook rate-matched to `K = 2**(2R)`
- Rate sweep R = 2, 3, 4 bits/dim (`config.yaml`; extend with 5 for a tighter
  space-filling estimate — slower)

## Hand-check anchors
- Uniform SQ: SQNR = `6.02·R` dB exactly (no companding)
- Gaussian SQ: Lloyd–Max table — 9.30 / 14.62 / 20.22 dB at R = 2 / 3 / 4
- Uniform VQ gain → 0.167 dB, approached from below
- All granular gains ≈ rate-independent near the top of the sweep

## Run
```bash
pip install -e .          # once, from repo root
python experiments/exp_01_memoryless_floor/run.py
```
Output → `results/exp_01_memoryless_floor/results_<timestamp>/`:
`granular_floor.png`, `sq_sanity.png`, `metrics.csv`, `decomposition.csv`,
`config.snapshot.yaml`, `meta.json`.

## Definition of done
- [ ] `run.py` sweep implemented
- [ ] plots + `metrics.csv` + `decomposition.csv` generated and committed
- [ ] one-paragraph interpretation added below
- [ ] space-filling and shape numbers recorded for EXP-02 to build on

## Result
_(fill in after the run: the floor value and its split into space-filling vs
shape, whether the uniform gain tracks 0.167 dB, whether SQ matches the closed
forms)_
