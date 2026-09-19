# EXP-05: Classical SQ vs VQ baseline on a ring source

Tracks **[#13](https://github.com/awais-de/codec_lab/issues/13)** · Stage: Reference (classical baseline for the neural stage)

## Question
On a source whose optimal transform is not linear, how much does VQ beat SQ?

## Hypothesis
The ring's covariance is isotropic by construction (`Σ ≈ 0.5·I`, printed and checked
by the script, not just asserted) — a rotation gives zero benefit. Any VQ advantage
here can't be memory gain; it's VQ placing codewords on the ring itself while SQ's
axis-aligned grid can't.

## Setup
- Source: 2D ring, radius 1.0, radial jitter 0.1 (`sources.ring_2d`)
- SQ: per-axis Lloyd-Max, 2 bits/dim
- VQ: LBG, K=16 (rate-matched)

## Run
```bash
python experiments/exp_05_ring_baseline/run.py
```
Output → `results/exp_05_ring_baseline/results_<timestamp>/`: `metrics.csv`,
`ring_sq_vs_vq.png` (source + reconstruction scatter, SQ vs VQ side by side).

## Definition of done
- [x] `codeclab.sources.ring_2d` implemented
- [x] Classical SQ vs VQ run
- [x] Result recorded, becomes the reference ceiling for EXP-06/07/08

## Result
Run `results_20260914202537`. SQ 13.164 dB, VQ 16.454 dB, **gain +3.290 dB**.
The printed source covariance `[[0.507, -0.002], [-0.002, 0.502]]` is isotropic
as predicted, so none of this is memory gain. The scatter shows VQ's codewords
sitting on the ring while SQ's grid points mostly miss it.

Anchor added later by AUDIT-02 (#17): 16.454 dB is exactly the closed form for
16 ideal points on the circle, `10·log₁₀[((1+σ²)/2) / ((σ² + (2π/16)²/12)/2)]`
at σ=0.1 — classical VQ reaches that ceiling here with no encoder at all.
Full write-up on issue #13.
