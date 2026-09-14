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
- [ ] Classical SQ vs VQ run
- [ ] Result recorded, becomes the reference ceiling for EXP-06/07/08

## Result
_(fill in after the run: the gain, and whether the scatter plot visibly shows VQ's
codewords sitting on the ring vs SQ's grid missing it)_
