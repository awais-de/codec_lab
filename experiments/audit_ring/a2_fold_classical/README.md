# AUDIT-02: Seamless fold — classical existence proof

Tracks **[#17](https://github.com/awais-de/codec_lab/issues/17)** · Stage: Diagnosis

## Question
Does a seamless (continuous, injective) encoder exist on the ring under which SQ reaches
VQ's ceiling — in signal space, not just latent space?

## Hypothesis
Yes. EXP-07's topological argument forbids *unrolling* the ring; SQ doesn't need it
unrolled, it needs the support to thread a product grid. A fold of the circle along a
Hamiltonian cycle of the 4×4 grid does that: both quantizers then land on the same 16
points, and the 3.29 dB gap should collapse to the floor. A discontinuous polar seam
should do the same. Together they enumerate the escape routes.

## Setup
- Source: `ring_2d` (radius 1.0, radial noise 0.1, EXP-05's config)
- Encoders (`codeclab.folds`), each with an exact inverse: `none`, `seam` (polar),
  `fold` at two perpendicular scales — *matched* (ring's own noise-to-spacing ratio,
  ~5% of points cross into the neighbouring strand) and *compressed* (scale 1.0, no
  crossings)
- Quantizer: SQ (per-axis Lloyd-Max + water-filling) vs VQ (LBG), K=16, same latent per
  row; 2 bits/dim
- No learning anywhere

Note on the inverse: `unfold_ring` is exact on the path and at the grid vertices (which
is all the quantized pipeline feeds it). Off the path near the cycle's corners it is
ambiguous — a property of a non-smooth fold, irrelevant to the quantized numbers, but it
means `unfold(fold(X))` is not an identity for raw samples.

## Anchors
- 16 ideal points on the circle: 16.454 dB at σ = 0.1 — equals EXP-05's classical VQ.
- `seam` and `fold (compressed)` should hit that for both quantizers; `fold (matched)`
  sits below it by the strand-crossing penalty, for both quantizers equally.

## Run
```bash
python experiments/audit_ring/a2_fold_classical/run.py
```
Output → `results/audit_ring/a2_fold_classical/results_<timestamp>/`: `metrics.csv`,
`results.json`, `fold_classical_sq_vs_vq.png`.

## Definition of done
- [x] `none` row reproduces EXP-05 to the printed digits
- [x] `seam` matches the 16.454 dB closed form; `fold` closes the gap but not the ceiling
- [x] Table `{none, seam, fold} × {SQ, VQ}` recorded, latent and signal space
- [x] One-paragraph interpretation: what the ring is and is not a limit of

## Result
Run `results_20260918214539`, signal space: `none` 13.16 / 16.45 dB (gain +3.29, = EXP-05);
`seam` 16.44 / 16.44 (gain 0, = ceiling); `fold` compressed 15.71 / 15.05 (gain −0.66);
`fold` matched 11.89 / 11.66 (gain −0.24, 5 % strand crossings). A continuous injective
encoder removes the gap — SQ's 16 points all land on the ring — so the ring limits
*unrolling*, not SQ. Neither quantizer reaches the ceiling under this fold: Lloyd-Max
boundaries sit at ±0.90 instead of the segment midpoints (uniform fill on the marginals)
and the fold is not an isometry at its corners (LBG codewords decode to uneven angles,
0.33–0.52 rad). Full write-up on #17.
