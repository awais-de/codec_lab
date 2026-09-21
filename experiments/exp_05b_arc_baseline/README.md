# EXP-05b: Classical SQ vs VQ on the arc family

Tracks **[#19](https://github.com/awais-de/codec_lab/issues/19)** · Stage: Reference

## Question
How do the classical, no-encoder SQ-vs-VQ numbers move as a curve closes from
nearly-linear to a ring, and what is the best any encoder could do at each angle?
Generalises EXP-05: the ring is the α=360° special case.

## Hypothesis
`gain(no transform)` rises with α — a shallow arc is nearly a line, so there is
little beyond the granular floor for VQ to exploit; as the arc closes the gain
climbs toward the ring's 3.29 dB. `gain(PCA + allocation)` rises too but stays
below it, since PCA captures only the linear part, and at α=360° PCA is a no-op
so the two must coincide.

## Setup
- Source: `sources.arc_2d` — θ ~ U(0, α), r = 1.0 + N(0, 0.1), centred on the
  analytic mean; α = 60°, 200°, 340°, 360°
- Conditions: `{no transform, PCA + bit allocation, ideal straightening} × {SQ, VQ}`
- Quantizer: SQ (per-axis Lloyd-Max + water-filling) vs VQ (LBG), K=16, 2 bits/dim
- No networks; deterministic, single seed

## Run
```bash
python experiments/exp_05b_arc_baseline/run.py
```
Output → `results/exp_05b_arc_baseline/results_<timestamp>/`: `metrics.csv`,
`results.json`, `gain_vs_angle.png`, `arc_sources.png`.

## Definition of done
- [x] `sources.arc_2d` implemented; α=360° reproduces `ring_2d` (to 5.6e-17; see Result)
- [x] Closed-form covariance validated numerically at each α
- [x] `{no transform, PCA+alloc} × {SQ, VQ}` table at α = 60/200/340/360
- [x] α=360° row reproduces EXP-05 (13.164 / 16.454 / 3.290)
- [x] Straightened-arc ceiling computed per angle
- [x] One-paragraph interpretation
- [x] Numbers recorded as the reference for the neural capacity sweep

## Result
Run `results_20260921170445`. VQ gain (dB), signal space:

| α | no transform | PCA + alloc | ideal straightening | PCA bits | straight bits |
|---|---|---|---|---|---|
| 60° | +1.955 | +0.614 | **+0.260** | [3,1] | [3,1] |
| 200° | +3.954 | +4.462 | **+0.004** | [3,1] | [4,0] |
| 340° | +3.421 | +3.298 | **+0.003** | [2,2] | [4,0] |
| 360° | +3.290 | +3.321 | **+0.002** | [2,2] | [4,0] |

α=360° reproduces EXP-05 on all three plain-condition numbers. Closed-form
covariance matches the sample at every angle (max error 4e-4 … 4e-3, sampling
noise), and gives exactly `0.505·I` at 360°. The predicted allocation shift
([3,1] at 60° → [4,0] at 340°) is confirmed.

**Both shape predictions were wrong, for separable reasons.** `gain(no transform)`
is *non-monotone*, peaking at 200°. Decomposing it with PCA shows why: the memory
component (`gain_plain − gain_pca`) falls monotonically with α — 1.341, —, 0.123,
−0.031 — tracking the anisotropy, which vanishes at 360° by symmetry; while the
manifold component (`gain_pca`) rises and saturates — 0.614 → 3.298 → 3.321. The
sum peaks in between. The 200° PCA point is contaminated: its eigenvalue ratio
4.04 wants a (2.5, 1.5) split and greedy integer allocation gives (3,1),
over-feeding the dominant axis — the same granularity artifact EXP-03 recorded at
ρ ≥ 0.9, and the reason PCA there makes the gain *larger*, not smaller.

**The ceiling is essentially zero for every open arc.** Straightening collapses
the gain to 0.002–0.004 dB at 200°/340°/360° — SQ matches VQ outright — and to
0.260 dB at 60°, where the allocation keeps one bit on the radial axis and leaves
VQ a little 2-D structure to exploit.

**The structural result: the straightening map's Lipschitz constant diverges as
α → 360°.** Measured as the largest `|Δz|/|Δx|` over random sample pairs:

| α | 60° | 200° | 340° | 360° |
|---|---|---|---|---|
| max `|Δz|/|Δx|` | 1.46 | 2.22 | 17.98 | 161.66 |
| max `|Δz|` among signal-neighbours (`|Δx|` < 0.05) | 0.061 | 0.058 | 0.062 | **6.273** |

For every open arc the map is continuous with a bounded constant that grows as the
gap narrows; at 360° it becomes a cut — signal-space neighbours are thrown 6.27
apart. The α=360° "straightening" row *is* AUDIT-02's polar seam, and reproduces
its number (16.441 vs 16.44 dB). So the ceiling is equally low everywhere, but its
*reachability* is not: reachable by a continuous map for α < 360°, only by a
discontinuous one at 360°. That is the distinction the arc family exists to expose,
and it gives EXP-08 a sharper prediction than "capacity climbs as the arc closes" —
the network must approximate a map whose Lipschitz constant is diverging.

Full write-up on issue #19.
