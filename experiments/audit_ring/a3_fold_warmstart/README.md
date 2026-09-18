# AUDIT-03: Warm-start at the fold — attributing the unfolding

Tracks **[#18](https://github.com/awais-de/codec_lab/issues/18)** · Stage: Diagnosis

## Question
Is the rate proxy what prevents trained encoders from folding the ring? AUDIT-02 showed a
fold exists under which SQ ≈ VQ; EXP-06/07's trained encoders never found one. Start a
network *at* the fold and continue training under the exact blind-proxy objective: does
it stay folded, or get dragged back to a ring/ellipse?

## Pre-registered predictions
- **Proxy is the culprit:** at `rate_noise = 1.0` the gain climbs from ~0.4 dB toward
  3–5 dB and the latent marginals lose their 4 modes; at 0.1–0.2 the fold survives. The
  transition tracks noise width vs strand spacing (~0.65 after variance conservation).
- **Not the culprit:** the fold survives at 1.0; the EXP-07 twelve-run failure was
  initialisation/optimisation and the claim narrows to "not found from random init".

## Setup
- Source: `ring_2d` (EXP-05/06/07's config); fold from AUDIT-02, compressed scale
- Networks: h=64, 2×64, and h=4 (expected unable to represent the 16-segment path)
- Warm start (`codeclab.models.warmstart.fit_to_map`): supervised fit to the fold,
  accepted only if the fitted latent gives SQ-vs-VQ gain ≤ 1.0 dB
- Continue (`continue_training`): `train_autoencoder`'s exact objective, 2000 epochs,
  lr 0.01, logging gain / participation ratio / marginal modes every 100 epochs;
  `rate_noise ∈ {0.1, 0.2, 0.5, 1.0}`, 3 seeds
- Quantizers: SQ vs VQ, K=16, freeze-then-swap as on the main ladder

## Run
```bash
python experiments/audit_ring/a3_fold_warmstart/run.py
```
Output → `results/audit_ring/a3_fold_warmstart/results_<timestamp>/`: `results.json`,
`trajectories.csv`, `gain_vs_epoch.png`, `latent_before_after.png`.

## Definition of done
- [x] Warm start reaches gain ≤ 1.0 dB at h=64 (h=4 cannot represent the fold: +2.1–2.4 dB)
- [x] Trajectories logged at `rate_noise` 1.0, 3 seeds (2 for h=64)
- [x] Noise-width arm logged
- [x] Verdict against the pre-registered predictions
- [x] Wording for the EXP-07 (#6) correction drafted from the verdict

## Result
Run `results_20260918221754`. Neither pre-registered branch fits exactly. At width 1.0
the objective drives a fold-started network all the way to EXP-07's flat ellipse (gain
4.5–4.9 dB, PR 1.6–1.7) — the objective is the culprit at the main ladder's setting. At
widths ≤ 0.5 the fold is only partly inflated into a dented loop and settles at
0.65–2.6 dB, well below the classical 3.29 — the objective tolerates a partially folded
encoder that random-init training never finds. One mechanism at all widths: the
additive-noise proxy rewards strand separation, a fold needs strand proximity, pressure
scales with noise width. The first-100-epoch jump in every arm is a fresh-optimizer
transient (cross-checked at lr 0.001, same endpoints). Full write-up on #18.
