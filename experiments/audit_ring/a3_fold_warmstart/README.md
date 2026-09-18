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
- [ ] Warm start reaches gain ≤ 1.0 dB at h=64 (h=4 attempt recorded either way)
- [ ] Trajectories logged at `rate_noise` 1.0, 3 seeds
- [ ] Noise-width arm logged
- [ ] Verdict against the pre-registered predictions, one paragraph
- [ ] Wording for the EXP-07 (#6) correction drafted from the verdict

## Result
_(fill in after the run)_
