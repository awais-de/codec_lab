# EXP-04b: Gaussian, quantizer-in-the-loop (control — run, abandoned)

Tracks **[#14](https://github.com/awais-de/codec_lab/issues/14)** · Stage: Ablation · **closed as not planned**

## Question
EXP-04 showed a fairness-respecting (frozen, quantizer-blind) encoder converges
to the RD-optimal linear/decorrelating transform on a Gaussian source. Does
letting the encoder train through its own real quantizer instead change that
answer at all?

## Hypothesis
It shouldn't. Decorrelation is the unconstrained RD optimum for a Gaussian
source, not an artifact of the blind training proxy -- an SQ-in-the-loop
encoder and a VQ-in-the-loop encoder should each independently land back on
~the same transform EXP-04 found (gain(signal) ~ -0.49 dB at h=4). A miss here
would mean the new training machinery does something unexpected, and that
would need explaining before trusting EXP-06b.

## Setup
- Source: 2D correlated Gaussian, ρ=0.85 (`sources.gaussian_2d`, EXP-04's config)
- Codec: autoencoder, 1 hidden layer, 4 units -- two independently trained
  instances (`codeclab.models.train_joint`)
- Training: alternating optimization -- freeze quantizer, train encoder+decoder
  via straight-through estimator, refit quantizer (Lloyd-Max + water-filling for
  SQ, LBG for VQ), repeat. Fairness condition 1 (shared latent) deliberately
  relaxed; condition 2 (matched rate, K=16) kept.

## Run
```bash
python experiments/exp_04b_gaussian_joint/run.py
```
Output → `results/exp_04b_gaussian_joint/results_<timestamp>/`: `results.json`,
`gaussian_joint_latents.png`.

## Definition of done
- [x] SQ-in-the-loop and VQ-in-the-loop encoders trained (ρ=0.85, capacity=4, K=16)
- [x] gain_signal_db compared against EXP-04 (-0.494 dB)
- [x] Latent diagnostics logged for both encoders
- [x] One-paragraph verdict: null result confirmed, or discrepancy explained

## Result
Run `results_20260917091806`. The predicted null result did not occur: latent
gain +2.98 dB, signal gain +4.50 dB, against EXP-04's fair −0.49 dB. The
SQ-in-the-loop encoder never decorrelates — its latent eigenvalues (1.83/0.19)
are nearly the source's own (1.85/0.15) and off-diagonal energy is 0.67 against
the fair pipeline's 0.011 — so water-filling splits bits evenly and SQ ends up
worse than it was on the shared latent. Seeds 0/1/2: off-diag 0.67/0.39/0.17.

**Abandoned rather than fixed.** Neither more frequent quantizer refits (20 → 400
rounds) nor a coarse-to-fine rate anneal reaches the fair pipeline's ~0.01
off-diagonal energy, and there is no monotone trend across schedules. This is an
optimization pathology of training through a coarse quantizer with a
straight-through estimator, not a finding about SQ vs VQ. EXP-06b (#15), the ring
counterpart, was dropped with it. Full write-up on issue #14.
